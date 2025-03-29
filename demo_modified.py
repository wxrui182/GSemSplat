#!/usr/bin/env python3
# The MASt3R Gradio demo, modified for predicting 3D Gaussian Semantic Splats

# --- Original License ---
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import sys
import workspace
import numpy as np
import einops
import cv2
from einops import rearrange, repeat
from pathlib import Path
from sklearn.decomposition import PCA, IncrementalPCA
from PIL import Image, ImageEnhance
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt

import torch
import torchvision

sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')
sys.path.append('src/pixelsplat_src')
from dust3r.utils.image import load_images
from src.pixelsplat_src.cuda_splatting import render_cuda
import main_modified_deeper_conv1
import utils.export as export
from utils.geometry import normalize_intrinsics
from utils_activate import colormap_saving
import colormaps
from openclip_encoder import OpenCLIPNetwork
from utils.export import save_as_ply, save_3d

class CameraParams:
    def __init__(self, H: int = 512, W: int = 512):
        self.H = H
        self.W = W
        self.focal = (5.8269e+02, 5.8269e+02)
        self.fov = (2*np.arctan(self.W / (2*self.focal[0])), 2*np.arctan(self.H / (2*self.focal[1])))
        self.K = np.array([
            [self.focal[0], 0., self.W/2],
            [0., self.focal[1], self.H/2],
            [0.,            0.,       1.],
        ]).astype(np.float32)

def generate_seed_newpreset():
    degsum = 60
    thlist = np.concatenate((np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:], np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:]))
    philist = np.concatenate((np.linspace(0,0,7), np.linspace(22.5,22.5,7)))

    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.zeros((3,1))
    return render_poses

def generate_seed_llff(degree, nviews, round=4, d=2.3):
    assert round%4==0
    thlist = degree * np.sin(np.linspace(0, 2*np.pi*round, nviews))
    philist = degree * np.cos(np.linspace(0, 2*np.pi*round, nviews))
    zlist = d/15 * np.sin(np.linspace(0, 2*np.pi*round//4, nviews))
    assert len(thlist) == len(philist)
 
    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        z = zlist[i]
       
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.array([d*np.sin(th/180*np.pi), 0, -z+d-d*np.cos(th/180*np.pi)]).reshape(3,1) + np.array([0, d*np.sin(phi/180*np.pi), -z+d-d*np.cos(phi/180*np.pi)]).reshape(3,1)# Transition vector
    return render_poses

def orbit_camera_torch(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    """construct a camera pose matrix orbiting a target with elevation & azimuth angle.
 
    Args:
        elevation (float): elevation in (-90, 90), from +y to -y is (-90, 90)
        azimuth (float): azimuth in (-180, 180), from +z to +x is (0, 90)
        radius (int, optional): camera radius. Defaults to 1.
        is_degree (bool, optional): if the angles are in degree. Defaults to True.
        target (np.ndarray, optional): look at target position. Defaults to None.
        opengl (bool, optional): whether to use OpenGL camera convention. Defaults to True.
 
    Returns:
        np.ndarray: the camera pose matrix, float [4, 4]
    """
   
    if is_degree:
        elevation = torch.deg2rad(elevation)
        azimuth = torch.deg2rad(azimuth)
    x = radius * torch.cos(elevation) * torch.sin(azimuth)
    y = - radius * torch.sin(elevation)
    z = radius * torch.cos(elevation) * torch.cos(azimuth)
    if target is None:
        target = torch.zeros([3]).float().cuda()
    campos = torch.stack([x, y, z]) + target  # [3]
    T = torch.eye(4).float().cuda()
    T[:3, :3] = look_at_torch(campos, target, opengl)
    T[:3, 3] = campos
    return T
 
def look_at_torch(campos, target, opengl=True):
    """construct pose rotation matrix by look-at.
 
    Args:
        campos (torch.Tensor): camera position, float [3]
        target (torch.Tensor): look at target, float [3]
        opengl (bool, optional): whether use opengl camera convention (forward direction is target --> camera). Defaults to True.
 
    Returns:
        torch.Tensor: the camera pose rotation matrix, float [3, 3], normalized.
    """
 
    def safe_normalize(v):
        norm = torch.norm(v, p=2, dim=0)
        return v / norm if norm > 0 else v
 
    up_vector = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=campos.device)
 
    if not opengl:
        # forward is camera --> target
        forward_vector = safe_normalize(target - campos)
        right_vector = safe_normalize(torch.cross(forward_vector, up_vector))
        up_vector = safe_normalize(torch.cross(right_vector, forward_vector))
    else:
        # forward is target --> camera
        forward_vector = safe_normalize(campos - target)
        right_vector = safe_normalize(torch.cross(up_vector, forward_vector))
        up_vector = safe_normalize(torch.cross(forward_vector, right_vector))
 
    R = torch.stack([right_vector, up_vector, forward_vector], dim=1)
    # print(R.grad)
    return R

def activate_stream(sem_map,
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    batch_idx=0, 
                    v_idx=0,  
                    colormap_options = None):

    valid_map = clip_model.get_max_across(sem_map)                 
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    for k in range(n_prompt):
        for i in range(n_head):

            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)   
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])    

            
            output_path_relev = image_name / 'heatmap' / f'batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}_{i}'   
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options, output_path_relev)
            
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 1.0

            output_path_compo = image_name / 'composited' / f'batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)  

    flag = "processed all"

    return flag

if __name__ == '__main__':
    save_dir = Path("demo_results")
    save_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        image_size = 512
        silent = False
        s3l_ckpt_path = "path_to_checkpoint/epoch=11-step=66900.ckpt"
        config_path = "path_to_configs/main_modified.yaml"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        colormap_options = colormaps.ColormapOptions(   
            colormap="turbo",
            normalize=True,
            colormap_min=-1.0,
            colormap_max=1.0,
        )
        pca = PCA(n_components=3)

        filelist = ["demo_examples/in_the_wild_27_img_1.jpg", "demo_examples/in_the_wild_2_img_2.jpg"]

        imgs = load_images(filelist, size=image_size, verbose=not silent)

        idx = 0
        for img in imgs:
            img['img'] = img['img'].to(device)
            img['original_img'] = img['original_img'].to(device)
            img['true_shape'] = torch.from_numpy(img['true_shape'])
            torchvision.utils.save_image(img['original_img'], os.path.join(save_dir, f"original_img_{idx}.jpg"))
            idx += 1
        

        clip_model = OpenCLIPNetwork(device)
        checkpoint = torch.load(s3l_ckpt_path, map_location=device)
        config = workspace.load_config(config_path)
        model = main_modified_deeper_conv1.MAST3RGaussians(config).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        output = model(imgs[0], imgs[1])
        pred1, pred2 = output
        # print("Keys in pred1:", pred1.keys())
        # print("Keys in pred2:", pred2.keys())
        # save_as_ply(pred1, pred2, os.path.join(save_dir, f"gaussians.ply"))

        # render_poses = generate_seed_newpreset()
        # render_poses = generate_seed_llff(5, 400, round=4, d=2)
        # render_poses = render_poses[:20]
        # render_poses = render_poses[::20][:20]
        ver = torch.tensor(0).float().cuda()
        # hor = torch.tensor(40).float().cuda()
        rad = torch.tensor(0.0001).float().cuda()
        # pose = orbit_camera_torch(ver, hor, rad).unsqueeze(0)

        poses = []
        
        for hor_value in range(-16, 20, 2):  
            hor = torch.tensor(hor_value).float().cuda()
            pose = orbit_camera_torch(ver, hor, rad).unsqueeze(0)
            poses.append(pose)
        
        poses = torch.cat(poses, dim=0)  # [num_poses, ...]

        num_poses = poses.shape[0]
        
        camera_params = CameraParams(H=image_size, W=image_size)
        intrinsics = torch.tensor(camera_params.K).unsqueeze(0).unsqueeze(1).expand(1, num_poses, -1, -1).to(device)
        intrinsics = normalize_intrinsics(intrinsics, (image_size, image_size))[..., :3, :3]
        
        extrinsics = poses.unsqueeze(0).to(device)
        
        
        means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
        covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
        harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)
        opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)
        language_feature_precomp = torch.stack([pred1["clip_features"], pred2["clip_features"]], dim=1)

        b, v, _, _ = extrinsics.shape
        near = torch.full((b, v), 0.1, device=device)
        far = torch.full((b, v), 1000.0, device=device)

        color, semantics = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (image_size, image_size),
            repeat(torch.tensor([1.0, 1.0, 1.0], device=device), "c -> (b v) c", b=b, v=v),
            repeat(rearrange(means, "b v h w xyz -> b (v h w) xyz"), "b g xyz -> (b v) g xyz", v=v),
            repeat(rearrange(covariances, "b v h w i j -> b (v h w) i j"), "b g i j -> (b v) g i j", v=v),
            repeat(rearrange(harmonics, "b v h w c d_sh -> b (v h w) c d_sh"), "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(rearrange(opacities, "b v h w 1 -> b (v h w)"), "b g -> (b v) g", v=v),
            repeat(rearrange(language_feature_precomp, "b v h w dim -> b (v h w) dim"), "b g dim -> (b v) g dim", v=v),
        )

        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        semantics = rearrange(semantics, "(b v) c h w -> b v c h w", b=b, v=v)
        print("Rendering complete.")

        b, v, _, _, _ = semantics.shape
        semantics = einops.rearrange(semantics, "b v c h w -> (b v) c h w")

        semantics_sam = semantics[:, :16, :, :]
        for layer in model.language_decoder_sam:
            semantics_sam = layer(semantics_sam)
        semantics_sam = einops.rearrange(semantics_sam, "(b v) c h w -> b v c h w", b=b, v=v)

        semantics_patch = semantics[:, 16:32, :, :]
        for layer in model.language_decoder_patch:
            semantics_patch = layer(semantics_patch)
        semantics_patch = einops.rearrange(semantics_patch, "(b v) c h w -> b v c h w", b=b, v=v)

        for b_idx in range(color.shape[0]):  
            for v_idx in range(color.shape[1]):     
                rendered_color = color[b_idx, v_idx]  # Shape: [3, h, w]
                color_save_path = save_dir / f"sample_{b_idx}_view_{v_idx}_rendered_color.jpg"
                torchvision.utils.save_image(rendered_color, color_save_path)

        for b_idx in range(color.shape[0]):

            img_ann_keys = ["calculator"]   # query texts
            clip_model.set_positives(img_ann_keys)
            
            for v_idx in range(color.shape[1]):
                    
                    img_path = save_dir / f"sample_{b_idx}_view_{v_idx}_rendered_color.jpg"
                    rgb_img = cv2.imread(str(img_path))[..., ::-1]
                    rgb_img = (rgb_img / 255.0).astype(np.float32)
                    rgb_img = torch.from_numpy(rgb_img).to(device)

                    restored_feat_sam = semantics_sam[b_idx, v_idx]
                    restored_feat_sam = restored_feat_sam.permute(1, 2, 0)  
                    restored_feat_sam = restored_feat_sam.unsqueeze(0)   

                    restored_feat_patch = semantics_patch[b_idx, v_idx]
                    restored_feat_patch = restored_feat_patch.permute(1, 2, 0)  
                    restored_feat_patch = restored_feat_patch.unsqueeze(0)      

                    restored_feat = torch.cat((restored_feat_sam, restored_feat_patch), dim=0)

                    flag = activate_stream(restored_feat, rgb_img, clip_model, save_dir, batch_idx=b_idx, v_idx=v_idx, colormap_options=colormap_options)
    

        color = color.squeeze(0)  
        v, c, h, w = color.shape
        output_path = save_dir / 'rgb_video.mp4'
        fps = 16  
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for i in range(v):
            frame = color[i].to('cpu').permute(1, 2, 0).numpy()  
            frame = (frame * 255).astype('uint8')
            frame = frame[..., ::-1]
            video_writer.write(frame)

        video_writer.release()
        print(f"RGB Video saved to {output_path}")
        
        
        b, v, c, h, w = semantics_sam.shape
        semantics_sam = semantics_sam / (semantics_sam.norm(dim=2, keepdim=True) + 1e-9)  
        n_components = 3  

        batch_size_v = 1  

        ipca = IncrementalPCA(n_components=n_components, batch_size=1024)

        
        for i in range(b):  
            for j in range(0, v, batch_size_v):  
                batch_data = semantics_sam[i, j:j + batch_size_v].permute(0, 2, 3, 1).reshape(-1, c)
                ipca.partial_fit(batch_data.cpu().numpy())

        semantics_sam_pca = []  
        for i in range(b):  
            for j in range(0, v, batch_size_v):
                
                batch_data = semantics_sam[i, j:j + batch_size_v].permute(0, 2, 3, 1).reshape(-1, c)
                reduced_data = ipca.transform(batch_data.cpu().numpy())  
                reduced_data = torch.tensor(reduced_data).reshape(batch_size_v, h, w, n_components).permute(0, 3, 1, 2)
                semantics_sam_pca.append(reduced_data)

        semantics_sam_pca = torch.cat(semantics_sam_pca, dim=0)
        semantics_sam_pca = semantics_sam_pca.squeeze(0)
        v, c, h, w = semantics_sam_pca.shape

        output_path = save_dir / 'pca_video.mp4'
        fps = 16
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        pca_image_save_dir = save_dir / "pca_images"
        pca_image_save_dir.mkdir(parents=True, exist_ok=True)

        for i in range(v):  
            frame = semantics_sam_pca[i].permute(1, 2, 0).cpu().numpy()  # [3, h, w] -> [h, w, 3]
            frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype('uint8')
            frame = frame[..., ::-1]  
            video_writer.write(frame)  

            pca_image_path = pca_image_save_dir / f"frame_{i}.png"
            cv2.imwrite(str(pca_image_path), frame)
            print(f"Saved frame {i} to {pca_image_path}")

        video_writer.release()
        print(f"Semantics Video saved to {output_path}")