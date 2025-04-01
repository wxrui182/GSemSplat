import os

from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
import einops
import numpy as np
import torch
import torchvision
import trimesh
import lightning as L
from sklearn.decomposition import PCA

import utils.loss_mask as loss_mask
from src.mast3r_src.dust3r.dust3r.viz import OPENGL, pts3d_to_trimesh, cat_meshes


class SaveBatchData(L.Callback):
    '''A Lightning callback that occasionally saves batch inputs and outputs to disk.
    It is not critical to the training process, and can be disabled if unwanted.'''

    def __init__(self, save_dir, train_save_interval=100, val_save_interval=100, test_save_interval=100):
        self.save_dir = save_dir
        self.train_save_interval = train_save_interval
        self.val_save_interval = val_save_interval
        self.test_save_interval = test_save_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.train_save_interval == 0 and trainer.global_rank == 0:
            self.save_batch_data('train', trainer, pl_module, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.val_save_interval == 0 and trainer.global_rank == 0:
            self.save_batch_data('val', trainer, pl_module, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.test_save_interval == 0 and trainer.global_rank == 0:
            self.save_batch_data('test', trainer, pl_module, batch, batch_idx)

    def save_batch_data(self, prefix, trainer, pl_module, batch, batch_idx):
        with torch.no_grad():
            print(f'Saving {prefix} data at epoch {trainer.current_epoch} and batch {batch_idx}')

            # Run the batch through the model again
            _, _, h, w = batch["context"][0]["img"].shape
            view1, view2 = batch['context']
            pred1, pred2 = pl_module.forward(view1, view2)
            color, semantics, depth = pl_module.decoder(batch, pred1, pred2, (h, w))

            b_sem, v_sem, _, _, _ = semantics.shape
            semantics_upscale = einops.rearrange(semantics, "b v c h w -> (b v) c h w")

            semantics_upscale_sam = semantics_upscale[:, :16, :, :]
            for layer in pl_module.language_decoder_sam:
                semantics_upscale_sam = layer(semantics_upscale_sam)
            semantics_upscale_sam = einops.rearrange(semantics_upscale_sam, "(b v) c h w -> b v c h w", b=b_sem, v=v_sem)

            semantics_upscale_patch = semantics_upscale[:, 16:32, :, :]
            for layer in pl_module.language_decoder_patch:
                semantics_upscale_patch = layer(semantics_upscale_patch)
            semantics_upscale_patch = einops.rearrange(semantics_upscale_patch, "(b v) c h w -> b v c h w", b=b_sem, v=v_sem)

            mask = loss_mask.calculate_loss_mask(batch)

            # Save the data
            save_dir = os.path.join(
                self.save_dir,
                f"{prefix}_epoch_{trainer.current_epoch}_batch_{batch_idx}"
            )

            log_batch_files(batch, color, semantics, semantics_upscale_sam, semantics_upscale_patch, depth, mask, view1, view2, pred1, pred2, save_dir)


def save_as_ply(pred1, pred2, save_path):
    """Save the 3D Gaussians as a point cloud in the PLY format.
    Adapted loosely from PixelSplat"""

    def construct_list_of_attributes(num_rest: int) -> list[str]:
        '''Construct a list of attributes for the PLY file format. This
        corresponds to the attributes used by online readers, such as
        https://niujinshuchong.github.io/mip-splatting-demo/index.html'''
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(3):
            attributes.append(f"f_dc_{i}")
        for i in range(num_rest):
            attributes.append(f"f_rest_{i}")
        attributes.append("opacity")
        for i in range(3):
            attributes.append(f"scale_{i}")
        for i in range(4):
            attributes.append(f"rot_{i}")
        # for i in range(49):
            # attributes.append(f"semantic_{i}")
        return attributes

    def covariance_to_quaternion_and_scale(covariance):
        '''Convert the covariance matrix to a four dimensional quaternion and
        a three dimensional scale vector'''

        # Perform singular value decomposition
        U, S, V = torch.linalg.svd(covariance)

        # The scale factors are the square roots of the eigenvalues
        scale = torch.sqrt(S)
        scale = scale.detach().cpu().numpy()

        # The rotation matrix is U*Vt
        rotation_matrix = torch.bmm(U, V.transpose(-2, -1))
        rotation_matrix_np = rotation_matrix.detach().cpu().numpy()

        # Use scipy to convert the rotation matrix to a quaternion
        rotation = Rotation.from_matrix(rotation_matrix_np)
        quaternion = rotation.as_quat()

        return quaternion, scale

    # Collect the Gaussian parameters
    means = torch.stack([pred1["pts3d"], pred2["pts3d_in_other_view"]], dim=1)
    covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
    harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)[..., 0]  # Only use the first harmonic
    opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)
    # semantics = torch.stack([pred1["clip_features"], pred2["clip_features"]], dim=1)

    # Rearrange the tensors to the correct shape
    means = einops.rearrange(means[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    covariances = einops.rearrange(covariances[0], "v h w i j -> (v h w) i j")
    harmonics = einops.rearrange(harmonics[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    opacities = einops.rearrange(opacities[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    # semantics = einops.rearrange(semantics[0], "view h w c -> (view h w) c").detach().cpu().numpy()

    # Convert the covariance matrices to quaternions and scales
    rotations, scales = covariance_to_quaternion_and_scale(covariances)

    # Construct the attributes
    rest = np.zeros_like(means)
    attributes = np.concatenate((means, rest, harmonics, opacities, np.log(scales), rotations), axis=-1)
    # attributes = np.concatenate((means, rest, harmonics, opacities, np.log(scales), rotations, semantics), axis=-1)
    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    # Save the point cloud
    point_cloud = PlyElement.describe(elements, "vertex")
    scene = PlyData([point_cloud])
    scene.write(save_path)


def save_3d(view1, view2, pred1, pred2, save_dir, as_pointcloud=True, all_points=True):
    """Save the 3D points as a point cloud or as a mesh. Adapted from DUSt3R"""

    os.makedirs(save_dir, exist_ok=True)
    batch_size = pred1["pts3d"].shape[0]
    views = [view1, view2]

    for b in range(batch_size):

        pts3d = [pred1["pts3d"][b].cpu().numpy()] + [pred2["pts3d_in_other_view"][b].cpu().numpy()]
        imgs = [einops.rearrange(view["original_img"][b], "c h w -> h w c").cpu().numpy() for view in views]
        mask = [view["valid_mask"][b].cpu().numpy() for view in views]

        # Treat all pixels as valid, because we want to render the entire viewpoint
        if all_points:
            mask = [np.ones_like(m) for m in mask]

        # Construct the scene from the 3D points as a point cloud or as a mesh
        scene = trimesh.Scene()
        if as_pointcloud:
            pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
            col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
            pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
            scene.add_geometry(pct)
            save_path = os.path.join(save_dir, f"{b}.ply")
        else:
            meshes = []
            for i in range(len(imgs)):
                meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
            mesh = trimesh.Trimesh(**cat_meshes(meshes))
            scene.add_geometry(mesh)
            save_path = os.path.join(save_dir, f"{b}.glb")

        # Save the scene
        scene.export(file_obj=save_path)


# @torch.no_grad()
# def log_batch_files(batch, color, depth, mask, view1, view2, pred1, pred2, save_dir, should_save_3d=False):
#     '''Save all the relevant debug files for a batch'''

#     os.makedirs(save_dir, exist_ok=True)

#     # Save the 3D Gaussians as a .ply file
#     save_as_ply(pred1, pred2, os.path.join(save_dir, f"gaussians.ply"))

#     # Save the 3D points as a point cloud and as a mesh (disabled)
#     if should_save_3d:
#         save_3d(view1, view2, pred1, pred2, os.path.join(save_dir, "3d_mesh"), as_pointcloud=False)
#         save_3d(view1, view2, pred1, pred2, os.path.join(save_dir, "3d_pointcloud"), as_pointcloud=True)

#     # Save the color, depth and valid masks for the input context images
#     context_images = torch.stack([view["img"] for view in batch["context"]], dim=1)
#     context_original_images = torch.stack([view["original_img"] for view in batch["context"]], dim=1)
#     context_depthmaps = torch.stack([view["depthmap"] for view in batch["context"]], dim=1)
#     context_valid_masks = torch.stack([view["valid_mask"] for view in batch["context"]], dim=1)
#     for b in range(min(context_images.shape[0], 4)):
#         torchvision.utils.save_image(context_images[b], os.path.join(save_dir, f"sample_{b}_img_context.jpg"))
#         torchvision.utils.save_image(context_original_images[b], os.path.join(save_dir, f"sample_{b}_original_img_context.jpg"))
#         torchvision.utils.save_image(context_depthmaps[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_depthmap.jpg"), normalize=True)
#         torchvision.utils.save_image(context_valid_masks[b, :, None, ...].float(), os.path.join(save_dir, f"sample_{b}_valid_mask_context.jpg"), normalize=True)

#     # Save the color and depth images for the target images
#     target_original_images = torch.stack([view["original_img"] for view in batch["target"]], dim=1)
#     target_depthmaps = torch.stack([view["depthmap"] for view in batch["target"]], dim=1)
#     context_valid_masks = torch.stack([view["valid_mask"] for view in batch["context"]], dim=1)
#     for b in range(min(target_original_images.shape[0], 4)):
#         torchvision.utils.save_image(target_original_images[b], os.path.join(save_dir, f"sample_{b}_original_img_target.jpg"))
#         torchvision.utils.save_image(target_depthmaps[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_depthmap_target.jpg"), normalize=True)

#     # Save the rendered images and depths
#     for b in range(min(color.shape[0], 4)):
#         torchvision.utils.save_image(color[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_color.jpg"))
#     if depth is not None:
#         for b in range(min(color.shape[0], 4)):
#             torchvision.utils.save_image(depth[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_rendered_depth.jpg"), normalize=True)

#     # Save the loss masks
#     for b in range(min(mask.shape[0], 4)):
#         torchvision.utils.save_image(mask[b, :, None, ...].float(), os.path.join(save_dir, f"sample_{b}_loss_mask.jpg"), normalize=True)

#     # Save the masked target and rendered images
#     target_original_images = torch.stack([view["original_img"] for view in batch["target"]], dim=1)
#     masked_target_original_images = target_original_images * mask[..., None, :, :]
#     masked_predictions = color * mask[..., None, :, :]
#     for b in range(min(target_original_images.shape[0], 4)):
#         torchvision.utils.save_image(masked_target_original_images[b], os.path.join(save_dir, f"sample_{b}_masked_original_img_target.jpg"))
#         torchvision.utils.save_image(masked_predictions[b], os.path.join(save_dir, f"sample_{b}_masked_rendered_color.jpg"))

@torch.no_grad()
def log_batch_files(batch, color, semantics, semantics_upscale, semantics_upscale_patch, depth, mask, view1, view2, pred1, pred2, save_dir, should_save_3d=False):
    '''Save all the relevant debug files for a batch'''

    os.makedirs(save_dir, exist_ok=True)

    # Save the 3D Gaussians as a .ply file
    # save_as_ply(pred1, pred2, os.path.join(save_dir, f"gaussians.ply"))

    # Save the 3D points as a point cloud and as a mesh (disabled)
    if should_save_3d:
        save_3d(view1, view2, pred1, pred2, os.path.join(save_dir, "3d_mesh"), as_pointcloud=False)
        save_3d(view1, view2, pred1, pred2, os.path.join(save_dir, "3d_pointcloud"), as_pointcloud=True)

    # Save the color, depth and valid masks for the input context images
    context_images = torch.stack([view["img"] for view in batch["context"]], dim=1)
    context_original_images = torch.stack([view["original_img"] for view in batch["context"]], dim=1)
    context_depthmaps = torch.stack([view["depthmap"] for view in batch["context"]], dim=1)
    context_valid_masks = torch.stack([view["valid_mask"] for view in batch["context"]], dim=1)
    for b in range(min(context_images.shape[0], 4)):
        torchvision.utils.save_image(context_images[b], os.path.join(save_dir, f"sample_{b}_img_context.jpg"))
        torchvision.utils.save_image(context_original_images[b], os.path.join(save_dir, f"sample_{b}_original_img_context.jpg"))
        torchvision.utils.save_image(context_depthmaps[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_depthmap.jpg"), normalize=True)
        torchvision.utils.save_image(context_valid_masks[b, :, None, ...].float(), os.path.join(save_dir, f"sample_{b}_valid_mask_context.jpg"), normalize=True)

    # Save the color and depth images for the target images
    target_original_images = torch.stack([view["original_img"] for view in batch["target"]], dim=1)
    target_depthmaps = torch.stack([view["depthmap"] for view in batch["target"]], dim=1)
    context_valid_masks = torch.stack([view["valid_mask"] for view in batch["context"]], dim=1)
    for b in range(min(target_original_images.shape[0], 4)):
        torchvision.utils.save_image(target_original_images[b], os.path.join(save_dir, f"sample_{b}_original_img_target.jpg"))
        torchvision.utils.save_image(target_depthmaps[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_depthmap_target.jpg"), normalize=True)

    # Save the rendered images, semantics and depths
    # print('color_shape:', color.shape)  # [b, v, 3, 512, 512]
    '''
    if semantics is not None or semantics_upscale is not None:
        pca1 = PCA(n_components=3)
        semantics_masks = torch.stack([view["point_feature_mask"] for view in batch["target"]], dim=1)
        target_semantics_upscale = torch.stack([view["point_feature"] for view in batch["target"]], dim=1)
        target_semantics_upscale = target_semantics_upscale / (target_semantics_upscale.norm(dim=2, keepdim=True) + 1e-9)
        target_semantics_upscale = target_semantics_upscale * semantics_masks
        b_tsu, v_tsu, c_tsu, h_tsu, w_tsu = target_semantics_upscale.shape
        target_semantics_upscale_flat = target_semantics_upscale.permute(0, 1, 3, 4, 2).reshape(-1, c_tsu).cpu().numpy()
        target_valid_indices = np.any(target_semantics_upscale_flat != 0, axis=1)
        target_semantics_upscale_flat_valid = target_semantics_upscale_flat[target_valid_indices]
        target_semantics_upscale_pca_valid = pca1.fit_transform(target_semantics_upscale_flat_valid)
        
        target_semantics_upscale_pca = np.zeros((target_semantics_upscale_flat.shape[0], 3))
        target_semantics_upscale_pca[target_valid_indices] = target_semantics_upscale_pca_valid
        
        target_semantics_upscale_pca_tensor = torch.from_numpy(target_semantics_upscale_pca).view(b_tsu, v_tsu, h_tsu, w_tsu, 3).permute(0, 1, 4, 2, 3)
        device = semantics_masks.device
        target_semantics_upscale_pca_tensor = target_semantics_upscale_pca_tensor.to(device)
        
        target_min_value = target_semantics_upscale_pca_tensor.min()
        target_max_value = target_semantics_upscale_pca_tensor.max()
        target_semantics_upscale_pca_tensor = (target_semantics_upscale_pca_tensor - target_min_value) / (target_max_value - target_min_value + 1e-9)
        
        # if semantics is not None:
        #     # semantics_masks = torch.stack([view["point_feature_mask"] for view in batch["target"]], dim=1)
        #     semantics = semantics / (semantics.norm(dim=2, keepdim=True) + 1e-9)    
        #     semantics = semantics * semantics_masks     
        #     # print('semantics_shape:', semantics.shape)  # [b, v, dim, 512, 512]
        
        if semantics_upscale is not None:
            pca2 = PCA(n_components=3)
            semantics_upscale = semantics_upscale / (semantics_upscale.norm(dim=2, keepdim=True) + 1e-9)
            semantics_upscale = semantics_upscale * semantics_masks
            b_su, v_su, c_su, h_su, w_su = semantics_upscale.shape
            semantics_upscale_flat = semantics_upscale.permute(0, 1, 3, 4, 2).reshape(-1, c_su).cpu().numpy()
            valid_indices = np.any(semantics_upscale_flat != 0, axis=1)
            semantics_upscale_flat_valid = semantics_upscale_flat[valid_indices]
            semantics_upscale_pca_valid = pca2.fit_transform(semantics_upscale_flat_valid)
            
            semantics_upscale_pca = np.zeros((semantics_upscale_flat.shape[0], 3))
            semantics_upscale_pca[valid_indices] = semantics_upscale_pca_valid
            
            semantics_upscale_pca_tensor = torch.from_numpy(semantics_upscale_pca).view(b_su, v_su, h_su, w_su, 3).permute(0, 1, 4, 2, 3)
            device = semantics_masks.device
            semantics_upscale_pca_tensor = semantics_upscale_pca_tensor.to(device)
            
            min_value = semantics_upscale_pca_tensor.min()
            max_value = semantics_upscale_pca_tensor.max()
            semantics_upscale_pca_tensor = (semantics_upscale_pca_tensor - min_value) / (max_value - min_value + 1e-9)
            # semantics_upscale_pca_tensor = torch.from_numpy(semantics_upscale_pca).view(b_sp, v_sp, h_sp, w_sp, 3).permute(0, 1, 4, 2, 3)
            # semantics_upscale_pca_tensor = semantics_upscale_pca_tensor / (semantics_upscale_pca_tensor.norm(dim=2, keepdim=True) + 1e-9)
            # semantics_upscale_pca_tensor = semantics_upscale_pca_tensor * semantics_masks
    '''
    if semantics is not None or semantics_upscale is not None or semantics_upscale_patch is not None:
        pca = PCA(n_components=3)
        
        semantics_masks = torch.stack([view["point_feature_mask"] for view in batch["target"]], dim=1)
        target_semantics_upscale = torch.stack([view["point_feature"] for view in batch["target"]], dim=1)
        target_semantics_upscale = target_semantics_upscale / (target_semantics_upscale.norm(dim=2, keepdim=True) + 1e-9)
        
        target_semantics_upscale_patch = torch.stack([view["point_feature_patch"] for view in batch["target"]], dim=1)
        target_semantics_upscale_patch = target_semantics_upscale_patch / (target_semantics_upscale_patch.norm(dim=2, keepdim=True) + 1e-9)
        
        target_semantics_upscale = target_semantics_upscale * semantics_masks
        target_semantics_upscale_patch = target_semantics_upscale_patch * semantics_masks

        b_tsu, v_tsu, c_tsu, h_tsu, w_tsu = target_semantics_upscale.shape

        if semantics_upscale is not None:
            semantics_upscale = semantics_upscale / (semantics_upscale.norm(dim=2, keepdim=True) + 1e-9)
            semantics_upscale = semantics_upscale * semantics_masks
            b_su, v_su, c_su, h_su, w_su = semantics_upscale.shape

            semantics_upscale_patch = semantics_upscale_patch / (semantics_upscale_patch.norm(dim=2, keepdim=True) + 1e-9)
            semantics_upscale_patch = semantics_upscale_patch * semantics_masks

            '''
            # Concat target_semantics_upscale 和 semantics_upscale
            combined_upscale = torch.cat([target_semantics_upscale, semantics_upscale, target_semantics_upscale_patch, semantics_upscale_patch], dim=0)
            b_combined, v_combined, c_combined, h_combined, w_combined = combined_upscale.shape
            
            # Flatten
            combined_upscale_flat = combined_upscale.permute(0, 1, 3, 4, 2).reshape(-1, c_combined).cpu().numpy()
            combined_valid_indices = np.any(combined_upscale_flat != 0, axis=1)
            combined_upscale_flat_valid = combined_upscale_flat[combined_valid_indices]
            
            # PCA
            combined_upscale_pca_valid = pca.fit_transform(combined_upscale_flat_valid)
            combined_upscale_pca = np.zeros((combined_upscale_flat.shape[0], 3))
            combined_upscale_pca[combined_valid_indices] = combined_upscale_pca_valid
            
            combined_upscale_pca_tensor = torch.from_numpy(combined_upscale_pca).view(b_combined, v_combined, h_combined, w_combined, 3).permute(0, 1, 4, 2, 3)
            device = semantics_masks.device
            combined_upscale_pca_tensor = combined_upscale_pca_tensor.to(device)
            
            combined_min_value = combined_upscale_pca_tensor.min()
            combined_max_value = combined_upscale_pca_tensor.max()
            combined_upscale_pca_tensor = (combined_upscale_pca_tensor - combined_min_value) / (combined_max_value - combined_min_value + 1e-9)
            
            target_semantics_upscale_pca_tensor = combined_upscale_pca_tensor[:b_tsu]
            semantics_upscale_pca_tensor = combined_upscale_pca_tensor[b_tsu:2*b_tsu]
            target_semantics_upscale_patch_pca_tensor = combined_upscale_pca_tensor[2*b_tsu:3*b_tsu]
            semantics_upscale_patch_pca_tensor = combined_upscale_pca_tensor[3*b_tsu:4*b_tsu]
            '''

            combined_upscale_1 = torch.cat([target_semantics_upscale, semantics_upscale], dim=0)
            b_combined_1, v_combined_1, c_combined_1, h_combined_1, w_combined_1 = combined_upscale_1.shape

            # Flatten
            combined_upscale_flat_1 = combined_upscale_1.permute(0, 1, 3, 4, 2).reshape(-1, c_combined_1).cpu().numpy()
            combined_valid_indices_1 = np.any(combined_upscale_flat_1 != 0, axis=1)
            combined_upscale_flat_valid_1 = combined_upscale_flat_1[combined_valid_indices_1]

            # PCA
            combined_upscale_pca_valid_1 = pca.fit_transform(combined_upscale_flat_valid_1)
            combined_upscale_pca_1 = np.zeros((combined_upscale_flat_1.shape[0], 3))
            combined_upscale_pca_1[combined_valid_indices_1] = combined_upscale_pca_valid_1

            combined_upscale_pca_tensor_1 = torch.from_numpy(combined_upscale_pca_1).view(b_combined_1, v_combined_1, h_combined_1, w_combined_1, 3).permute(0, 1, 4, 2, 3)
            combined_upscale_pca_tensor_1 = combined_upscale_pca_tensor_1.to(semantics_masks.device)

            combined_min_value_1 = combined_upscale_pca_tensor_1.min()
            combined_max_value_1 = combined_upscale_pca_tensor_1.max()
            combined_upscale_pca_tensor_1 = (combined_upscale_pca_tensor_1 - combined_min_value_1) / (combined_max_value_1 - combined_min_value_1 + 1e-9)

            target_semantics_upscale_pca_tensor = combined_upscale_pca_tensor_1[:b_tsu]
            semantics_upscale_pca_tensor = combined_upscale_pca_tensor_1[b_tsu:]

            combined_upscale_2 = torch.cat([target_semantics_upscale_patch, semantics_upscale_patch], dim=0)
            b_combined_2, v_combined_2, c_combined_2, h_combined_2, w_combined_2 = combined_upscale_2.shape

            # Flatten
            combined_upscale_flat_2 = combined_upscale_2.permute(0, 1, 3, 4, 2).reshape(-1, c_combined_2).cpu().numpy()
            combined_valid_indices_2 = np.any(combined_upscale_flat_2 != 0, axis=1)
            combined_upscale_flat_valid_2 = combined_upscale_flat_2[combined_valid_indices_2]

            # PCA
            combined_upscale_pca_valid_2 = pca.fit_transform(combined_upscale_flat_valid_2)
            combined_upscale_pca_2 = np.zeros((combined_upscale_flat_2.shape[0], 3))
            combined_upscale_pca_2[combined_valid_indices_2] = combined_upscale_pca_valid_2

            combined_upscale_pca_tensor_2 = torch.from_numpy(combined_upscale_pca_2).view(b_combined_2, v_combined_2, h_combined_2, w_combined_2, 3).permute(0, 1, 4, 2, 3)
            combined_upscale_pca_tensor_2 = combined_upscale_pca_tensor_2.to(semantics_masks.device)

            combined_min_value_2 = combined_upscale_pca_tensor_2.min()
            combined_max_value_2 = combined_upscale_pca_tensor_2.max()
            combined_upscale_pca_tensor_2 = (combined_upscale_pca_tensor_2 - combined_min_value_2) / (combined_max_value_2 - combined_min_value_2 + 1e-9)

            target_semantics_upscale_patch_pca_tensor = combined_upscale_pca_tensor_2[:b_tsu]
            semantics_upscale_patch_pca_tensor = combined_upscale_pca_tensor_2[b_tsu:]

    for b in range(min(color.shape[0], 4)):
        torchvision.utils.save_image(color[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_color.jpg"))
    # if semantics is not None:
    #     for b in range(min(color.shape[0], 4)):
    #         torchvision.utils.save_image(semantics[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_semantics.jpg"))
    if target_semantics_upscale is not None:
        for b in range(min(color.shape[0], 4)):
            torchvision.utils.save_image(target_semantics_upscale_pca_tensor[b, ...], os.path.join(save_dir, f"sample_{b}_semantics_upscale_target.jpg"))
    if semantics_upscale is not None:
        for b in range(min(color.shape[0], 4)):
            torchvision.utils.save_image(semantics_upscale_pca_tensor[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_semantics_upscale.jpg"))
    if target_semantics_upscale_patch is not None:
        for b in range(min(color.shape[0], 4)):
            torchvision.utils.save_image(target_semantics_upscale_patch_pca_tensor[b, ...], os.path.join(save_dir, f"sample_{b}_semantics_upscale_patch_target.jpg"))
    if semantics_upscale_patch is not None:
        for b in range(min(color.shape[0], 4)):
            torchvision.utils.save_image(semantics_upscale_patch_pca_tensor[b, ...], os.path.join(save_dir, f"sample_{b}_rendered_semantics_upscale_patch.jpg"))
    if depth is not None:
        for b in range(min(color.shape[0], 4)):
            torchvision.utils.save_image(depth[b, :, None, ...], os.path.join(save_dir, f"sample_{b}_rendered_depth.jpg"), normalize=True)

    # Save the loss masks
    for b in range(min(mask.shape[0], 4)):
        torchvision.utils.save_image(mask[b, :, None, ...].float(), os.path.join(save_dir, f"sample_{b}_loss_mask.jpg"), normalize=True)

    # Save the masked target, semantics and rendered images
    target_original_images = torch.stack([view["original_img"] for view in batch["target"]], dim=1)
    masked_target_original_images = target_original_images * mask[..., None, :, :]
    masked_predictions = color * mask[..., None, :, :]
    # masked_semantics = semantics * mask[..., None, :, :]
    masked_target_semantics_upscale = target_semantics_upscale_pca_tensor * mask[..., None, :, :]
    masked_semantics_upscale = semantics_upscale_pca_tensor * mask[..., None, :, :]
    masked_target_semantics_upscale_patch = target_semantics_upscale_patch_pca_tensor * mask[..., None, :, :]
    masked_semantics_upscale_patch = semantics_upscale_patch_pca_tensor * mask[..., None, :, :]
    for b in range(min(target_original_images.shape[0], 4)):
        torchvision.utils.save_image(masked_target_original_images[b], os.path.join(save_dir, f"sample_{b}_masked_original_img_target.jpg"))
        torchvision.utils.save_image(masked_predictions[b], os.path.join(save_dir, f"sample_{b}_masked_rendered_color.jpg"))
        # torchvision.utils.save_image(masked_semantics[b], os.path.join(save_dir, f"sample_{b}_masked_rendered_semantics.jpg"))
        torchvision.utils.save_image(masked_target_semantics_upscale[b], os.path.join(save_dir, f"sample_{b}_masked_semantics_upscale_target.jpg"))
        torchvision.utils.save_image(masked_semantics_upscale[b], os.path.join(save_dir, f"sample_{b}_masked_rendered_semantics_upscale.jpg"))
        torchvision.utils.save_image(masked_target_semantics_upscale_patch[b], os.path.join(save_dir, f"sample_{b}_masked_semantics_upscale_patch_target.jpg"))
        torchvision.utils.save_image(masked_semantics_upscale_patch[b], os.path.join(save_dir, f"sample_{b}_masked_rendered_semantics_upscale_patch.jpg"))