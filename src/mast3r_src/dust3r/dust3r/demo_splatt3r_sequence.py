# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------
import argparse
import math
import builtins
import datetime
import gradio
import os
import torch
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import einops
from einops import rearrange, repeat
import torchvision
from PIL import Image
import re
import json
import colormaps
import mediapy as media
from openclip_encoder import OpenCLIPNetwork
import cv2
from pathlib import Path
from typing import Dict, Union
import matplotlib.pyplot as plt
from collections import defaultdict

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils_activate import smooth, colormap_saving, global_colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result, save_masked_image

import matplotlib.pyplot as pl


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片目录路径")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"], help="全局对齐策略")
    parser.add_argument("--niter", type=int, default=300, help="全局对齐迭代次数")
    parser.add_argument("--min_conf_thr", type=float, default=3.0, help="置信度阈值")
    parser.add_argument("--cam_size", type=float, default=0.05, help="相机显示尺寸")
    parser.add_argument("--as_pointcloud", action='store_true', default='true', help="输出为点云模式")
    parser.add_argument("--mask_sky", action='store_true', default='False', help="启用天空遮罩")
    parser.add_argument("--clean_depth", action='store_true', default='True', help="启用深度清洗")
    parser.add_argument("--transparent_cams", action='store_true', default='False', help="透明相机显示")
    parser.add_argument("--scenegraph_type", type=str, default="swin", 
                       choices=["complete", "swin", "oneref"], help="场景图构建方式")
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default="/home/v-xingrwang/splatt3r_49dim/results/2024-10-23-16-56-15/2024-10-23-16-56-15/version_0/checkpoints/epoch11/epoch=11-step=66900.ckpt")
    parser.add_argument("--config_path", type=str, help="path to the model config", default="/home/v-xingrwang/splatt3r_49dim/configs/main_modified.yaml")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default='/home/v-xingrwang/splatt3r_49dim/tmp', help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    return parser

def set_print_with_timestamp(time_format="%Y-%m-%d %H:%M:%S"):
    builtin_print = builtins.print

    def print_with_timestamp(*args, **kwargs):
        now = datetime.datetime.now()
        formatted_date_time = now.strftime(time_format)

        builtin_print(f'[{formatted_date_time}] ', end='')  # print with time stamp
        builtin_print(*args, **kwargs)

    builtins.print = print_with_timestamp

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

class CameraParams_focals:
    def __init__(self, H: int = 512, W: int = 512, focal: float = 5.8269e+02):
        self.H = H
        self.W = W
        # self.focal = (5.8269e+02, 5.8269e+02)
        self.focal = (focal, focal)
        self.fov = (2*np.arctan(self.W / (2*self.focal[0])), 2*np.arctan(self.H / (2*self.focal[1])))
        self.K = np.array([
            [self.focal[0], 0., self.W/2],
            [0., self.focal[1], self.H/2],
            [0.,            0.,       1.],
        ]).astype(np.float32)

# sequence = '7b6477cb95'
# P = np.array([
#             [1, 0, 0, 0],
#             [0, -1, 0, 0],
#             [0, 0, -1, 0],
#             [0, 0, 0, 1]]
#         ).astype(np.float32)
# input_processed_folder = os.path.join('/home/v-xingrwang/splatt3r_49dim/data/scannetpp', 'processed', sequence)
# cams_metadata_path = f"{input_processed_folder}/dslr/nerfstudio/transforms_undistorted.json"
# with open(cams_metadata_path, "r") as f:
#     cams_metadata = json.load(f)
# file_path_to_frame_metadata = {}
# for frame in cams_metadata["frames"]:
#     file_path_to_frame_metadata[frame["file_path"]] = frame

# frame_metadata_ref = file_path_to_frame_metadata['DSC03693.JPG']
# c2w_ref = np.array(frame_metadata_ref["transform_matrix"], dtype=np.float32)
# c2w_ref = P @ c2w_ref @ P.T
# # print('c2w_ref_shape:', c2w_ref.shape)  # [4, 4]

# frame_metadata_target = file_path_to_frame_metadata['DSC03688.JPG']
# c2w_target = np.array(frame_metadata_target["transform_matrix"], dtype=np.float32)
# c2w_target = P @ c2w_target @ P.T

# inv_c2w_ref = np.linalg.inv(c2w_ref)
# aligned_c2w_target = inv_c2w_ref @ c2w_target

def align_camera_poses(sequence, ref_frame_name, target_frame_name):
    P = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]]
            ).astype(np.float32)
    input_processed_folder = os.path.join('/home/v-xingrwang/splatt3r_49dim/data/scannetpp', 'processed', sequence)
    cams_metadata_path = f"{input_processed_folder}/dslr/nerfstudio/transforms_undistorted.json"
    with open(cams_metadata_path, "r") as f:
        cams_metadata = json.load(f)
    file_path_to_frame_metadata = {}
    for frame in cams_metadata["frames"]:
        file_path_to_frame_metadata[frame["file_path"]] = frame

    frame_metadata_ref = file_path_to_frame_metadata[ref_frame_name]
    c2w_ref = np.array(frame_metadata_ref["transform_matrix"], dtype=np.float32)
    c2w_ref = P @ c2w_ref @ P.T
    # print('c2w_ref_shape:', c2w_ref.shape)  # [4, 4]

    frame_metadata_target = file_path_to_frame_metadata[target_frame_name]
    c2w_target = np.array(frame_metadata_target["transform_matrix"], dtype=np.float32)
    c2w_target = P @ c2w_target @ P.T

    inv_c2w_ref = np.linalg.inv(c2w_ref)
    aligned_c2w_target = inv_c2w_ref @ c2w_target

    return aligned_c2w_target

def sequence_camera_poses(sequence, start_frame_name, end_frame_name):
    P = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]]
            ).astype(np.float32)
    input_processed_folder = os.path.join('/home/v-xingrwang/splatt3r_49dim/data/scannetpp', 'processed', sequence)
    cams_metadata_path = f"{input_processed_folder}/dslr/nerfstudio/transforms_undistorted.json"
    with open(cams_metadata_path, "r") as f:
        cams_metadata = json.load(f)
    file_path_to_frame_metadata = {}
    for frame in cams_metadata["frames"]:
        file_path_to_frame_metadata[frame["file_path"]] = frame

    start_num = int(start_frame_name.split('DSC')[1].split('.JPG')[0])
    end_num = int(end_frame_name.split('DSC')[1].split('.JPG')[0])

    sequence_c2ws = []

    for num in range(start_num, end_num + 1):
        frame_name = f"DSC{num:05d}.JPG"  # Format the number with leading zeros
        if frame_name in file_path_to_frame_metadata:
            frame_metadata = file_path_to_frame_metadata[frame_name]
            c2w = np.array(frame_metadata["transform_matrix"], dtype=np.float32)
            c2w = P @ c2w @ P.T
            sequence_c2ws.append(c2w)

    return sequence_c2ws

def colormap_saving(image: torch.Tensor, colormap_options, save_path):
    """
    if image's shape is (h, w, 1): draw colored relevance map;
    if image's shape is (h, w, 3): return directively;
    if image's shape is (h, w, c): execute PCA and transform it into (h, w, 3).
    """
    output_image = (
        colormaps.apply_colormap(
            image=image,
            colormap_options=colormap_options,
        ).cpu().numpy()
    )
    if save_path is not None:
        media.write_image(save_path.with_suffix(".png"), output_image, fmt="png")
    return output_image

def normalize_intrinsics(intrinsics, image_shape):
    '''Normalize an intrinsics matrix given the image shape'''
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] /= image_shape[1]
    intrinsics[..., 1, :] /= image_shape[0]
    return intrinsics

def process_gt_annotations(json_paths: list, output_path: Path = None) -> Dict:
    """
    Organize ground truth annotations based on provided JSON paths.
    GT format:
        file content: labelme format
    Args:
        json_paths: list of JSON file paths
        output_path: directory to save visualization or output (if needed)
    Returns:
        gt_ann: dict
            keys: str(int(idx))
            values: dict
                keys: str(label)
                values: dict containing 'bboxes' and 'mask'
    """
    gt_ann = {}
    for js_path in json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        
        key = js_path  # 使用完整路径作为索引
        
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']  # Label
            box = np.asarray(prompt_data['bbox']).reshape(-1)  # Bounding box (x1, y1, x2, y2)
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])  # Segmentation mask
            
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)  # Stack masks if multiple instances of the label
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0
                )
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # Save visualization if needed
            # if output_path is not None:
            #     save_path = output_path / 'gt' / gt_data['info']['name'].split('.JPG')[0] / f'{label}.jpg'
            #     save_path.parent.mkdir(exist_ok=True, parents=True)
            #     vis_mask_save(mask, save_path)
        
        # Store annotations for the current index
        gt_ann[key] = img_ann

    return gt_ann

def activate_stream(sem_map,    # 还原的语义图
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    batch_idx=0, 
                    v_idx=0, 
                    input_mask=None,
                    semantic_mask=None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None):

    output_path_loca = image_name / 'localization'
    # output_path_loca = os.path.join(image_name, 'localization')
    output_path_loca.mkdir(exist_ok=True, parents=True)
    # os.makedirs(output_path_loca, exist_ok=True)
    
    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264, 获取激活图
    valid_map_just_cos = clip_model.get_max_across_just_cos(sem_map)    # without canonical phrase, just cos sim
    valid_map_all_canonical_phrase = clip_model.get_positive_similarity(sem_map)
    valid_map_masked_all = torch.zeros_like(valid_map).to(valid_map.device)
    # valid_map = clip_model.get_positive_similarity(sem_map)     # 不使用negatives
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list, iou_scores = [], [], []
    iou_results = {}
    acc_num = 0
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        mask_gt_localization = np.zeros((h, w))   # 记录下mask_gt, 方便后续判断定位精度
        valid_iou = False
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)   # 对当前激活图进行滤波
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])    # 将平滑后的激活图与原始激活图加权平均
            # valid_map[i][k] = valid_map[i][k]    # 不加平滑

            if input_mask is not None:
                # mask_selected = input_mask[batch_idx, v_idx]  # 选择对应的 mask
                mask_selected = input_mask

            # 热力图处理
            valid_map_masked = valid_map[i][k].clone()  # 克隆一个副本来应用mask
            valid_map_just_cos_masked = valid_map_just_cos[i][k].clone()
            valid_map_all_canonical_phrase_masked = valid_map_all_canonical_phrase[i][k].clone()
            if input_mask is not None:
                valid_map_masked *= mask_selected  # 应用mask
                # valid_map_masked *= semantic_mask.squeeze(0)
                valid_map_masked_all[i][k] = valid_map_masked   # 这里的valid_map加入了mask
                valid_map_just_cos_masked *= mask_selected
                valid_map_all_canonical_phrase_masked *= mask_selected
            
            output_path_relev = image_name / 'heatmap' / f'batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}_{i}'   # 保存激活图
            # output_path_relev = os.path.join(image_name, 'heatmap', f'batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}_{i}')
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            # os.makedirs(output_path_relev, exist_ok=True)
            colormap_saving(valid_map_masked.unsqueeze(-1), colormap_options, output_path_relev)

            output_path_global = image_name / 'heatmap_global' / f'batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}_{i}'   # 保存激活图
            # output_path_global = os.path.join(image_name, 'heatmap_global', f'batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}_{i}')
            output_path_global.parent.mkdir(exist_ok=True, parents=True)
            # os.makedirs(output_path_global, exist_ok=True)
            global_colormap_saving(valid_map_masked.unsqueeze(-1), colormap_options, output_path_global)

            output_path_just_cos = image_name / 'heatmap_global_just_cos' / f'batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}_{i}'   # 保存激活图
            # output_path_just_cos = os.path.join(image_name, 'heatmap_global_just_cos', f'batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}_{i}')
            output_path_just_cos.parent.mkdir(exist_ok=True, parents=True)
            os.makedirs(output_path_just_cos, exist_ok=True)
            global_colormap_saving(valid_map_just_cos_masked.unsqueeze(-1), colormap_options, output_path_just_cos)

            output_path_all_canonical_phrase = image_name / 'heatmap_global_all_canonical_phrase' / f'batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}_{i}'   # 保存激活图
            # output_path_all_canonical_phrase = os.path.join(image_name, 'heatmap_global_all_canonical_phrase', f'batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}_{i}')
            output_path_all_canonical_phrase.parent.mkdir(exist_ok=True, parents=True)
            # os.makedirs(output_path_all_canonical_phrase, exist_ok=True)
            valid_map_all_canonical_phrase_enlarge = valid_map_all_canonical_phrase_masked
            global_colormap_saving(valid_map_all_canonical_phrase_enlarge.unsqueeze(-1), colormap_options, output_path_all_canonical_phrase)
            
            
            # NOTE 与lerf一致，激活值低于0.5的认为是背景
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 1.0

            # 最后应用mask保存组合图
            if input_mask is not None:
                valid_composited *= mask_selected.unsqueeze(-1)  # 在保存组合图时应用 mask
                # valid_composited *= semantic_mask.squeeze(0).unsqueeze(-1)
            
            output_path_compo = image_name / 'composited' / f'batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}_{i}'
            # output_path_compo = os.path.join(image_name, 'composited', f'batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}_{i}')
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            # os.makedirs(output_path_compo, exist_ok=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)  # 保存合成图像
            
            
            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred
            mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
            # mask_gt = img_ann[clip_model.positives[k].replace('the ', '')]['mask'].astype(np.uint8)
            mask_gt = smooth(mask_gt)   # 也给gt加了smooth
            
            # calculate iou
            # 先加上还是后加上mask_selected?
            # print('mask_pred_shape:', mask_pred.shape)  # [512, 512]
            # print('mask_selected_shape:', mask_selected.shape)  # [512, 512]
            if input_mask is not None:
                mask_selected_np = mask_selected.cpu().numpy()  # 将torch.Tensor转为numpy.ndarray
                mask_gt *= mask_selected_np  # 应用mask
                # print('mask_gt_type:', mask_gt.dtype)
                # print('semantic_mask:', semantic_mask.squeeze(0).cpu().numpy().dtype)
                # mask_gt *= semantic_mask.squeeze(0).cpu().numpy()
                mask_pred *= mask_selected_np
                # mask_pred *= semantic_mask.squeeze(0).cpu().numpy()

                mask_gt_localization = mask_gt

                if not np.any(mask_gt):  
                    continue  # 直接跳过当前的迭代
            
            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))

            # 如果union为0, 跳过当前的迭代
            if union == 0:
                continue  # 跳过无效的IOU计算

            iou = np.sum(intersection) / np.sum(union)
            iou_lvl[i] = iou

            valid_iou = True

            # 将该次的iou分数及相关信息保存到列表
            # iou_info = {
            #     'prompt': clip_model.positives[k],
            #     'iou': iou_lvl[i]
            # }
            # iou_scores.append(iou_info)
            # if clip_model.positives[k] not in iou_results:
            #     iou_results[clip_model.positives[k]] = []  # 初始化存储空间
            # iou_results[clip_model.positives[k]].append(iou_lvl[i])  # 将iou分数添加到对应的prompt下

        chosen_lvl = -1
        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            # score = valid_map_masked_all[i, k].max()
            # score = valid_map_all_canonical_phrase[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)

        score_lvl_localization = np.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            score = valid_map_masked_all.cpu().numpy()[i, k].max()
            coord = np.nonzero(valid_map_masked_all.cpu().numpy()[i, k] == score)
            score_lvl_localization[i] = score
            coord_lvl.append(coord)
        selec_head = np.argmax(score_lvl_localization)
        coord_final = coord_lvl[selec_head]
        
        # 记录chosen_lvl对应的IoU值
        if valid_iou:
            iou_info = {
                'prompt': clip_model.positives[k],
                'iou': iou_lvl[chosen_lvl],  # 记录chosen_lvl的IoU值
                'chosen_head': chosen_lvl
            }
            # iou_info = {
            #     'prompt': clip_model.positives[k],
            #     'iou': iou_lvl[chosen_lvl],  # 记录chosen_lvl的IoU值
            #     'chosen_head': chosen_lvl,
            #     'sam': iou_lvl[0],
            #     'patch': iou_lvl[1],
            #     'difference': iou_lvl[0] - iou_lvl[1],
            #     'sam_loc': 1 if (mask_gt_localization[coord_lvl[0]] == True).any() else 0,
            #     'patch_loc': 1 if (mask_gt_localization[coord_lvl[1]] == True).any() else 0,
            #     'ratio': (mask_gt_localization != 0).sum() / (512 ** 2)
            # }
            iou_scores.append(iou_info)  # 保存IoU信息
            if clip_model.positives[k] not in iou_results:
                iou_results[clip_model.positives[k]] = []  # 初始化存储空间
            iou_results[clip_model.positives[k]].append(iou_lvl[chosen_lvl])  # 将chosen_lvl对应的IoU添加到prompt下

            chosen_iou_list.append(iou_lvl[chosen_lvl])
            chosen_lvl_list.append(chosen_lvl.cpu().numpy())

            if (mask_gt_localization[coord_final] == True).any():
                # print('prompt:', clip_model.positives[k])
                localization_save_path = output_path_loca / f"batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}.png"
                # localization_save_path = os.path.join(output_path_loca, f"batch_{batch_idx}_view_{v_idx}_{clip_model.positives[k]}.png")
                acc_num += 1
                plt.imshow((image.cpu().numpy() * 255).astype(np.uint8))
                # plt.scatter(coord_final[1], coord_final[0], color='red', s=10)
                plt.scatter(coord_final[1], coord_final[0], color='firebrick', marker='o', s=100, edgecolor='black', linewidth=2.5, alpha=1)
                plt.axis('off')
                plt.savefig(localization_save_path, bbox_inches='tight', pad_inches=0, dpi=200)
                plt.close()

        '''
        # 基于IoU分数选择最佳head
        chosen_lvl = -1
        if valid_iou:
            chosen_lvl = np.argmax(iou_lvl)  # 选择IoU分数最高的head

            # 记录chosen_lvl对应的IoU值
            iou_info = {
                'prompt': clip_model.positives[k],
                'iou': iou_lvl[chosen_lvl],  # 记录chosen_lvl的IoU值
                'chosen_head': chosen_lvl
            }
            iou_scores.append(iou_info)  # 保存IoU信息

            if clip_model.positives[k] not in iou_results:
                iou_results[clip_model.positives[k]] = []  # 初始化存储空间
            iou_results[clip_model.positives[k]].append(iou_lvl[chosen_lvl])  # 将chosen_lvl对应的IoU添加到prompt下

            chosen_iou_list.append(iou_lvl[chosen_lvl])
            chosen_lvl_list.append(chosen_lvl)
        '''
        if chosen_lvl != -1:
            # save for visulsization
            save_path = image_name / 'chosen' / f'batch_{batch_idx}_view_{v_idx}_chosen_{clip_model.positives[k]}.png'
            # vis_mask_save(mask_lvl[chosen_lvl] * mask_selected_np * semantic_mask.squeeze(0).cpu().numpy(), save_path)   # 保存时也考虑了可见不可见
            vis_mask_save(mask_lvl[chosen_lvl] * mask_selected_np, save_path)
            gt_save_path = image_name / 'gt' / f'batch_{batch_idx}_view_{v_idx}_chosen_{clip_model.positives[k]}.png'
            vis_mask_save(mask_gt, gt_save_path)    # 保存时也考虑了可见不可见
            masked_image_save_path = image_name / 'gt_masked_image' / f'batch_{batch_idx}_view_{v_idx}_chosen_{clip_model.positives[k]}.png'
            save_masked_image(image, mask_gt, masked_image_save_path)   # 保存时也考虑了可见不可见
        
        # 在循环结束后保存iou_scores到文件
        # 假设image_name是一个Path对象
        iou_save_dir = image_name / 'iou'
        iou_save_dir.mkdir(exist_ok=True)  # 创建目录，若不存在

        iou_save_path = iou_save_dir / f'batch_{batch_idx}_view_{v_idx}_iou_scores.txt'  # 构建完整的路径
        with open(iou_save_path, 'w') as f:
            for iou_info in iou_scores:
                # 以清晰的方式写入文件，标注出 batch、view、prompt、head 和 iou 分数
                f.write(f"Prompt {iou_info['prompt']}, "
                        f"Chosen Head: {iou_info['chosen_head']}, "
                        f"IOU: {iou_info['iou']:.4f}\n")
        # iou_scores.sort(key=lambda x: x['difference'], reverse=True)
        # with open(iou_save_path, 'w') as f:
        #     for iou_info in iou_scores:
        #         # 以清晰的方式写入文件，标注出 batch、view、prompt、head 和 iou 分数
        #         f.write(f"Prompt {iou_info['prompt']}, "
        #                 f"Chosen Head: {iou_info['chosen_head']}, "
        #                 f"sam: {iou_info['sam']:.4f}, "
        #                 f"patch: {iou_info['patch']:.4f}, "
        #                 f"sam_loc: {iou_info['sam_loc']}, "
        #                 f"patch_loc: {iou_info['patch_loc']}, "
        #                 f"difference: {iou_info['difference']:.4f}, "
        #                 f"ratio: {iou_info['ratio'] * 100:.2f}%\n")
        
    # print('chosen_iou_list:', chosen_iou_list)
    # print('chosen_lvl_list:', chosen_lvl_list)
    # return chosen_iou_list, chosen_lvl_list
    # print('acc_num:', acc_num)
    if len(chosen_iou_list) == 0:
        localization_acc = None
    else:
        localization_acc = round(acc_num / len(chosen_iou_list), 4)
    # localization_acc = round(acc_num / len(chosen_iou_list), 4)
    print('localization_acc:', localization_acc)
    return iou_results, chosen_iou_list, chosen_lvl_list, localization_acc

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    # print('rgbimg_length:', len(rgbimg))    # N
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)


def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, sequence, target_frame_name, target_frame_index, save_npy, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid, render_cuda_func):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)
    
    print('after get_3D_model_from_scene !!!')

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    # print('rgbimg:', rgbimg)
    rgbimg = np.array(rgbimg)
    # print('rgbimg_shape:', rgbimg.shape)    # (6, 512, 512, 3)

    # for i in range(rgbimg.shape[0]):
    #     img = rgbimg[i]
    #     img = (img * 255).astype(np.uint8)
    #     img_pil = Image.fromarray(img)
    #     output_filename = os.path.join(outdir, f'image_{i}.png')
    #     img_pil.save(output_filename)

    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    covariances = scene.final_gaussian_covariances
    # print('covariances[0]_shape:', covariances[0].shape)    # torch.Size([512, 512, 3, 3])
    covariances = torch.stack(covariances).unsqueeze(0)
    # print('covariances_shape:', covariances.shape)    # torch.Size([6, 512, 512, 3, 3])
    # print('covariances:', covariances)
    harmonics = scene.final_gaussian_harmonics
    harmonics = torch.stack(harmonics).unsqueeze(0)

    opacities = scene.final_gaussian_opacities
    opacities = torch.stack(opacities).unsqueeze(0)

    language_feature_precomp = scene.final_gaussian_language_features_precomp
    language_feature_precomp = torch.stack(language_feature_precomp).unsqueeze(0)

    '''
    extrinsics = to_numpy(scene.get_im_poses().cpu())
    # print('extrinsics_shape:', extrinsics.shape)    # [N, 4, 4]
    extrinsics = torch.tensor(extrinsics).unsqueeze(0).to(device)
    # print('extrinsics_shape:', extrinsics.shape)    # torch.Size([1, 6, 4, 4])
    # print('extrinsics:', extrinsics)
    extrinsics = extrinsics[:, 9:10, :, :]  # 选出特定的某一个
    np.save('/home/v-xingrwang/splatt3r_49dim/tmp/extrinsics.npy', extrinsics.cpu().numpy())
    '''
    extrinsics = to_numpy(scene.get_im_poses().cpu())
    extrinsics = torch.tensor(extrinsics).unsqueeze(0).to(device)

    if save_npy:
        extrinsics = extrinsics[:, target_frame_index:target_frame_index+1, :, :]  # 选出特定的某一个
        np.save('/home/v-xingrwang/splatt3r_49dim/tmp/extrinsics.npy', extrinsics.cpu().numpy())
    else:
        extrinsics = np.load('/home/v-xingrwang/splatt3r_49dim/tmp/extrinsics.npy')
        extrinsics = torch.tensor(extrinsics).to(device)
    
    # sequence = 'bde1e479ad'
    # ref_frame_name = 'DSC02337'
    # target_frame_name = 'DSC02212'

    # sequence = '7b6477cb95'
    # ref_frame_name = 'DSC03693'
    # target_frame_name = 'DSC03688'
    
    # sequence = '40aec5fffa'
    # ref_frame_name = 'DSC09719'
    # target_frame_name = 'DSC09696'
    '''
    extrinsics = align_camera_poses(sequence, ref_frame_name + '.JPG', target_frame_name + '.JPG')
    # extrinsics = align_camera_poses('7b6477cb95', 'DSC03693.JPG', 'DSC03679.JPG')
    extrinsics = torch.tensor(extrinsics).unsqueeze(0).unsqueeze(0).to(device)
    '''

    focals = to_numpy(scene.get_focals().cpu()) # maybe use the default focal length?
    # print('focals:', focals)    # [[596.6384], [589.4419], [603.9808], [628.5698], [593.3170], [597.7216]]

    intrinsics = []
    for focal in focals:
        intrinsics.append(CameraParams_focals(H=image_size, W=image_size, focal=focal[0]).K)
    
    '''
    intrinsics = np.array(intrinsics)
    # print('intrinsics_shape:', intrinsics.shape)    # [N, 3, 3]
    intrinsics = torch.tensor(intrinsics).unsqueeze(0).to(device)
    intrinsics = normalize_intrinsics(intrinsics, (image_size, image_size))[..., :3, :3]
    # print('intrinsics_shape:', intrinsics.shape)    # torch.Size([1, 6, 3, 3])
    intrinsics = intrinsics[:, 9:10, :, :]  # 待修改, 须看是第多少个
    np.save('/home/v-xingrwang/splatt3r_49dim/tmp/intrinsics.npy', intrinsics.cpu().numpy())
    '''

    intrinsics = np.array(intrinsics)
    intrinsics = torch.tensor(intrinsics).unsqueeze(0).to(device)
    intrinsics = normalize_intrinsics(intrinsics, (image_size, image_size))[..., :3, :3]

    if save_npy:
        intrinsics = intrinsics[:, target_frame_index:target_frame_index+1, :, :]  # 待修改, 须看是第多少个
        np.save('/home/v-xingrwang/splatt3r_49dim/tmp/intrinsics.npy', intrinsics.cpu().numpy())
    else:
        intrinsics = np.load('/home/v-xingrwang/splatt3r_49dim/tmp/intrinsics.npy')
        intrinsics = torch.tensor(intrinsics).to(device)
    
    means = scene.get_pts3d()
    means = torch.stack(means).unsqueeze(0)
    # print('means_shape:', means.shape)  # means_shape: torch.Size([1, 6, 512, 512, 3])

    b, v, _, _ = extrinsics.shape  # 批次大小和视角数
    near = torch.full((b, v), 0.1, device=device)
    far = torch.full((b, v), 1000.0, device=device)

    color, semantics = render_cuda_func(
        rearrange(extrinsics, "b v i j -> (b v) i j"),
        rearrange(intrinsics, "b v i j -> (b v) i j"),
        rearrange(near, "b v -> (b v)"),
        rearrange(far, "b v -> (b v)"),
        (image_size, image_size),
        repeat(torch.tensor([1.0, 1.0, 1.0], device=device), "c -> (b v) c", b=b, v=v),  # 假设背景颜色为白色
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

    for b_idx in range(color.shape[0]):  # 遍历批次中的每个样本
        for v_idx in range(color.shape[1]):  # 遍历每个视角      
            # 保存渲染的颜色图像
            rendered_color = color[b_idx, v_idx]  # Shape: [3, h, w]
            # 使用 Path 来构建文件路径
            color_save_path = os.path.join(outdir, f"sample_{b_idx}_view_{v_idx}_rendered_color.jpg")
            torchvision.utils.save_image(rendered_color, color_save_path)

    clip_model = OpenCLIPNetwork(device)
    colormap_options = colormaps.ColormapOptions(   # 颜色映射选项
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    for b_idx in range(color.shape[0]):
        
        save_dir = Path(outdir)
        target_original_images_json_path = os.path.join('/home/v-xingrwang/splatt3r_49dim/data/scannetpp/undistort_2D_semantics_512_json', sequence, target_frame_name + '.json')
        target_original_images_json_path = [target_original_images_json_path]
        gt_ann = process_gt_annotations(target_original_images_json_path, save_dir)
        json_name_idx = target_original_images_json_path[0]
        img_ann = gt_ann[json_name_idx]
        img_ann_keys = list(img_ann.keys())
        if sequence == '7b6477cb95':
            input_str = "[storage cabinet, office chair, floor, table, wall, whiteboard]"
            mask_path = '/home/v-xingrwang/splatt3r_49dim/collection/single_scene_splatt3r/eval_result_7b6477cb95_1_1/batch_1/sample_1_view_0_mask.npy'
        elif sequence == '40aec5fffa':
            input_str = "[door, floor, wall, refrigerator, kitchen cabinet]"
            mask_path = '/home/v-xingrwang/splatt3r_49dim/collection/single_scene_splatt3r/eval_result_40aec5fffa_6_3/batch_6/sample_3_view_0_mask.npy'
        elif sequence == 'bde1e479ad':
            input_str = "[ceiling, wall, door, floor, whiteboard, table]"
            mask_path = '/home/v-xingrwang/splatt3r_49dim/collection/single_scene_splatt3r/eval_result_bde1e479ad_13_0/batch_13/sample_0_view_0_mask.npy'
        elif sequence == 'e398684d27':
            input_str = "[floor, door, wall, storage cabinet, ceiling, box]"
            mask_path = '/home/v-xingrwang/splatt3r_49dim/collection/single_scene_splatt3r/eval_result_e398684d27_14_2/batch_14/sample_2_view_0_mask.npy'
        elif sequence == '825d228aec':
            input_str = "[floor, wall, table, door, ceiling, box, shelf]"
            mask_path = '/home/v-xingrwang/splatt3r_49dim/collection/single_scene_splatt3r/eval_result_825d228aec_2_3/batch_2/sample_3_view_0_mask.npy'
        elif sequence == 'fb5a96b1a2':
            input_str = "[ceiling, floor, whiteboard, chair, table, wall]"
            mask_path = '/home/v-xingrwang/splatt3r_49dim/collection/single_scene_splatt3r/eval_result_fb5a96b1a2_6_0/batch_6/sample_0_view_0_mask.npy'
        elif sequence == 'bde1e479ad':
            input_str = "[ceiling, door, floor, table, wall, whiteboard]"
            mask_path = '/home/v-xingrwang/splatt3r_49dim/collection/single_scene_splatt3r/eval_result_bde1e479ad_13_0/batch_13/sample_0_view_0_mask.npy'

        if input_str:  
            provided_categories = input_str.strip("[]").split(",")  
            provided_categories = [item.strip() for item in provided_categories]  
            provided_categories = [str(item) for item in provided_categories]    
            intersection = list(set(img_ann_keys) & set(provided_categories))  
        else:
            intersection = img_ann_keys
        
        clip_model.set_positives(intersection)

        mask = torch.from_numpy(np.load(mask_path)).squeeze(0).to(device)     # 记得修改
        # print('mask_shape:', mask.shape)
        # mask = torch.ones_like(mask)    # 先不使用mask
        # total_sum = torch.sum(mask)
        # num_elements = torch.numel(mask)
        # if total_sum == num_elements:
        #     print("Mask中全部为1!")
        # else:
        #     print("Mask中不全部为1!")

        sample_iou_results = []  # 用于存储当前样本的所有iou值

        for v_idx in range(color.shape[1]):
                
                img_path = os.path.join(outdir, f"sample_{b_idx}_view_{v_idx}_rendered_color.jpg")
                rgb_img = cv2.imread(str(img_path))[..., ::-1]  # BGR转RGB
                rgb_img = (rgb_img / 255.0).astype(np.float32)
                rgb_img = torch.from_numpy(rgb_img).to(device)

                # 加载渲染的sam语义特征
                restored_feat_sam = semantics_sam[b_idx, v_idx]
                restored_feat_sam = restored_feat_sam.permute(1, 2, 0)  # 变为 [h, w, c]
                restored_feat_sam = restored_feat_sam.unsqueeze(0)      # 变为 [1, h, w, c]

                # 加载渲染的patch语义特征
                restored_feat_patch = semantics_patch[b_idx, v_idx]
                restored_feat_patch = restored_feat_patch.permute(1, 2, 0)  # 变为 [h, w, c]
                restored_feat_patch = restored_feat_patch.unsqueeze(0)      # 变为 [1, h, w, c]

                # 将render的两类特征concat
                restored_feat = torch.cat((restored_feat_sam, restored_feat_patch), dim=0)

                iou_results, chosen_iou_list, chosen_lvl_list, localization_acc = activate_stream(restored_feat, rgb_img, clip_model, save_dir, batch_idx=b_idx, v_idx=v_idx, input_mask=mask, semantic_mask=None, img_ann=img_ann,
                                thresh=0.5, colormap_options=colormap_options)

                # 用iou_results而不是chosen_iou_list计算平均IoU
                if len(chosen_iou_list) != 0:
                    for prompt, iou_list in iou_results.items():
                        avg_iou_per_prompt = np.mean(iou_list)  # 对每个prompt的所有head的iou求平均
                        sample_iou_results.append(avg_iou_per_prompt)  # 将每个prompt的平均iou添加到sample_iou_results

        # 在每个样本处理完后计算该样本的mIoU
        if sample_iou_results:
            sample_miou = np.mean(sample_iou_results)  # 使用sample_iou_results计算mIoU
            print(f"mIoU: {sample_miou:.4f}")



    return scene, outfile, imgs


def main_demo(tmpdirname, model, device, image_size, render_cuda_func, silent=False):
    
    # get args
    parser = get_args_parser()
    args = parser.parse_args()

    # 获取输入图片
    filelist = [os.path.join(args.input_dir, f) 
        for f in os.listdir(args.input_dir) 
        if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    if not filelist:
        raise ValueError(f"No images found in {args.input_dir}")

    # 修改下列参数, 同时记得修改get_reconstructed_scene函数中的mask和query texts
    # sequence = 'fb5a96b1a2'
    # input_frame1_name = 'DSC02981'
    # input_frame2_name = 'DSC02998'
    # target_frame_name = 'DSC02789'
    # save_npy = False
    # start = 2789
    # end = 2998

    sequence = 'bde1e479ad'
    input_frame1_name = 'DSC02303'
    input_frame2_name = 'DSC02369'
    target_frame_name = 'DSC02212'
    save_npy = True
    start = 2212
    end = 2369

    all_list = [f"DSC0{num:04}" for num in range(start, end + 1)]
    all_list = [os.path.join(args.input_dir, f"{frame_name}.JPG") for frame_name in all_list]
    train_list = [f for f in all_list if target_frame_name not in f]
    train_list = train_list[::2]    # 间隔采样

    ensure_frames = [input_frame1_name, input_frame2_name, target_frame_name]
    for frame_name in ensure_frames:
        frame_path = next((f for f in all_list if frame_name in f), None) # 找到对应的路径
        if frame_path and frame_path not in train_list:
            train_list.append(frame_path)
    train_list = sorted(train_list, key=lambda filename: int(re.search(r'(\d+)', os.path.basename(filename)).group(1)) if re.search(r'(\d+)', os.path.basename(filename)) else -1)
    # print('len(train_list):', len(train_list))  # 80

    filelist = sorted(filelist, key=lambda filename: int(re.search(r'(\d+)', os.path.basename(filename)).group(1)) if re.search(r'(\d+)', os.path.basename(filename)) else -1)
    # filelist = filelist[:30]
    # print(filelist[26])
    # print('len(filelist):', len(filelist))  # 154
    filelist = [c for c in filelist if c in train_list]
    print('filelist:', filelist)

    # output_txt_path = '/home/v-xingrwang/debug_image_names/image_names_ours.txt'
    # with open(output_txt_path, 'w') as f:
    #     for file_path in filelist:
    #         image_name = os.path.basename(file_path)  # 提取文件名
    #         f.write(image_name + '\n')  # 写入 TXT 文件，每行一个文件名

    # print(f"成功将图片名称保存到 {output_txt_path}")

    try:
        target_frame_index = next(i for i, filename in enumerate(filelist) if target_frame_name in filename)
        print(f"索引为{target_frame_index}")
    except StopIteration:
        print(f"未在filelist中找到 {target_frame_name}")

    if not save_npy:
        del filelist[target_frame_index] # 把target view空出来
    

    # 固定参数设置
    fixed_params = {
        'schedule': args.schedule,  # linear
        'niter': args.niter,    # 300
        'min_conf_thr': args.min_conf_thr,  # 3.0
        'as_pointcloud': args.as_pointcloud,    # True
        'mask_sky': args.mask_sky,  # False
        'clean_depth': args.clean_depth,    # True
        'transparent_cams': args.transparent_cams,  # False
        'cam_size': args.cam_size,  # 0.05
        'scenegraph_type': args.scenegraph_type,    # complete
        'winsize': 1,  
        'refid': 0,
        'render_cuda_func': render_cuda_func
    }

    scene, glb_file, viz_imgs = get_reconstructed_scene(
        tmpdirname, model, device, silent, image_size,
        filelist, sequence, target_frame_name, target_frame_index, save_npy, **fixed_params
    )
    
