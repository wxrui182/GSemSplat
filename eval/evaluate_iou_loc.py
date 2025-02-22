#!/usr/bin/env python
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import PIL

import sys
sys.path.append("..")
import colormaps
from openclip_encoder import OpenCLIPNetwork
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    # if img is None:
    #     raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_loss_masks(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:

    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    render_img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'render_frame_*.jpg')))
    mask_paths = sorted(glob.glob(os.path.join(str(json_folder), 'mask_*.npy')))

    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        scene_name = gt_data['info']['scene_name']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)
            mask = polygon_to_mask((gt_data['info']['height'], gt_data['info']['width']), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, img_paths, render_img_paths, mask_paths

def activate_stream_loss_masks(sem_map,
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None,
                    input_mask = None,
                    semantic_mask = None):

    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)

    valid_map = clip_model.get_max_across(sem_map)
    valid_map_masked_all = torch.zeros_like(valid_map).to(valid_map.device)
    n_head, n_prompt, h, w = valid_map.shape

    iou_results = []
    total_iou = 0

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    acc_num = 0
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        valid_iou = False
        mask_gt_localization = np.zeros((h, w))
        for i in range(n_head):

            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
            
            valid_map_masked = valid_map[i][k].clone()
            if input_mask is not None:
                valid_map_masked *= input_mask
                # valid_map_masked *= semantic_mask.squeeze(0)
                valid_map_masked_all[i][k] = valid_map_masked

            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map_masked.unsqueeze(-1), colormap_options, output_path_relev)
            
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 1.0

            if input_mask is not None:
                valid_composited *= input_mask.unsqueeze(-1)

            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)
            
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
            mask_gt = smooth(mask_gt)
            
            if input_mask is not None:
                input_mask_np = input_mask.cpu().numpy()
                mask_gt *= input_mask_np
                mask_pred *= input_mask_np
                mask_gt_localization = mask_gt

                if not np.any(mask_gt):
                    continue

            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))

            if union == 0:
                continue

            iou = np.sum(intersection) / np.sum(union)
            iou_lvl[i] = iou

            valid_iou = True

            print('prompt:', clip_model.positives[k])
            print('iou:', iou)
            # print('intersection:', intersection)
            # print('union:', union)

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
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
        
        if valid_iou:
            chosen_iou = iou_lvl[chosen_lvl]
            total_iou += chosen_iou
            chosen_iou_list.append(iou_lvl[chosen_lvl])
            chosen_lvl_list.append(chosen_lvl.cpu().numpy())

            iou_results.append(f"Prompt: {clip_model.positives[k]}, IoU: {chosen_iou:.4f}")

            if (mask_gt_localization[coord_final] == True).any():
                localization_save_path = output_path_loca / f"{clip_model.positives[k]}.png"
                acc_num += 1
                plt.imshow((image.cpu().numpy() * 255).astype(np.uint8))
                plt.scatter(coord_final[1], coord_final[0], color='firebrick', marker='o', s=100, edgecolor='black', linewidth=2.5, alpha=1)
                plt.axis('off')
                plt.savefig(localization_save_path, bbox_inches='tight', pad_inches=0, dpi=200)
                plt.close()
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl] * input_mask_np, save_path)

    localization_acc = round(acc_num / len(chosen_iou_list), 4)
    iou_results.append(f"\nMean Localization Acc: {localization_acc:.4f}")

    mean_iou = total_iou / len(chosen_iou_list)
    iou_results.append(f"\nMean IoU: {mean_iou:.4f}")

    iou_output_path = image_name / 'iou_results.txt'
    with open(iou_output_path, 'w') as f:
        f.write("\n".join(iou_results))

    return chosen_iou_list, chosen_lvl_list


def evaluate_loss_masks(feat_dir, output_path, json_folder, mask_thresh, logger):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    gt_ann, image_paths, render_image_paths, mask_paths = eval_gt_loss_masks(Path(json_folder), Path(output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]

    clip_model = OpenCLIPNetwork(device)
    chosen_iou_all, chosen_lvl_list = [], []

    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx+1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)

        render_rgb_img = imread_cv2(render_image_paths[j])
        mask_be_seen = torch.from_numpy(np.load(mask_paths[j])).to(device)

        with torch.no_grad():

            restored_feat = np.load(feat_dir)
            restored_feat = torch.from_numpy(restored_feat).to(device)
            # restored_feat = restored_feat * mask_be_seen.unsqueeze(0).unsqueeze(-1)

            render_rgb_img = np.array(render_rgb_img)
            render_rgb_img = (render_rgb_img / 255.0).astype(np.float32)
            render_rgb_img = torch.from_numpy(render_rgb_img).to(device)

            render_rgb_img_masked = render_rgb_img * mask_be_seen.unsqueeze(-1)
            render_rgb_img_masked_np = render_rgb_img_masked.cpu().numpy()
            render_rgb_img_masked_np = (render_rgb_img_masked_np * 255).astype(np.uint8)
            cv2.imwrite(str(image_name / "masked_image.png"), render_rgb_img_masked_np[..., ::-1])

        img_ann = gt_ann[f'{idx}']
        img_ann_keys = list(img_ann.keys())
        input_str = "[door, floor, wall, refrigerator, kitchen cabinet]"
        if input_str:  
            provided_categories = input_str.strip("[]").split(",")  
            provided_categories = [item.strip() for item in provided_categories]  
            provided_categories = [str(item) for item in provided_categories]    
            intersection = list(set(img_ann_keys) & set(provided_categories))  
        else:  
            intersection = img_ann_keys
        
        clip_model.set_positives(intersection)
        c_iou_list, c_lvl = activate_stream_loss_masks(restored_feat, render_rgb_img, clip_model, image_name, img_ann,
                                            thresh=mask_thresh, colormap_options=colormap_options, input_mask=mask_be_seen, semantic_mask=None)
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

    # iou
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")


if __name__ == "__main__":
    
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--json_folder", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    args = parser.parse_args()

    # NOTE config setting
    dataset_name = args.dataset_name
    mask_thresh = args.mask_thresh
    feat_dir = os.path.join(args.feat_dir, dataset_name, "restored_feat.npy")
    output_path = os.path.join(args.output_dir, dataset_name)
    json_folder = os.path.join(args.json_folder, dataset_name)

    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

    evaluate_loss_masks(feat_dir, output_path, json_folder, mask_thresh, logger)