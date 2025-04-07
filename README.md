# GSemSplat: Generalizable Semantic 3D Gaussian Splatting from Uncalibrated Image Pairs

This is the code release for the paper **GSemSplat: Generalizable Semantic 3D Gaussian Splatting from Uncalibrated Image Pairs**, by Xingrui Wang, Cuiling Lan, Hanxin Zhu, Zhibo Chen and Yan Lu.

- Arxiv: https://arxiv.org/pdf/2412.16932

![](https://github.com/wxrui182/GSemSplat/blob/main/figs/Framework.png)

## :sparkles:Abstract

*Modeling and understanding the 3D world is crucial for various applications, from augmented reality to robotic navigation. Recent advancements based on 3D Gaussian Splatting have integrated semantic information from multi-view images into Gaussian primitives. However, these methods typically require costly per-scene optimization from dense calibrated images, limiting their practicality. In this paper, we consider the new task of generalizable 3D semantic field modeling from sparse, uncalibrated image pairs. Building upon the Splatt3R architecture, we introduce GSemSplat, a framework that learns open-vocabulary semantic representations linked to 3D Gaussians without the need for per-scene optimization, dense image collections or calibration. To ensure effective and reliable learning of semantic features in 3D space, we employ a dual-feature approach that leverages both region-specific and context-aware semantic features as supervision in the 2D space. This allows us to capitalize on their complementary strengths. Experimental results on the ScanNet++ dataset demonstrate the effectiveness and superiority of our approach compared to the traditional scene-specific method. We hope our work will inspire more research into generalizable 3D understanding.*

## :bell:To Do List

- [x] GSemSplat Test Results (Tab. 1)
- [ ] Training and Inference Code
- [ ] Complete Semantic Dataset

## :one: GSemSplat Test Results (Tab. 1)

For environment setup, please refer to [LangSplat](https://github.com/minghanqin/LangSplat).

### 1. Preparing Tab. 1 Data

The Tab. 1 dataset is accessible for download via the following link: [eval_datasets](https://drive.google.com/file/d/1HyFko44xCELD_1G1YZKqOqkhMIo3B3kE/view) .

```
eval_datasets/
├── 7b6477cb95/
├── 40aec5fffa/
├── 825d228aec/
└── ...

eval/
├── colormaps.py
├── colors.py
├── eval.sh
└── ...
```

### 2. Test Results

Two points to note:

- In `eval/eval.sh`, replace `CASE_NAME` with the name of your testing scenario.
- In `eval/evaluate_iou_loc.py`, modify `input_str = "[]"` by referring to the `queries.txt` files within the different scenarios in `eval_datasets`.

```
cd GSemSplat/eval
bash eval.sh
```

## :two: Inference Demo (updating)
The pre-trained model checkpoints can be downloaded from [this link](https://rec.ustc.edu.cn/share/d4689c70-137d-11f0-badd-bb93c350eb7f).
