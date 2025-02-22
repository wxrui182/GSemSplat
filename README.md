# GSemSplat

This is the code release for the paper **GSemSplat: Generalizable Semantic 3D Gaussian Splatting from Uncalibrated Image Pairs**, by Xingrui Wang, Cuiling Lan, Hanxin Zhu, Zhibo Chen and Yan Lu.

- Arxiv: https://arxiv.org/pdf/2412.16932

## :sparkles:To Do List

- [x] GSemSplat Test Results (Tab. 1)
- [ ] Training and Inference Code
- [ ] Complete Semantic Dataset

## :sparkles: GSemSplat Test Results (Tab. 1)

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

