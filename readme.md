# ST-ReID

本项目基于 ACT (AAAI 2020) 的开源代码 ([论文](https://arxiv.org/abs/1912.01349), [代码](https://github.com/FlyingRoastDuck/ACT_AAAI20))

根据初始匹配结果的时间戳 (用GMM) 学习到一个时空模型，然后用于度量学习中的三元组选取、距离矩阵的重排序。


|                | M2D  | D2M  |
| -------------- | ---- | ---- |
| ACT            | 54.5 | 60.6 |
| ST-ReID visual | 60.9 | 64.1 |
| MMT            | 68.7 | 74.5 |
| ST-ReID fused  | 76.0 | 69.9 |

## Requirements
* python 3.7
* Market1501, DukeMTMC-reID and other datasets.
  > Download all necessary datasets and move them to 'data' by following instructions in 'data/readme.md'
* Other necessary packages listed in requirements.txt
* ACT pre-trained models
  > Download models from [Baidu NetDisk](https://pan.baidu.com/s/1uPjKpkdZjqSJdk3XxR1-Yg) (Password: 9aba) or [Google Drive](https://drive.google.com/file/d/1W1BcmHjmzxR3TVj2rFpnV703Huat3AeA/view?usp=sharing). Models are named by the following formula:
    `ada{src}2{tgt}.pth` where "src" and "tgt" are the initial letter of source and target dataset's name.
    
## Training Re-ID model

```
CUDA_VISIBLE_DEVICES=0,1,2 python selftrainingBayes.py  --src_dataset market1501 --tgt_dataset dukemtmc --resume ACT_pretrain/logMar/adaM2D.pth --data_dir ./data --logs_dir ./log/M2D > out.log
```

avaliable choices to fill "src_dataset_name" and "tgt_dataset_name" are: 
market1501 (for Market1501), dukemtmc (for DukeMTMC-reID), cuhk03 (for CUHK03).

## 代码介绍

`selftraining*.py` 文件各代表一种训练算法，现已有 ACT、PAST-ReID的复现，不同分类 loss 或 triplet loss 的对比，改用hough变换作为时空约束等。

实现算法时，除了 `selftraining*.py`，主要改动还有 `reid/trainers.py`, `reid/loss/*.py`, `reid/utils/data/sampler.py`

`test_*.ipynb` 是对时空模型的具体测试，包括对时空转移概率分布的可视化、三元组选取准确率的计算、重排序方法的对比等。

