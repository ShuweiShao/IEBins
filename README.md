<div align="center">

<h1>IEBins: Iterative Elastic Bins for Monocular Depth Estimation</h1>

<div>
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=ecZHSVQAAAAJ' target='_blank'>Shuwei Shao</a><sup>1</sup>&emsp;
    <a target='_blank'>Zhongcai Pei</a><sup>1</sup>&emsp;
    <a target='_blank'>Xingming Wu</a><sup>1</sup>&emsp;
    <a target='_blank'>Zhong Liu</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=5PoZrcYAAAAJ' target='_blank'>Weihai Chen</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=LiUX7WQAAAAJ' target='_blank'>Zhengguo Li</a><sup>3</sup>
</div>
<div>
    <sup>1</sup>Beihang University, <sup>2</sup>Anhui University, <sup>3</sup>A*STAR
</div>


<div>
    <h4 align="center">
        • <a href="https://arxiv.org/pdf/2309.14137.pdf" target='_blank'>NeurIPS 2023</a> •
    </h4>
</div>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/iebins-iterative-elastic-bins-for-monocular/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=iebins-iterative-elastic-bins-for-monocular)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/iebins-iterative-elastic-bins-for-monocular/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=iebins-iterative-elastic-bins-for-monocular)


<strong>We propose a novel concept of iterative elastic bins for the classification-regression-based MDE. The proposed IEBins aims to search for high-quality depth by progressively optimizing the search range, which involves multiple stages and each stage performs a finer-grained depth search in the target bin on top of its previous stage. To alleviate the possible error accumulation during the iterative process, we utilize a novel elastic target bin to replace the original target bin, the width of which is adjusted elastically based on the depth uncertainty. </strong>

<div style="text-align:center">
<img src="assets/teaser.jpg"  width="80%" height="80%">
</div>

---

</div>

## Installation
```
conda create -n iebins python=3.8
conda activate iebins
conda install pytorch=1.10.0 torchvision cudatoolkit=11.1
pip install matplotlib, tqdm, tensorboardX, timm, mmcv, open3d
```

## Datasets
You can prepare the datasets KITTI and NYUv2 according to [here](https://github.com/cleinc/bts) and download the SUN RGB-D dataset from [here](https://rgbd.cs.princeton.edu/), and then modify the data path in the config files to your dataset locations.


## Training
First download the pretrained encoder backbone from [here](https://github.com/microsoft/Swin-Transformer), and then modify the pretrain path in the config files.

Training the NYUv2 model:
```
python iebins/train.py configs/arguments_train_nyu.txt
```

Training the KITTI model:
```
python iebins/train.py configs/arguments_train_kittieigen.txt
```

## Evaluation
Evaluate the NYUv2 model:
```
python iebins/eval.py configs/arguments_eval_nyu.txt
```

Evaluate the NYUv2 model on the SUN RGB-D dataset:
```
python iebins/eval_sun.py configs/arguments_eval_sun.txt
```

Evaluate the KITTI model:
```
python iebins/eval.py configs/arguments_eval_kittieigen.txt
```


## Qualitative Depth and Point Cloud Results
Please refer to the test.py.


## Models
| Model | Abs.Rel. | Sqr.Rel | RMSE | RMSElog | a1 | a2 | a3| SILog| 
| :--- | :---: | :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
|[NYUv2 (Swin-L)](https://drive.google.com/file/d/14Rn-vxvpXO2EXRaWqCPmh2JufvOurwtl/view?usp=drive_link) | 0.087 | 0.040 | 0.314 | 0.112 | 0.936 | 0.992 | 0.998 | 8.777 |
|[NYUv2 (Swin-T)](https://drive.google.com/file/d/1eYkTb3grbDitQ9tJdg1DhAOaGmqgHWHK/view?usp=drive_link)| 0.108 | 0.061 | 0.375 | 0.136 | 0.893 | 0.984 | 0.996 | 10.932 |
|[KITTI_Eigen (Swin-L)](https://drive.google.com/file/d/1xaVLDq7zJ-C2GtFvABolSUtK7gzvNQNd/view?usp=drive_link)| 0.050 | 0.142 | 2.011 | 0.075 | 0.978 | 0.998 | 0.999 | 6.821 |
|[KITTI_Eigen (Swin-T)](https://drive.google.com/file/d/1s0LXZmS6_Q4_H_0hmbOldPcVhlRw8Dut/view?usp=drive_link)| 0.056 | 0.169 | 2.205 | 0.084 | 0.970 | 0.996 | 0.999 | 7.738 |


## Citation

If you find our work useful in your research please consider citing our paper:

```
@inproceedings{shao2023IEBins,
title={IEBins: Iterative Elastic Bins for Monocular Depth Estimation},
author={Shao, Shuwei and Pei, Zhongcai and Wu, Xingming and Liu, Zhong and Chen, Weihai and Li, Zhengguo},
booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
year={2023}
}
```

## Contact

If you have any questions, please feel free to contact swshao@buaa.edu.cn.


## Acknowledgement

Our code is based on the implementation of [NeWCRFs](https://github.com/aliyun/NeWCRFs) and [BTS](https://github.com/cleinc/bts). We thank their excellent works.
