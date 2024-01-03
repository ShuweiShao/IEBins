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
        • <a href="https://arxiv.org/abs/2309.14137" target='_blank'>NeurIPS 2023</a> •
    </h4>
</div>

[![KITTI Benchmark](https://img.shields.io/badge/KITTI%20Benchmark-2nd%20among%20all%20at%20submission%20time-blue)](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)

## Abstract

<div style="text-align:center">
<img src="assets/teaser.jpg"  width="80%" height="80%">
</div>

</div>
<strong>We propose a novel concept of iterative elastic bins for the classification-regression-based MDE. The proposed IEBins aims to search for high-quality depth by progressively optimizing the search range, which involves multiple stages and each stage performs a finer-grained depth search in the target bin on top of its previous stage. To alleviate the possible error accumulation during the iterative process, we utilize a novel elastic target bin to replace the original target bin, the width of which is adjusted elastically based on the depth uncertainty. </strong>

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
You can prepare the datasets KITTI and NYUv2 according to [here](https://github.com/cleinc/bts/tree/master/pytorch) and download the SUN RGB-D dataset from [here](https://rgbd.cs.princeton.edu/), and then modify the data path in the config files to your dataset locations.


## Training
First download the pretrained encoder backbone from [here](https://github.com/microsoft/Swin-Transformer), and then modify the pretrain path in the config files. If you want to train the KITTI_Official model, first download the pretrained encoder backbone from [here](https://drive.google.com/file/d/1qjDnMwmEz0k0XWh7GP2aNPGiAjvOPF_5/view?usp=drive_link), which is provided by [MIM](https://github.com/SwinTransformer/MIM-Depth-Estimation).

Training the NYUv2 model:
```
python iebins/train.py configs/arguments_train_nyu.txt
```

Training the KITTI_Eigen model:
```
python iebins/train.py configs/arguments_train_kittieigen.txt
```

Training the KITTI_Official model:
```
python iebins_kittiofficial/train.py configs/arguments_train_kittiofficial.txt
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

Evaluate the KITTI_Eigen model:
```
python iebins/eval.py configs/arguments_eval_kittieigen.txt
```

To generate KITTI Online evaluation data for the KITTI_Official model:
```
python iebins_kittiofficial/test.py --data_path path to dataset --filenames_file ./data_splits/kitti_official_test.txt --max_depth 80 --checkpoint_path path to pretrained checkpoint  --dataset kitti --do_kb_crop
```

## Qualitative Depth and Point Cloud Results
You can download the qualitative depth results of [IEBins](https://arxiv.org/abs/2309.14137), [NDDepth](https://arxiv.org/abs/2309.10592), [NeWCRFs](https://openaccess.thecvf.com/content/CVPR2022/html/Yuan_Neural_Window_Fully-Connected_CRFs_for_Monocular_Depth_Estimation_CVPR_2022_paper.html), [PixelFormer](https://openaccess.thecvf.com/content/WACV2023/html/Agarwal_Attention_Attention_Everywhere_Monocular_Depth_Prediction_With_Skip_Attention_WACV_2023_paper.html), [AdaBins](https://openaccess.thecvf.com/content/CVPR2021/html/Bhat_AdaBins_Depth_Estimation_Using_Adaptive_Bins_CVPR_2021_paper.html) and [BTS](https://arxiv.org/abs/1907.10326) on the test sets of NYUv2 and KITTI_Eigen from [here](https://pan.baidu.com/s/1zaFe40mwpQ5cvdDlLZRrCQ?pwd=vfxd) and download the qualitative point cloud results of IEBins, NDDepth, NeWCRFS, PixelFormer, AdaBins and BTS on the NYUv2 test set from [here](https://pan.baidu.com/s/1WwpFuPBGBUaSGPEdThJ6Rw?pwd=n9rw). 

If you want to derive these results by yourself, please refer to the test.py.

If you want to perform inference on a single image, run:
```
python iebins/inference_single_image.py --dataset kitti or nyu --image_path path to image --checkpoint_path path to pretrained checkpoint --max_depth 80 or 10
```
Then you can acquire the qualitative depth result.


## Models
| Model | Abs Rel | Sq Rel | RMSE | a1 | a2 | a3| Link|
| ------------ | :---: | :---: | :---: |  :---: |  :---: |  :---: |  :---: |
|NYUv2 (Swin-L)| 0.087 | 0.040 | 0.314 | 0.936 | 0.992 | 0.998 |[[Google]](https://drive.google.com/file/d/14Rn-vxvpXO2EXRaWqCPmh2JufvOurwtl/view?usp=drive_link) [[Baidu]](https://pan.baidu.com/s/1E2KAHtQ-ul99RGp_G7QK1w?pwd=7o4d)|
|NYUv2 (Swin-T)| 0.108 | 0.061 | 0.375 | 0.893 | 0.984 | 0.996 |[[Google]](https://drive.google.com/file/d/1eYkTb3grbDitQ9tJdg1DhAOaGmqgHWHK/view?usp=drive_link) [[Baidu]](https://pan.baidu.com/s/1v5_MJtP0YOSoark9Yw1RaQ?pwd=2k5d)|
|KITTI_Eigen (Swin-L)| 0.050 | 0.142 | 2.011 | 0.978 | 0.998 | 0.999 |[[Google]](https://drive.google.com/file/d/1xaVLDq7zJ-C2GtFvABolSUtK7gzvNQNd/view?usp=drive_link) [[Baidu]](https://pan.baidu.com/s/16mRrKrr9PdZhuZ3ZlkmNlA?pwd=lcjd)|
|KITTI_Eigen (Swin-T)| 0.056 | 0.169 | 2.205 | 0.970 | 0.996 | 0.999 |[[Google]](https://drive.google.com/file/d/1s0LXZmS6_Q4_H_0hmbOldPcVhlRw8Dut/view?usp=drive_link) [[Baidu]](https://pan.baidu.com/s/1xgeqIX5WP5F2MFwypMWV5A?pwd=ygfi)|
|KITTI_Official (Swinv2-L)| 5.200 | 0.788 | 2.335 | 0.974 | 0.996 | 0.999 |[[Google]](https://drive.google.com/file/d/19ARBiDTIvtZSWJVvhbEWBcZMonXsiOX1/view?usp=drive_link)|


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
