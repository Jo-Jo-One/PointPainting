# PointPainting Implementation
---

## Installation
mmsegmentation install


https://mmsegmentation.readthedocs.io/en/latest/get_started.html

```
cuda 11.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install opencv-python==4.8.0.76
```

## checkpoint download
```
!wget https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth -P checkpoint
```

## run instruction
```
python pointpainting.py config/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py training/image_2/000004.png training/velodyne/000004.bin training/calib/000004.txt checkpoint/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth result
```

## data file structure(KITTI Dataset)
```
training
├── calid
├── image_2
├── label_2
├──velodyne
```
