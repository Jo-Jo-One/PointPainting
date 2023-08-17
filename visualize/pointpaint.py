from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

import os
import numpy as np
import open3d as o3d
import cv2

import util.calibration as ca
import util.utils as ut

config_file = 'mmsegmentation/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = 'mmsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')
img = 'C:/Users/JY/Desktop/Study/Programmers/Final project/visualize/training/image_2/000004.png'
lidar_dir = 'C:/Users/JY/Desktop/Study/Programmers/Final project/visualize/training/velodyne/000004.bin'

palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180],
           [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
           [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

# result = inference_model(model, img)
result = inference_model(model, img)
seg_img = result.pred_sem_seg.data[0].cpu()

# show_result_pyplot(model, img, result, show=True)
# show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)

calib = ca.CalibrationData('C:/Users/JY/Desktop/Study/Programmers/Final project/visualize/training/calib/000004.txt')
R = calib.R0
P = calib.P
Tr_lidar_to_cam = calib.Tr_lidar_to_cam
P_lidar_to_cam = ut.projection_velo_to_cam(R, Tr_lidar_to_cam,P)



rgb_img = cv2.imread(img)
fused_img = rgb_img.copy()
file = os.path.join(lidar_dir)
segmented_img = cv2.imread('C:/Users/JY/Desktop/Study/Programmers/Final project/visualize/result.jpg')


pcd_file = ut.convert_bin_to_pcd(file,'C:/Users/JY/Desktop/Study/Programmers/Final project/visualize/training/velodyne/000004.pcd')
point_cloud = np.asarray(o3d.io.read_point_cloud('C:/Users/JY/Desktop/Study/Programmers/Final project/visualize/training/velodyne/000004.pcd').points)

idx = point_cloud[:,0] >= 0
point_cloud = point_cloud[idx]

pts_2D,depth, pts_3D_img = ut.project_lidar_on_image(P_lidar_to_cam, point_cloud, (rgb_img.shape[1], rgb_img.shape[0]))

N = pts_3D_img.shape[0]
semantic = np.zeros((N,1), dtype=np.float32)

for i in range(pts_2D.shape[0]):
    if i >= 0:

        x = np.int32(pts_2D[i, 0])
        y = np.int32(pts_2D[i, 1])

        classID = np.float64(segmented_img[y, x]) 
        
        pt = (x,y)
        cv2.circle(fused_img, pt, 2, color=tuple(classID), thickness=1)

        semantic[i] = seg_img[y,x]
    
stacked_img = np.vstack((rgb_img, segmented_img,fused_img))
cv2.imwrite('projection.png',stacked_img)

rgb_pointcloud = np.hstack((pts_3D_img[:,:3], semantic))
# ut.visuallize_pointcloud(rgb_pointcloud,"C:/Users/JY/Desktop/Study/Programmers/Final project/visualize","000004",palette)
# ut.visualize_with_window(rgb_pointcloud,"C:/Users/JY/Desktop/Study/Programmers/Final project/visualize","000004",palette)
ut.visualize_object(rgb_pointcloud,"C:/Users/JY/Desktop/Study/Programmers/Final project/visualize","000004",palette)


