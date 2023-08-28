from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

import os
import numpy as np
import open3d as o3d
import cv2
import argparse

import util.calibration as ca
import util.utils as ut


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('image_dir', help='image file path')
    parser.add_argument('lidar_dir', help='lidar file path')
    parser.add_argument('calib_dir', help='calibration file path')
    parser.add_argument('checkpoint', help='the checkpoint file to load weights from')
    parser.add_argument('result', help='the dir to save result')
    
    args = parser.parse_args()
    
    return args
    

def main():
    
    args = parse_args()

    config_file = args.config
    checkpoint_file = args.checkpoint
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    img = args.image_dir
    lidar_dir = args.lidar_dir

    palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180],
            [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
            [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    result = inference_model(model, img)
    seg_img = result.pred_sem_seg.data[0].cpu()
    
    show_result_pyplot(model, img, result, show=True, out_file=args.result + '/result.png', opacity=0.5)

    calib = ca.CalibrationData(args.calib_dir)
    R = calib.R0
    P = calib.P
    Tr_lidar_to_cam = calib.Tr_lidar_to_cam
    P_lidar_to_cam = ut.projection_velo_to_cam(R, Tr_lidar_to_cam,P)


    rgb_img = cv2.imread(img)
    fused_img = rgb_img.copy()
    file = os.path.join(lidar_dir)
    segmented_img = cv2.imread(args.result + '/result.png')
    
    pcd_file = ut.convert_bin_to_pcd(file,file[:-3]+"pcd")
    point_cloud = np.asarray(o3d.io.read_point_cloud(file[:-3]+"pcd").points)

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

    cv2.imwrite(args.result+'/projection.png',stacked_img)

    rgb_pointcloud = np.hstack((pts_3D_img[:,:3], semantic))

    ut.visuallize_pointcloud(rgb_pointcloud,palette)
    ut.visualize_with_window(rgb_pointcloud,palette)
    ut.visualize_object(rgb_pointcloud,palette)


if __name__ == '__main__':
    main()