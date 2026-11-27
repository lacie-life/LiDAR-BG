"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.08
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

sys.path.append('../')

import config.a9_config as cnf
from data_process import a9_data_utils, a9_bev_utils
from data_process.a9_dataloader import create_val_dataloader
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.visualization_utils import show_image_with_boxes, merge_predition_gt_bev, predictions_to_kitti_format
from models.experimental import attempt_load

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for YOLO3D Implementation')
    parser.add_argument('--saved_fn', type=str, default='yolo3d_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/yolo3d_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default="/mnt/hdd/A9_dataset/complex_yolo3D_weights/complex_yolov3D_A9_dataset_background_baseline_retrain_4th/Model_complex_yolov3D_A9_dataset_background_baseline_retrain_4th_epoch_180.pth", metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default="0", type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.4,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.01,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_image', action='store_true',
                        help='If true, save the image during demostration')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_yolo3d_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    
    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'A9_dataset')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_test_configs()
    configs.distributed = False  # For testing
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    
    model = attempt_load(os.path.join(configs.pretrained_path), map_location=configs.device)  # load FP32 model
            # summary(model,torch.rand(1,3,608,608).to(configs.device))  # create
    
    model = model.to(device=configs.device)

    out_cap = None
    model.eval()

    test_dataloader = create_val_dataloader(configs)
    labels = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            
            img_paths, imgs, targets = batch_data
            
            labels += targets[:, 1].tolist()
            
            targets[:, 2:4] *= configs.img_size
            targets[:, 5:8] *= configs.img_size     
            
            targets = targets.to(configs.device, non_blocking=True)
            imgs = imgs.to(configs.device, non_blocking=True)
            
            outputs = model(imgs)

            # Outputs: (batch_size x ... x 12) 12 includes: x,y,z,h,w,l,im,re,conf,cls
            detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            img_detections = []  # Stores detections for each image index
            img_detections.extend(detections)
            
            img_bev = imgs.clone()
            img_bev = img_bev.squeeze()*255
            img_bev = img_bev.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # img_bev = img_bev.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
            
            img_bev = np.array(img_bev[:,:,:3]) # 6ch -> 3ch
            
            img_bev_2 = imgs.clone()
            img_bev_2 = img_bev_2.squeeze()*255
            img_bev_2 = img_bev_2.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # img_bev_2 = img_bev_2.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            img_bev_2 = cv2.resize(img_bev_2, (configs.img_size, configs.img_size))
            img_bev_2 = np.array(img_bev_2[:,:,:3])
            
            for c, x, y, z, h, w, l, im, re in targets[:, 1:].cpu().numpy():
                yaw = np.arctan2(im, re)
                # print(x, y, w, l, yaw)
                a9_bev_utils.drawRotatedBox_GT(img_bev, x, y, w, l, yaw, cnf.gt_colors[int(c)])
            
            for detections in img_detections:
                if detections is None:
                    continue
                # Rescale boxes to original image
                detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
                
                for x, y, z, h, w, l, im, re, *_, cls_pred in detections:
                    x, y, z, h, w, l, im, re,  cls_pred = x.cpu(),y.cpu(),z.cpu(),h.cpu(),w.cpu(),l.cpu(),im.cpu(),re.cpu(),cls_pred.cpu()
                    
                    # Draw rotated box
                    yaw = np.arctan2(im, re)
                    a9_bev_utils.drawRotatedBox(img_bev_2, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

            img_bev = cv2.rotate(img_bev, cv2.ROTATE_180)
            img_bev_2 = cv2.rotate(img_bev_2, cv2.ROTATE_180)
            
            out_img = merge_predition_gt_bev(img_bev, img_bev_2, output_width=configs.img_size)
            save_path = os.path.join("./A9_dataset_inference_results/baseline_results",img_paths[0].split('/')[-1])
            print(save_path)
            if configs.show_image:
                cv2.imshow('test-img', out_img)
                print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
                if cv2.waitKey(0) & 0xFF == 27:
                    break
            if configs.save_image:
                cv2.imwrite(save_path,img_bev_2)
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
