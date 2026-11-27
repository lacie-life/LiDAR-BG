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
import tqdm

sys.path.append('../')

import config.a9_config as cnf
from data_process import a9_data_utils, a9_bev_utils
from data_process.a9_dataloader import create_test_dataloader, create_val_dataloader
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.visualization_utils import show_image_with_boxes_a9, merge_rgb_to_bev_with_gt, predictions_to_kitti_format_a9


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4_leaky.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default='/mnt/hdd/A9_dataset/complex_yolo_weights/3_classes_r0_north_south/Model_best_epoch_280.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint') # /mnt/hdd/A9_dataset/complex_yolo_weights/3_classes_north_south/Model_north_south_3_classes_epoch_275.pth   
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'A9_dataset')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.working_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_test_configs()
    configs.distributed = False  # For testing

    model = create_model(configs)
    model.print_network()
    print('\n\n' + '-*=' * 30 + '\n\n')
    
    device_string = 'cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx)
    
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location=device_string))

    configs.device = torch.device(device_string)
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    val_loader = create_val_dataloader(configs)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            img_paths, imgs, targets = batch_data
            input_imgs = imgs.to(device=configs.device).float()

            t1 = time_synchronized()
            outputs = model(input_imgs)
            t2 = time_synchronized()

            detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            img_detections = []  # Stores detections for each image index
            img_detections.extend(detections)

            img_bev = imgs.squeeze() * 255
            img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
            img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
            
            gt = imgs.squeeze() * 255
            gt = gt.permute(1, 2, 0).numpy().astype(np.uint8)
            gt = cv2.resize(gt, (configs.img_size, configs.img_size))

            for detections in img_detections:
                if detections is None:
                    continue
                # Rescale boxes to original image
                detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
                for x, y, w, l, im, re, *_, cls_pred in detections:
                    yaw = np.arctan2(im, re)
                    # Draw rotated box
                    a9_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

            
            # np.save(os.path.join("south_weight_north_lidar_conf_0.05_prediction",img_paths[0].split('/')[-1].replace(".jpg",".npy")),np.array(img_detections[0]))

            targets[:, 2:6] *= configs.img_size
            # Get yaw angle
            targets[:, 6] = torch.atan2(targets[:, 6], targets[:, 7])
 
            for c, x, y, w, l, yaw in targets[:, 1:7].numpy():
                # Draw rotated box
                a9_bev_utils.drawRotatedBox(gt, x, y, w, l, yaw, cnf.colors[int(c)])

            gt = cv2.rotate(gt, cv2.ROTATE_180)

            img_rgb = cv2.imread(img_paths[0])
            
            img_rgb_2 = cv2.imread(img_paths[0].replace("image_north_south_1","image_north_south_2"))

            img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)

            out_img = merge_rgb_to_bev_with_gt(img_rgb, img_rgb_2, img_bev, gt, output_width=608)

            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
                                                                                           1 / (t2 - t1)))

            if configs.save_test_output:
                if configs.output_format == 'image':
                    img_fn = os.path.basename(img_paths[0])[:-4]
                    cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img)
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
                            fourcc, 30, (out_cap_w, out_cap_h))

                    out_cap.write(out_img)
                else:
                    raise TypeError

            if configs.show_image:
                # cv2.imshow('test-img', out_img)
                # print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
                # if cv2.waitKey(0) & 0xFF == 27:
                #     break
                # print(img_paths[0].split('/')[-1])
                cv2.imwrite(os.path.join('north_south_r0_conf_0.5',img_paths[0].split('/')[-1]), out_img)
        # print("average FPS : ", fps_np.mean())
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
