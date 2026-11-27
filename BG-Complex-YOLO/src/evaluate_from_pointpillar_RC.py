"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.31
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for evaluation

"""

import argparse
import os
import time
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.utils.data.distributed
import cv2
from tqdm import tqdm
from easydict import EasyDict as edict

sys.path.append('./')

from data_process import a9_data_utils,a9_bev_utils
from data_process.rcooper_dataloader import create_val_dataloader
from models.model_utils import create_model
from utils.misc import AverageMeter, ProgressMeter
from utils.evaluation_utils import post_processing, get_batch_statistics_rotated_bbox, get_batch_statistics_rotated_bbox_BEV, rescale_boxes,ap_per_class, load_classes, post_processing_v2
from models.experimental import attempt_load
from utils.visualization_utils import show_image_with_boxes,merge_rgb_to_bev_with_gt, merge_rgb_to_bev, predictions_to_kitti_format
import config.rcooper_config as cnf

def evaluate_mAP(val_loader, model, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time],
                             prefix="Evaluation phase...")
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    bev_sample_metrics = []
    # switch to evaluate mode
    # model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch_data in enumerate(tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            img_paths, imgs, targets = batch_data
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale x, y, w, h of targets ((box_idx, class, x, y, z, h, w, l, im, re))
            targets[:, 2:4] *= configs.img_size
            targets[:, 5:8] *= configs.img_size
            
            targets = targets.to(configs.device, non_blocking=True)
            imgs = imgs.to(configs.device, non_blocking=True)

            # outputs = model(imgs)
            # outputs = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            outputs = torch.tensor(np.load(os.path.join("/home/ivpg/PJW/CUDA-PointPillars/OpenPCDet/BG_pointpillars_best_weight_RC",img_paths[0].split("/")[-1][:-4]+"_pointpillar_a9.npy"))).cuda()
            
            bc = cnf.boundary
            for i in range(outputs.shape[0]):
                y, x, z, h, w, l, im, re, conf, cls_, cl = outputs[i]
  
                outputs[i][0] = ((y - bc["minY"]) / (bc["maxY"] - bc["minY"]) *configs.img_size)
                outputs[i][1] = ((x - bc["minX"]) / (bc["maxX"] - bc["minX"]) *configs.img_size)
                outputs[i][2] = (z - bc["minZ"]) / (bc["maxZ"] - bc["minZ"])
                outputs[i][3] = (h / (bc["maxZ"] - bc["minZ"]) *configs.img_size)
                outputs[i][4] = (w / (bc["maxY"] - bc["minY"]) *configs.img_size)
                outputs[i][5] = (l / (bc["maxX"] - bc["minX"]) *configs.img_size)
                outputs[i][-1] = cl-1
            
            outputs = [outputs]
            
            sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=configs.iou_thresh)
            bev_sample_metrics += get_batch_statistics_rotated_bbox_BEV(outputs, targets, iou_threshold=configs.iou_thresh)
            
            # measure elapsed time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)
            
            # img_detections = []  # Stores detections for each image index
            # img_detections.extend(outputs)
            # predictions = []
            # pred_scores_for_openpcdet = np.array([])
            # pred_labels_for_openpcdet = np.array([])
            # for detections in img_detections:
            #     if detections is None:
            #         continue
            #     # Rescale boxes to original image
                
            #     for x, y, z, h, w, l, im, re, obj_conf, class_score, cls_pred in detections:
            #         pred_scores_for_openpcdet = np.append(pred_scores_for_openpcdet,[obj_conf.cpu()])
            #         # pred_scores_for_openpcdet = np.append(pred_scores_for_openpcdet,[class_score.cpu()])
            #         pred_labels_for_openpcdet = np.append(pred_labels_for_openpcdet,[cls_pred.cpu()+1])
                    
            #         predictions.append([cls_pred, x / configs.img_size, y / configs.img_size, z , h / configs.img_size, w / configs.img_size, l / configs.img_size, im, re])
            
            # predictions = a9_bev_utils.inverse_yolo_target_for_openpcdet(predictions, cnf.boundary)
        
            
            # np.save(os.path.join("complex_yolov7_3D_a9_results_openpcdet",img_paths[0].split("/")[-1][:-4]+"_pred_boxes.npy"),predictions[:,1:])
            # np.save(os.path.join("complex_yolov7_3D_a9_results_openpcdet",img_paths[0].split("/")[-1][:-4]+"_pred_scores.npy"),pred_scores_for_openpcdet)
            # np.save(os.path.join("complex_yolov7_3D_a9_results_openpcdet",img_paths[0].split("/")[-1][:-4]+"_pred_labels.npy"),pred_labels_for_openpcdet)
            
            # img_bev = imgs.squeeze() * 255
            # img_bev = img_bev.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            # img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
            
            # gt = imgs.squeeze() * 255
            # gt = gt.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            # gt = cv2.resize(gt, (configs.img_size, configs.img_size))

            # for detections in img_detections:
            #     if detections is None:
            #         continue
            #     # Rescale boxes to original image
            #     detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
            #     detections = detections.cpu()
            #     for x, y, z, h, w, l, im, re, *_, cls_pred in detections:
            #         yaw = np.arctan2(im, re)
            #         # Draw rotated box
            #         a9_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])
            #             # np.save(os.path.join("south_weight_north_lidar_conf_0.05_prediction",img_paths[0].split('/')[-1].replace(".jpg",".npy")),np.array(img_detections[0]))
            
            # # Rescale x, y, w, h of targets ((box_idx, class, x, y, z, h, w, l, im, re))
            # for c, x, y, z, h, w, l, im,re in targets[:, 1:10].cpu().numpy():
            #     yaw = np.arctan2(im,re)
            #     a9_bev_utils.drawRotatedBox(gt, x, y, w, l, yaw, cnf.colors[int(c)])

            # gt = cv2.rotate(gt, cv2.ROTATE_180)
            # # cv2.imshow('test-img', gt)
            # # if cv2.waitKey(0) & 0xFF == 27:
            # #         break
            # img_rgb = cv2.imread(img_paths[0])
            # img_rgb_2 = cv2.imread(img_paths[0].replace("image_north_south_1","image_north_south_2"))
            
            # # print(img_paths[0])
            # img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)

            # out_img = merge_rgb_to_bev_with_gt(img_rgb, img_rgb_2, img_bev, gt, output_width=608)

            # # if batch_idx != 0:
            # #     fps_np[batch_idx] = 1 / (t2-t1)
            
            # if configs.show_image:
            #     cv2.imshow('test-img', out_img)
            #     print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
            #     if cv2.waitKey(0) & 0xFF == 27:
            #         break
            #     # print(img_paths[0].split('/')[-1])
                
            #     # cv2.imwrite(os.path.join('a9_pc_inference_results',img_paths[0].split('/')[-1]), out_img)    
            # # Log message
            # if logger is not None:
            #     if ((batch_idx + 1) % configs.print_freq) == 0:
            #         logger.info(progress.get_message(batch_idx))

            # start_time = time.time()

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
        bev_true_positives, bev_pred_scores, bev_pred_labels = [np.concatenate(x, 0) for x in list(zip(*bev_sample_metrics))]
        bev_precision, bev_recall, bev_AP, bev_f1, bev_ap_class = ap_per_class(bev_true_positives, bev_pred_scores, bev_pred_labels,labels)
        
    return precision, recall, AP, f1, ap_class, bev_precision,bev_recall, bev_AP, bev_f1, bev_ap_class


def parse_eval_configs():
    parser = argparse.ArgumentParser(description='Evaluation config for YOLO3D Implementation')
    parser.add_argument('--classnames-infor-path', type=str, default='/mnt/8TB/PJW/RC_dataset_yolo3d/classes_names.txt',
                        metavar='PATH', help='The class names of objects in the task')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture') 
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/yolo3d_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default="/mnt/hdd/A9_dataset/complex_yolo3D_weights/complex_yolov7_3D_a9_openpcdet/Model_complex_yolov7_3D_a9_openpcdet_epoch_300.pth", metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE lossfor training')
    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default="0", type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf-thresh', type=float, default=0.01,
                        help='for evaluation - the threshold for class conf')
    parser.add_argument('--nms-thresh', type=float, default=0.01,
                        help='for evaluation - the threshold for nms')
    parser.add_argument('--iou-thresh', type=float, default=0.6,
                        help='for evaluation - the threshold for IoU')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '/mnt/8TB/PJW/RC_dataset_yolo3d'
    configs.dataset_dir = os.path.join(configs.root_dir)

    return configs


if __name__ == '__main__':
    configs = parse_eval_configs()
    configs.distributed = False  # For evaluation
    class_names = load_classes(configs.classnames_infor_path)

    # model = create_model(configs)
    # model.print_network()
    print('\n\n' + '-*=' * 30 + '\n\n')
    # assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    # model.load_state_dict(torch.load(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    # model = model.to(device=configs.device)

    # model.eval()
    print('Create the validation dataloader')
    val_dataloader = create_val_dataloader(configs)
    # model = attempt_load(configs.pretrained_path, map_location=configs.device)
    print("\nStart computing mAP...\n")
    model = []
    precision, recall, AP, f1, ap_class ,bev_precision,bev_recall, bev_AP, bev_f1, bev_ap_class = evaluate_mAP(val_dataloader, model, configs, None)
    print("\nDone computing mAP...\n")
    for idx, cls in enumerate(ap_class):
        print("\t>>>\t Class {} ({}): 3D_precision = {:.4f}, 3D_recall = {:.4f}, 3D_AP = {:.4f}, 3D_f1: {:.4f}, BEV_precision = {:.4f}, BEV_recall = {:.4f}, BEV_AP = {:.4f}, BEV_f1: {:.4f}".format(cls, \
                class_names[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx], bev_precision[idx], bev_recall[idx], bev_AP[idx], bev_f1[idx]))

    print("\nmAP: {} BEV mAP: {}\n".format(AP.mean(),bev_AP.mean()))

    print("\nmAP: {}\n".format(AP.mean()))
