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
import natsort
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.utils.data.distributed
from tqdm import tqdm
from easydict import EasyDict as edict

sys.path.append('./')

from data_process.a9_dataloader import create_val_dataloader,create_train_dataloader
from models.model_utils import create_model
from utils.misc import AverageMeter, ProgressMeter
from utils.evaluation_utils import post_processing, get_batch_statistics_rotated_bbox, get_batch_statistics_rotated_bbox_BEV, ap_per_class, load_classes, post_processing_v2
from models.experimental import attempt_load

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
            _, imgs, targets = batch_data
            # Extract labels
            labels += targets[:, 1].tolist()
            
            # Rescale x, y, w, h of targets ((box_idx, class, x, y, z, h, w, l, im, re))
            targets[:, 2:4] *= configs.img_size
            targets[:, 5:8] *= configs.img_size
            
            
            targets = targets.to(configs.device, non_blocking=True)
            imgs = imgs.to(configs.device, non_blocking=True)

            outputs = model(imgs)
            # print(outputs)
            outputs = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

            sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=configs.iou_thresh)
            bev_sample_metrics += get_batch_statistics_rotated_bbox_BEV(outputs, targets, iou_threshold=configs.iou_thresh)
            
            # measure elapsed time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)

            # Log message
            if logger is not None:
                if ((batch_idx + 1) % configs.print_freq) == 0:
                    logger.info(progress.get_message(batch_idx))

            start_time = time.time()
        
        val_car_counts = labels.count(0.0)
        val_wheeler_counts = labels.count(1.0)
        val_pedestrian_counts = labels.count(2.0)
        
        print("val_car_counts : ", val_car_counts)
        print("val_wheeler_counts : ", val_wheeler_counts)
        print("val_ped_counts : ", val_pedestrian_counts)
        
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
        bev_true_positives, bev_pred_scores, bev_pred_labels = [np.concatenate(x, 0) for x in list(zip(*bev_sample_metrics))]
        bev_precision, bev_recall, bev_AP, bev_f1, bev_ap_class = ap_per_class(bev_true_positives, bev_pred_scores, bev_pred_labels,labels)
        
    return precision, recall, AP, f1, ap_class, bev_precision,bev_recall, bev_AP, bev_f1, bev_ap_class


def parse_eval_configs():
    parser = argparse.ArgumentParser(description='Evaluation config for YOLO3D Implementation')
    parser.add_argument('--classnames-infor-path', type=str, default='../dataset/A9_dataset_yolo3d/classes_names.txt',
                        metavar='PATH', help='The class names of objects in the task')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/yolo3d_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default="/home/ivpg/PJW/Complex-YOLO3D/dataset/A9_dataset_yolo3d/complex_yolo_v7_original_8.0_4th", metavar='PATH',
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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf-thresh', type=float, default=0.01,
                        help='for evaluation - the threshold for class conf')
    parser.add_argument('--nms-thresh', type=float, default=0.01,
                        help='for evaluation - the threshold for nms')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for IoU')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'A9_dataset_yolo3d')

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
    file_lists = natsort.natsorted(os.listdir(configs.pretrained_path))

    # file_lists = ["/mnt/8TB/PJW/BG-paper_weight/ComplexYOLO3D/A9/ComplexYOLO3D_baseline_best_weight_a9.pth"]
    
    max_ap = 0.0
    max_epoch = 0
    count = 0
    
    for pretrained_weight in file_lists:
        
        if "Model" in pretrained_weight:
            
            if count > 20:
            
                model = attempt_load(os.path.join(configs.pretrained_path,pretrained_weight), map_location=configs.device)  # load FP32 model
                # summary(model,torch.rand(1,3,608,608).to(configs.device))  # create
                model.fuse().eval()
                print("\nStart computing mAP... {}\n".format(pretrained_weight))
                
                precision, recall, AP, f1, ap_class ,bev_precision,bev_recall, bev_AP, bev_f1, bev_ap_class = evaluate_mAP(val_dataloader, model, configs, None)
                print("\nDone computing mAP...\n")
                for idx, cls in enumerate(ap_class):
                    print("\t>>>\t Class {} ({}): 3D_precision = {:.4f}, 3D_recall = {:.4f}, 3D_AP = {:.4f}, 3D_f1: {:.4f}, BEV_precision = {:.4f}, BEV_recall = {:.4f}, BEV_AP = {:.4f}, BEV_f1: {:.4f}".format(cls, \
                            class_names[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx], bev_precision[idx], bev_recall[idx], bev_AP[idx], bev_f1[idx]))

                print("\nmAP: {} BEV mAP: {}\n".format(AP.mean(),bev_AP.mean()))
                
                if AP.mean() >= max_ap:
                    max_ap_class = ap_class
                    max_precision = precision
                    max_recall = recall
                    max_AP = AP
                    max_f1 = f1
                    max_bev_precision = bev_precision
                    max_bev_recall = bev_recall
                    max_bev_AP = bev_AP
                    max_bev_f1 = bev_f1
                    max_bev_ap_class = bev_ap_class
                    max_ap = AP.mean()
                    max_epoch = pretrained_weight
            count = count + 1
                
    print("*****Maximum AP*****\n")    
    print("max_epoch : ", max_epoch, "\n")
    
    for idx, cls in enumerate(max_ap_class):
        print("\t>>>\t Class {} ({}): 3D_precision = {:.4f}, 3D_recall = {:.4f}, 3D_AP = {:.4f}, 3D_f1: {:.4f}, BEV_precision = {:.4f}, BEV_recall = {:.4f}, BEV_AP = {:.4f}, BEV_f1: {:.4f}".format(cls, \
            class_names[cls][:3], max_precision[idx], max_recall[idx], max_AP[idx], max_f1[idx], max_bev_precision[idx], max_bev_recall[idx], max_bev_AP[idx], max_bev_f1[idx]))
    
    print("\nMaximum mAP: {} BEV mAP: {}\n".format(max_AP.mean(),max_bev_AP.mean()))