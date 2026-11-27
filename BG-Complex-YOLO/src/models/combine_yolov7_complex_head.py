"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Refer: https://github.com/Tianxiaomo/pytorch-YOLOv4
"""

import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as  np

sys.path.append('../')

from models.yolo_layer import YoloLayer
from models.darknet_utils import parse_cfg, print_cfg, load_fc, load_conv_bn, load_conv
from utils.torch_utils import to_cpu

def combine_yolov7_head(model):
    print(model[-1])
    anchor_masks = [int(i) for i in block['mask'].split(',')]
    anchors = [float(i) for i in block['anchors'].split(',')]
    anchors = [(anchors[i], anchors[i + 1], math.sin(anchors[i + 2]), math.cos(anchors[i + 2])) for i in
                range(0, len(anchors), 3)]
    anchors = [anchors[i] for i in anchor_masks]

    num_classes = int(block['classes'])
    self.num_classes = num_classes
    scale_x_y = float(block['scale_x_y'])
    ignore_thresh = float(block['ignore_thresh'])
    

    yolo_layer = YoloLayer(num_classes=num_classes, anchors=anchors, stride=prev_stride,
                            scale_x_y=scale_x_y, ignore_thresh=ignore_thresh)

    out_filters.append(prev_filters)
    out_strides.append(prev_stride)
    models.append(yolo_layer)    

    return model