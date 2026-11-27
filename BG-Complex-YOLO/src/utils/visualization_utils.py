import sys
import math

import numpy as np
import cv2

sys.path.append('../')

from data_process import kitti_data_utils, kitti_bev_utils, transformation, a9_bev_utils, a9_data_utils
import config.kitti_config as cnf



def merge_rgb_to_bev(img_rgb, img_rgb_2, img_bev, img_bev_2,output_width):
    img_rgb_h, img_rgb_w = img_rgb.shape[:2]
    ratio_rgb = output_width / img_rgb_w
    output_rgb_h = int(ratio_rgb * img_rgb_h)
    ret_img_rgb = cv2.resize(img_rgb, (output_width, output_rgb_h))
    ret_img_rgb_2 = cv2.resize(img_rgb_2,(output_width, output_rgb_h))

    img_bev_h, img_bev_w = img_bev.shape[:2]
    ratio_bev = output_width / img_bev_w
    output_bev_h = int(ratio_bev * img_bev_h)

    ret_img_bev = cv2.resize(img_bev, (output_width, output_bev_h))
    ret_img_bev_2 = cv2.resize(img_bev_2, (output_width, output_bev_h))
    
    out_img = np.zeros((output_rgb_h + output_bev_h, output_width + output_width, 3), dtype=np.uint8)
    # Upper: RGB --> BEV
    out_img[:output_rgb_h, output_width : 2 * output_width] = ret_img_rgb
    out_img[output_rgb_h:, output_width : 2 * output_width] = ret_img_bev

    out_img[:output_rgb_h, :output_width] = ret_img_rgb_2
    out_img[output_rgb_h:, :output_width] = ret_img_bev_2

    return out_img

def merge_rgb_to_bev_with_gt(img_rgb, img_rgb_2, img_bev, img_bev_gt, output_width):

    img_rgb_h, img_rgb_w = img_rgb.shape[:2]
    ratio_rgb = output_width / img_rgb_w
    output_rgb_h = int(ratio_rgb * img_rgb_h)
    ret_img_rgb = cv2.resize(img_rgb, (output_width, output_rgb_h))
    ret_img_rgb_2 = cv2.resize(img_rgb_2,(output_width, output_rgb_h))

    img_bev_h, img_bev_w = img_bev.shape[:2]
    ratio_bev = output_width / img_bev_w
    output_bev_h = int(ratio_bev * img_bev_h)

    ret_img_bev = cv2.resize(img_bev, (output_width, output_bev_h))
    ret_img_bev_gt = cv2.resize(img_bev_gt, (output_width, output_bev_h))

    out_img = np.zeros((output_rgb_h + output_bev_h, output_width + output_width, 3), dtype=np.uint8)
    # Upper: RGB --> BEV
    out_img[:output_rgb_h, output_width : 2 * output_width] = ret_img_rgb
    out_img[output_rgb_h:, output_width : 2 * output_width] = ret_img_bev

    out_img[:output_rgb_h, :output_width] = ret_img_rgb_2
    out_img[output_rgb_h:, :output_width] = ret_img_bev_gt 

    return out_img

def show_image_with_boxes_a9(img, objects, calib, show3d=False):
    ''' Show image with 2D bounding boxes '''

    img2 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.type == 'OTHER': continue
        # cv2.rectangle(img2, (int(obj.xmin),int(obj.ymin)),
        #    (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
        box3d_pts_2d, box3d_pts_3d = a9_data_utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            img2 = a9_data_utils.draw_projected_box3d(img2, box3d_pts_2d, cnf_a9.colors[obj.cls_id])
 
    if show3d:
        cv2.imshow("img", img2)
    return img2



def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=0.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def show_image_with_boxes(img, objects, calib, show3d=False):
    ''' Show image with 2D bounding boxes '''

    img2 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.type == 'DontCare': continue
        # cv2.rectangle(img2, (int(obj.xmin),int(obj.ymin)),
        #    (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
        box3d_pts_2d, box3d_pts_3d = kitti_data_utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            img2 = kitti_data_utils.draw_projected_box3d(img2, box3d_pts_2d, cnf.colors[obj.cls_id])
    if show3d:
        cv2.imshow("img", img2)
    return img2



def merge_rgb_to_bev(img_rgb, img_rgb_2, img_bev, img_bev_2,output_width):
    img_rgb_h, img_rgb_w = img_rgb.shape[:2]
    ratio_rgb = output_width / img_rgb_w
    output_rgb_h = int(ratio_rgb * img_rgb_h)
    ret_img_rgb = cv2.resize(img_rgb, (output_width, output_rgb_h))
    ret_img_rgb_2 = cv2.resize(img_rgb_2,(output_width, output_rgb_h))

    img_bev_h, img_bev_w = img_bev.shape[:2]
    ratio_bev = output_width / img_bev_w
    output_bev_h = int(ratio_bev * img_bev_h)

    ret_img_bev = cv2.resize(img_bev, (output_width, output_bev_h))
    ret_img_bev_2 = cv2.resize(img_bev_2, (output_width, output_bev_h))
    
    out_img = np.zeros((output_rgb_h + output_bev_h, output_width + output_width, 3), dtype=np.uint8)
    # Upper: RGB --> BEV
    out_img[:output_rgb_h, output_width : 2 * output_width] = ret_img_rgb
    out_img[output_rgb_h:, output_width : 2 * output_width] = ret_img_bev

    out_img[:output_rgb_h, :output_width] = ret_img_rgb_2
    out_img[output_rgb_h:, :output_width] = ret_img_bev_2

    return out_img

def merge_predition_gt_bev(img_bev, img_bev_2,output_width):
    
    img_bev_h, img_bev_w = img_bev.shape[:2]
    ratio_bev = output_width / img_bev_w
    output_bev_h = int(ratio_bev * img_bev_h)

    ret_img_bev = cv2.resize(img_bev, (output_width, output_bev_h))
    ret_img_bev_2 = cv2.resize(img_bev_2, (output_width, output_bev_h))
    
    out_img = cv2.hconcat([ret_img_bev,ret_img_bev_2])
    

    return out_img


def invert_target(targets, calib, img_shape_2d, RGB_Map=None):
    predictions = targets.clone()
    predictions = kitti_bev_utils.inverse_yolo_target(predictions, cnf.boundary)
    if predictions.shape[0]:
        predictions[:, 1:] = transformation.lidar_to_camera_box(predictions[:, 1:], calib.V2C, calib.R0, calib.P)

    objects_new = []
    corners3d = []
    for index, l in enumerate(predictions):
        if l[0] == 0:
            str = "Car"
        elif l[0] == 1:
            str = "Pedestrian"
        elif l[0] == 2:
            str = "Cyclist"
        else:
            str = "Ignore"
        line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

        obj = kitti_data_utils.Object3d(line)
        obj.t = l[1:4]
        obj.h, obj.w, obj.l = l[4:7]
        obj.ry = np.arctan2(math.sin(l[7]), math.cos(l[7]))

        _, corners_3d = kitti_data_utils.compute_box_3d(obj, calib.P)
        corners3d.append(corners_3d)
        objects_new.append(obj)

    if len(corners3d) > 0:
        corners3d = np.array(corners3d)
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape_2d[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape_2d[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape_2d[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape_2d[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(img_boxes_w < img_shape_2d[1] * 0.8, img_boxes_h < img_shape_2d[0] * 0.8)

    for i, obj in enumerate(objects_new):
        x, z, ry = obj.t[0], obj.t[2], obj.ry
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        obj.alpha = alpha
        obj.box2d = img_boxes[i, :]

    if RGB_Map is not None:
        labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects_new)
        if not noObjectLabels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                               calib.P)  # convert rect cam to velo cord

        target = kitti_bev_utils.build_yolo_target(labels)
        kitti_bev_utils.draw_box_in_bev(RGB_Map, target)

    return objects_new


def invert_target_a9(targets, calib, img_shape_2d, RGB_Map=None):
    predictions = targets
    predictions = a9_bev_utils.inverse_yolo_target(predictions, cnf_a9.boundary)
    if predictions.shape[0]:
        predictions[:, 1:] = transformation.lidar_to_camera_box(predictions[:, 1:], calib.V2C, calib.R0, 
                                                                calib.P)

    objects_new = []
    corners3d = []

    for index, l in enumerate(predictions):
        if l[0] == 0:
            str = "CAR"
        elif l[0] == 1:
            str = "WHEELER"
        elif l[0] == 2:
            str = "PEDESTRIAN"
        else:
            str = "Ignore"                                                                         
        line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

        obj = a9_data_utils.Object3d(line)
        obj.t = l[1:4]
        obj.h, obj.w, obj.l = l[4:7]
        obj.ry = np.arctan2(math.sin(l[7]), math.cos(l[7]))

        _, corners_3d = a9_data_utils.compute_box_3d(obj, calib.P)
        corners3d.append(corners_3d)
        objects_new.append(obj)

    if len(corners3d) > 0:
        corners3d = np.array(corners3d)
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape_2d[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape_2d[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape_2d[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape_2d[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(img_boxes_w < img_shape_2d[1] * 0.8, img_boxes_h < img_shape_2d[0] * 0.8)

    for i, obj in enumerate(objects_new):
        x, z, ry = obj.t[0], obj.t[2], obj.ry
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        obj.alpha = alpha
        obj.box2d = img_boxes[i, :]

    if RGB_Map is not None:
        labels, noObjectLabels = a9_bev_utils.read_labels_for_bevbox(objects_new)
        if not noObjectLabels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                               calib.P)  # convert rect cam to velo cord

        target = a9_bev_utils.build_yolo_target(labels)
        a9_bev_utils.draw_box_in_bev(RGB_Map, target)

    return objects_new

def predictions_to_kitti_format(img_detections, calib, img_shape_2d, img_size, RGB_Map=None):
    predictions = []
    for detections in img_detections:
        if detections is None:
            continue
        # Rescale boxes to original image
        for x, y, z, h, w, l, im, re, *_, cls_pred in detections:
            predictions.append([cls_pred, x / img_size, y / img_size, z , h / img_size, w / img_size, l / img_size, im, re])

    predictions = kitti_bev_utils.inverse_yolo_target(np.array(predictions), cnf.boundary)
    if predictions.shape[0]:
        predictions[:, 1:] = transformation.lidar_to_camera_box(predictions[:, 1:], calib.V2C, calib.R0, calib.P)

    objects_new = []
    corners3d = []
    for index, l in enumerate(predictions):
        if l[0] == 0:
            str = "Car"
        elif l[0] == 1:
            str = "Pedestrian"
        elif l[0] == 2:
            str = "Cyclist"
        else:
            str = "Ignore"
        line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

        obj = kitti_data_utils.Object3d(line)
        obj.t = l[1:4]
        obj.h, obj.w, obj.l = l[4:7]
        obj.ry = np.arctan2(math.sin(l[7]), math.cos(l[7]))

        _, corners_3d = kitti_data_utils.compute_box_3d(obj, calib.P)
        corners3d.append(corners_3d)
        objects_new.append(obj)

    if len(corners3d) > 0:
        corners3d = np.array(corners3d)
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape_2d[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape_2d[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape_2d[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape_2d[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(img_boxes_w < img_shape_2d[1] * 0.8, img_boxes_h < img_shape_2d[0] * 0.8)

    for i, obj in enumerate(objects_new):
        x, z, ry = obj.t[0], obj.t[2], obj.ry
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        obj.alpha = alpha
        obj.box2d = img_boxes[i, :]

    if RGB_Map is not None:
        labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects_new)
        if not noObjectLabels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                               calib.P)  # convert rect cam to velo cord

        target = kitti_bev_utils.build_yolo_target(labels)
        kitti_bev_utils.draw_box_in_bev(RGB_Map, target)

    return objects_new