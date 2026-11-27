import os
import numpy as np
import cv2
import random
from pypcd4 import PointCloud
from tools import *

save_path = "./training/background_RC"
txt_file = open("./ImageSets/train.txt","r")

lines = txt_file.readlines()
random.shuffle(lines)

cnt_117 = 0
cnt_120 = 0
cnt_136 = 0
cnt_139 = 0

n117_background_r = np.zeros([1,BEV_HEIGHT,BEV_WIDTH],dtype = np.float32)
n117_background_g = np.zeros([1,BEV_HEIGHT,BEV_WIDTH],dtype = np.float32)
n117_background_b = np.zeros([1,BEV_HEIGHT,BEV_WIDTH],dtype = np.float32)

n120_background_r = np.zeros([1,BEV_HEIGHT,BEV_WIDTH],dtype = np.float32)
n120_background_g = np.zeros([1,BEV_HEIGHT,BEV_WIDTH],dtype = np.float32)
n120_background_b = np.zeros([1,BEV_HEIGHT,BEV_WIDTH],dtype = np.float32)

n136_background_r = np.zeros([1,BEV_HEIGHT,BEV_WIDTH],dtype = np.float32)
n136_background_g = np.zeros([1,BEV_HEIGHT,BEV_WIDTH],dtype = np.float32)
n136_background_b = np.zeros([1,BEV_HEIGHT,BEV_WIDTH],dtype = np.float32)

n139_background_r = np.zeros([1,BEV_HEIGHT,BEV_WIDTH],dtype = np.float32)
n139_background_g = np.zeros([1,BEV_HEIGHT,BEV_WIDTH],dtype = np.float32)
n139_background_b = np.zeros([1,BEV_HEIGHT,BEV_WIDTH],dtype = np.float32)

sample_num = 200

for line in lines:
    line = line.rstrip()
    pc_path = os.path.join("./training/pointcloud",line+".bin")

    if ("117" in line) and (cnt_117 < sample_num):
        pc = np.fromfile(pc_path,dtype = np.float32).reshape(-1,4)
        pc_reduce = removePoints(pc, boundary)
        rgb_map = makeBVFeature(pc_reduce, DISCRETIZATION, boundary)
        
        if cnt_117 == 0:
            n117_background_r = n117_background_r + rgb_map[0,:,:]
            n117_background_g = n117_background_g + rgb_map[1,:,:]
            n117_background_b = n117_background_b + rgb_map[2,:,:]
            
        else:
            n117_background_r = np.concatenate([n117_background_r,rgb_map[np.newaxis,0,:]],axis = 0)
            n117_background_g = np.concatenate([n117_background_g,rgb_map[np.newaxis,1,:]],axis = 0)
            n117_background_b = np.concatenate([n117_background_b,rgb_map[np.newaxis,2,:]],axis = 0)
            
        cnt_117 += 1
        
    elif ("120" in line) and (cnt_120 < sample_num):
        pc = np.fromfile(pc_path,dtype = np.float32).reshape(-1,4)
        pc_reduce = removePoints(pc, boundary)
        rgb_map = makeBVFeature(pc_reduce, DISCRETIZATION, boundary)
    
        if cnt_120 == 0:
            n120_background_r = n120_background_r + rgb_map[0,:,:]
            n120_background_g = n120_background_g + rgb_map[1,:,:]
            n120_background_b = n120_background_b + rgb_map[2,:,:]
            
        else:
            n120_background_r = np.concatenate([n120_background_r,rgb_map[np.newaxis,0,:]],axis = 0)
            n120_background_g = np.concatenate([n120_background_g,rgb_map[np.newaxis,1,:]],axis = 0)
            n120_background_b = np.concatenate([n120_background_b,rgb_map[np.newaxis,2,:]],axis = 0)
            
        cnt_120 += 1
        
    elif ("136" in line) and (cnt_136 < sample_num):
        pc = np.fromfile(pc_path,dtype = np.float32).reshape(-1,4)
        pc_reduce = removePoints(pc, boundary)
        rgb_map = makeBVFeature(pc_reduce, DISCRETIZATION, boundary)
        if cnt_136 == 0:
            n136_background_r = n136_background_r + rgb_map[0,:,:]
            n136_background_g = n136_background_g + rgb_map[1,:,:]
            n136_background_b = n136_background_b + rgb_map[2,:,:]
            
        else:
            n136_background_r = np.concatenate([n136_background_r,rgb_map[np.newaxis,0,:]],axis = 0)
            n136_background_g = np.concatenate([n136_background_g,rgb_map[np.newaxis,1,:]],axis = 0)
            n136_background_b = np.concatenate([n136_background_b,rgb_map[np.newaxis,2,:]],axis = 0)
            
        cnt_136 += 1
        
    elif ("139" in line) and (cnt_139 < sample_num):
        pc = np.fromfile(pc_path,dtype = np.float32).reshape(-1,4)
        pc_reduce = removePoints(pc, boundary)
        rgb_map = makeBVFeature(pc_reduce, DISCRETIZATION, boundary)
        if cnt_139 == 0:
            n139_background_r = n139_background_r + rgb_map[0,:,:]
            n139_background_g = n139_background_g + rgb_map[1,:,:]
            n139_background_b = n139_background_b + rgb_map[2,:,:]
            
        else:
            n139_background_r = np.concatenate([n139_background_r,rgb_map[np.newaxis,0,:]],axis = 0)
            n139_background_g = np.concatenate([n139_background_g,rgb_map[np.newaxis,1,:]],axis = 0)
            n139_background_b = np.concatenate([n139_background_b,rgb_map[np.newaxis,2,:]],axis = 0)
            
        cnt_139 += 1

n117_background_r_median = np.median(n117_background_r,axis = 0)
n117_background_g_median = np.median(n117_background_g,axis = 0)
n117_background_b_median = np.median(n117_background_b,axis = 0)

n120_background_r_median = np.median(n120_background_r,axis = 0)
n120_background_g_median = np.median(n120_background_g,axis = 0)
n120_background_b_median = np.median(n120_background_b,axis = 0)

n136_background_r_median = np.median(n136_background_r,axis = 0)
n136_background_g_median = np.median(n136_background_g,axis = 0)
n136_background_b_median = np.median(n136_background_b,axis = 0)

n139_background_r_median = np.median(n139_background_r,axis = 0)
n139_background_g_median = np.median(n139_background_g,axis = 0)
n139_background_b_median = np.median(n139_background_b,axis = 0)

n117_background_rgb_median = np.concatenate([n117_background_r_median[np.newaxis,:],n117_background_g_median[np.newaxis,:],n117_background_b_median[np.newaxis,:]],axis = 0)
n120_background_rgb_median = np.concatenate([n120_background_r_median[np.newaxis,:],n120_background_g_median[np.newaxis,:],n120_background_b_median[np.newaxis,:]],axis = 0)
n136_background_rgb_median = np.concatenate([n136_background_r_median[np.newaxis,:],n136_background_g_median[np.newaxis,:],n136_background_b_median[np.newaxis,:]],axis = 0)
n139_background_rgb_median = np.concatenate([n139_background_r_median[np.newaxis,:],n139_background_g_median[np.newaxis,:],n139_background_b_median[np.newaxis,:]],axis = 0)


n117_background_r_mean = np.mean(n117_background_r,axis = 0)
n117_background_g_mean = np.mean(n117_background_g,axis = 0)
n117_background_b_mean = np.mean(n117_background_b,axis = 0)

n120_background_r_mean = np.mean(n120_background_r,axis = 0)
n120_background_g_mean = np.mean(n120_background_g,axis = 0)
n120_background_b_mean = np.mean(n120_background_b,axis = 0)

n136_background_r_mean = np.mean(n136_background_r,axis = 0)
n136_background_g_mean = np.mean(n136_background_g,axis = 0)
n136_background_b_mean = np.mean(n136_background_b,axis = 0)

n139_background_r_mean = np.mean(n139_background_r,axis = 0)
n139_background_g_mean = np.mean(n139_background_g,axis = 0)
n139_background_b_mean = np.mean(n139_background_b,axis = 0)

n117_background_rgb_mean = np.concatenate([n117_background_r_mean[np.newaxis,:],n117_background_g_mean[np.newaxis,:],n117_background_b_mean[np.newaxis,:]],axis = 0)
n120_background_rgb_mean = np.concatenate([n120_background_r_mean[np.newaxis,:],n120_background_g_mean[np.newaxis,:],n120_background_b_mean[np.newaxis,:]],axis = 0)
n136_background_rgb_mean = np.concatenate([n136_background_r_mean[np.newaxis,:],n136_background_g_mean[np.newaxis,:],n136_background_b_mean[np.newaxis,:]],axis = 0)
n139_background_rgb_mean = np.concatenate([n139_background_r_mean[np.newaxis,:],n139_background_g_mean[np.newaxis,:],n139_background_b_mean[np.newaxis,:]],axis = 0)


np.save(os.path.join(save_path,"RC_117_background_median_roi_100.npy"),n117_background_rgb_median)
np.save(os.path.join(save_path,"RC_120_background_median_roi_100.npy"),n120_background_rgb_median)
np.save(os.path.join(save_path,"RC_136_background_median_roi_100.npy"),n136_background_rgb_median)
np.save(os.path.join(save_path,"RC_139_background_median_roi_100.npy"),n139_background_rgb_median)

np.save(os.path.join(save_path,"RC_117_background_mean_roi_100.npy"),n117_background_rgb_mean)
np.save(os.path.join(save_path,"RC_120_background_mean_roi_100.npy"),n120_background_rgb_mean)
np.save(os.path.join(save_path,"RC_136_background_mean_roi_100.npy"),n136_background_rgb_mean)
np.save(os.path.join(save_path,"RC_139_background_mean_roi_100.npy"),n139_background_rgb_mean)

print("saved!\n")

n117_background_rgb_mean *= 255
n120_background_rgb_mean *= 255
n136_background_rgb_mean *= 255
n139_background_rgb_mean *= 255
n117_background_rgb_median *= 255
n120_background_rgb_median *= 255
n136_background_rgb_median *= 255
n139_background_rgb_median *= 255


n117_background_rgb_median = n117_background_rgb_median.transpose(1,2,0).astype(np.uint8)
n120_background_rgb_median = n120_background_rgb_median.transpose(1,2,0).astype(np.uint8)
n136_background_rgb_median = n136_background_rgb_median.transpose(1,2,0).astype(np.uint8)
n139_background_rgb_median = n139_background_rgb_median.transpose(1,2,0).astype(np.uint8)
n117_background_rgb_mean = n117_background_rgb_mean.transpose(1,2,0).astype(np.uint8)
n120_background_rgb_mean = n120_background_rgb_mean.transpose(1,2,0).astype(np.uint8)
n136_background_rgb_mean = n136_background_rgb_mean.transpose(1,2,0).astype(np.uint8)
n139_background_rgb_mean = n139_background_rgb_mean.transpose(1,2,0).astype(np.uint8)


n117_background_rgb_median = cv2.rotate(n117_background_rgb_median, cv2.ROTATE_180)
n120_background_rgb_median = cv2.rotate(n120_background_rgb_median, cv2.ROTATE_180)
n136_background_rgb_median = cv2.rotate(n136_background_rgb_median, cv2.ROTATE_180)
n139_background_rgb_median = cv2.rotate(n139_background_rgb_median, cv2.ROTATE_180)
n117_background_rgb_mean = cv2.rotate(n117_background_rgb_mean, cv2.ROTATE_180)
n120_background_rgb_mean = cv2.rotate(n120_background_rgb_mean, cv2.ROTATE_180)
n136_background_rgb_mean = cv2.rotate(n136_background_rgb_mean, cv2.ROTATE_180)
n139_background_rgb_mean = cv2.rotate(n139_background_rgb_mean, cv2.ROTATE_180)

cv2.imwrite(os.path.join(save_path,"n117_background_rgb_median_roi_100.png"),n117_background_rgb_median)
cv2.imwrite(os.path.join(save_path,"n120_background_rgb_median_roi_100.png"),n120_background_rgb_median)
cv2.imwrite(os.path.join(save_path,"n136_background_rgb_median_roi_100.png"),n136_background_rgb_median)
cv2.imwrite(os.path.join(save_path,"n139_background_rgb_median_roi_100.png"),n139_background_rgb_median)

cv2.imwrite(os.path.join(save_path,"n117_background_rgb_mean_roi_100.png"),n117_background_rgb_mean)
cv2.imwrite(os.path.join(save_path,"n120_background_rgb_mean_roi_100.png"),n120_background_rgb_mean)
cv2.imwrite(os.path.join(save_path,"n136_background_rgb_mean_roi_100.png"),n136_background_rgb_mean)
cv2.imwrite(os.path.join(save_path,"n139_background_rgb_mean_roi_100.png"),n139_background_rgb_mean)


