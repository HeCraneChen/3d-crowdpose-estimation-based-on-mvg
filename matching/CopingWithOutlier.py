import sys
import json
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import cv2
import pylab as pl
from numpy import linalg as LA
import random
import math
from mpl_toolkits.mplot3d import axes3d, Axes3D
from pose_optimize.multiview_geo import get_distance
from all_dataset_para import Get_P_from_dataset
from AccomodateDataset import accomoDataset

def CopeOutlier(pts3D, median_bones, ratio_thresh):
    """cope with outliers by two constraints: 
       1) check the cluster, vote the points that are ridiculously far away as outliers
       2) check the boneL constraint 
       median_bones: [nose_eye, eye_ear, LRshoulder, upperArm, lowerArm, LRhip, hip_knee, knee_ankle, ankle_heel, heel_bigtoe, heel_smalltoe, ear_ear, wrist_wrist, ankle_ankle, heel_heel, toe_toe]
    -Args:
    pts3D: 4 by n numpy array, rows 0,1,2,3 are x,y,z, row3 is always 1
    median_bones: a list of bone lengths
    
    -Returns:
    pts3D_inlier 4 by m numpy array, rows 0,1,2,3 are x,y,z, row3 is always 1
         
    """
    pts3D_new = pts3D
    people_num = int(pts3D.shape[1]/23)
    Dist = {}
    for counter1 in range (people_num):
        adjust_result = []
        start = counter1 * 23
        end = start + 23
        person = pts3D[:,start:end]
        person_list = person.T.reshape((1,-1))[0]
        for counter2 in range(23):
            Dist[counter2] = getdistance(person[:,counter2], person)
        sorted_Dict = sorted(Dist.items(), key=lambda item: item[1])
        for counter3 in range(1,len(sorted_Dict)):
            distance = sorted_Dict[counter3][1]
            ind_i = sorted_Dict[counter3][0]
            ratio = sorted_Dict[counter3][1] / sorted_Dict[counter3 - 1][1]
            if ratio > ratio_thresh:
                x = person[0,ind_i]
                y = person[1,ind_i]
                z = person[2,ind_i]
                outlier_coor = [x, y, z]
                outlier_index = sorted_Dict[counter3][0] 
                inlier_from_outlier = adjustbone(outlier_index, outlier_coor, median_bones, person_list)
                adjust_result.append((outlier_index,inlier_from_outlier))
        
        person_new = person # 4 by 23 numpy array
        for counter4 in range (len(adjust_result)):
            temp = np.asarray(adjust_result[counter4][1]).reshape(3,1)
            test = person_new[:,0]
            person_new[:,adjust_result[counter4][0]] = np.concatenate((temp, np.array([[1]])),axis = 0).reshape(4)
        if counter1 == 0:
            pts3D_new = person_new
        else:
            pts3D_new  = np.concatenate((pts3D_new, person_new), axis = 1)
    return pts3D_new

def getdistance(pt, pts):
    """get the distance of one point w.r.t all points
    -Args:
    pt: 3 by 1 numpy array
    pts: 3 by n numpy array
    """
    d = 0
    for counter in range (pts.shape[1]):
        d = d + LA.norm(pts[:,counter] - pt)
    return d

def adjustbone(outlier_index, outlier_coor, median_bones, person):
    """adjust the outlier, and convert it to inlier based on bone length
    
    -Args:
    outlier_index: int 
    outlier_coor: a list [x,y,z]
    person: a list [x1,y1,z1,x2,y2,z2,...]
    median_bones: [nose_eye, eye_ear, LRshoulder, upperArm, lowerArm, LRhip, hip_knee, knee_ankle, ankle_heel, heel_bigtoe, heel_smalltoe, ear_ear, wrist_wrist, ankle_ankle, heel_heel, toe_toe]
    -Returns:
    inlier_from_outlier: in the form of a list [x,y,z]
    """
    nose = person[0:3] #linked to Leye and Reye
    Leye = person[4:7] #linked to nose and Leaer
    Reye = person[8:11] #linked to nose and Rear
    Lear = person[12:15] #linked to Leye
    Rear = person[16:19] #linked to Reye
    Lshoulder = person[20:23] #linked to Rshoulder
    Rshoulder = person[24:27] #linked to Lshoulder
    Lelbow = person[28:31] #linked to Lshoulder and Lwrist
    Relbow = person[32:35] #linked to Rshoulder and Rwrist
    Lwrist = person[36:39] #linked to Lelbow
    Rwrist = person[40:43] #linked to Relbow   
    Lhip = person[44:47] #linked to Lknee and Rhip
    Rhip = person[48:51] #linked to Rknee and Lhip
    
    Lknee = person[52:55] #linked to Lhip and Lankle
    Rknee = person[56:59] #linked to Rhip and Rankle
    Lankle = person[60:63] #linked to Lheel and Lknee
    Rankle = person[64:67] #linked to Rheel and Rknee
    
    Lbigtoe = person[68:71] #linked to Lheel
    Lsmalltoe = person[72:75] #linked to Lheel
    Lheel = person[76:79] #linked to Lbigtoe and Lankle
    
    Rbigtoe = person[80:83] #linked to Rheel and
    Rsmalltoe = person[84:87] #linked to Rheel
    Rheel = person[88:91] #linked to Rbigtoe and Rankle
    
    
    if outlier_index == 0: #nose, use left eye and right eye to adjust
        normal_bone1 = median_bones[0]
        normal_bone2 = median_bones[0]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Leye, Reye, normal_bone1, normal_bone2)
        
    if outlier_index == 1: #left eye, use nose and left ear to adjust
        normal_bone1 = median_bones[0]
        normal_bone2 = median_bones[1]
        inlier_from_outlier = CalcInlier_two(outlier_coor, nose, Lear, normal_bone1, normal_bone2)
        
    if outlier_index == 2: #right eye, use nose and right ear to adjust
        normal_bone1 = median_bones[0]
        normal_bone2 = median_bones[1]
        inlier_from_outlier = CalcInlier_two(outlier_coor, nose, Rear, normal_bone1, normal_bone2)
        
    if outlier_index == 3: #Lear, use Leye and Rear to adjust
        normal_bone1 = median_bones[1]
        normal_bone2 = median_bones[11]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Leye, Rear, normal_bone1, normal_bone2)
        
    if outlier_index == 4: #Rear, use Reye  and Lear to adjust
        normal_bone1 = median_bones[1]
        normal_bone2 = median_bones[11]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Reye, Lear, normal_bone1, normal_bone2)
        
    if outlier_index == 5: #Lshoulder, linked to Rshoulder and Lelbow
        normal_bone1 = median_bones[2]
        normal_bone2 = median_bones[3]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Rshoulder, Lelbow, normal_bone1, normal_bone2)
        
    if outlier_index == 6: #Rshoulder, linked to Lshoulder and Relbow
        normal_bone1 = median_bones[2]
        normal_bone2 = median_bones[3]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Lshoulder, Relbow, normal_bone1, normal_bone2)
        
    if outlier_index == 7: #Lelbow, linked to Lshoulder and Lwrist
        normal_bone1 = median_bones[3]
        normal_bone2 = median_bones[4]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Lshoulder, Lwrist, normal_bone1, normal_bone2)
        
    if outlier_index == 8: #Relbow, linked to Rshoulder and Rwrist
        normal_bone1 = median_bones[3]
        normal_bone2 = median_bones[4]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Rshoulder, Rwrist, normal_bone1, normal_bone2)
        
    if outlier_index == 9: #Lwrist, linked to Lelbow and Rwrist
        normal_bone1 = median_bones[4]
        normal_bone2 = median_bones[12]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Lelbow, Rwrist, normal_bone1, normal_bone2)
        
    if outlier_index == 10: #Rwrist, linked to Relbow and Lwrist
        normal_bone1 = median_bones[4]
        normal_bone2 = median_bones[12]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Relbow, Lwrist, normal_bone1, normal_bone2)
        
    if outlier_index == 11: #Lhip, linked to Lknee and Rhip
        normal_bone1 = median_bones[5]
        normal_bone2 = median_bones[6]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Rhip, Lknee, normal_bone1, normal_bone2)
        
    if outlier_index == 12: #Rhip, linked to Rknee and Lhip
        normal_bone1 = median_bones[5]
        normal_bone2 = median_bones[6]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Lhip, Rknee, normal_bone1, normal_bone2)
        
    if outlier_index == 13: #Lknee, linked to Lhip and Lankle
        normal_bone1 = median_bones[6]
        normal_bone2 = median_bones[7]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Lhip, Lankle, normal_bone1, normal_bone2)
        
    if outlier_index == 14: #Rknee, linked to Rhip and Rankle
        normal_bone1 = median_bones[6]
        normal_bone2 = median_bones[7]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Rhip, Rankle, normal_bone1, normal_bone2)
        
    if outlier_index == 15: #Lankle, linked to Lheel and Lknee
        normal_bone1 = median_bones[8]
        normal_bone2 = median_bones[7]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Lheel, Lknee, normal_bone1, normal_bone2)
        
    if outlier_index == 16: #Rankle, linked to Rheel and Rknee
        normal_bone1 =  median_bones[8]
        normal_bone2 =  median_bones[7]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Rheel, Rknee, normal_bone1, normal_bone2)
        
    if outlier_index == 17: #Lbigtoe, linked to Lheel and stride
        normal_bone1 =  median_bones[9]
        stride = (median_bones[13] + median_bones[14] + median_bones[15]) / 3
        normal_bone2 = stride
        ave_Rfoot = ((np.asarray(Rheel) + np.asarray(Rbigtoe) + np.asarray(Rsmalltoe) + np.asarray(Rankle))/4).tolist()
        inlier_from_outlier = CalcInlier_two(outlier_coor, Lheel, ave_Rfoot, normal_bone1, normal_bone2)
        
    if outlier_index == 18: #Lsmalltoe, linked to Lheel and stride
        normal_bone1 =  median_bones[10]
        stride = (median_bones[13] + median_bones[14] + median_bones[15]) / 3
        normal_bone2 = stride
        ave_Rfoot = ((np.asarray(Rheel) + np.asarray(Rbigtoe) + np.asarray(Rsmalltoe) + np.asarray(Rankle))/4).tolist()
        inlier_from_outlier = CalcInlier_two(outlier_coor, Lheel, ave_Rfoot, normal_bone1, normal_bone2)
        
    if outlier_index == 19: #Lheel, linked to Lbigtoe and Lankle
        normal_bone1 =  median_bones[9]
        normal_bone2 =  median_bones[8]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Lbigtoe, Lankle, normal_bone1, normal_bone2)
        
    if outlier_index == 20: #Rbigtoe, linked to Rheel and stride
        normal_bone1 =  median_bones[9]
        stride = (median_bones[13] + median_bones[14] + median_bones[15]) / 3
        normal_bone2 = stride
        ave_Lfoot = ((np.asarray(Lheel) + np.asarray(Lbigtoe) + np.asarray(Lsmalltoe) + np.asarray(Lankle))/4).tolist()
        inlier_from_outlier = CalcInlier_two(outlier_coor, Rheel, ave_Lfoot, normal_bone1, normal_bone2)
        
    if outlier_index == 21: #Rsmalltoe, linked to Rheel
        normal_bone1 =  median_bones[10]
        stride = (median_bones[13] + median_bones[14] + median_bones[15]) / 3
        normal_bone2 = stride
        ave_Lfoot = ((np.asarray(Lheel) + np.asarray(Lbigtoe) + np.asarray(Lsmalltoe) + np.asarray(Lankle))/4).tolist()
        inlier_from_outlier = CalcInlier_two(outlier_coor, Rheel, ave_Lfoot, normal_bone1, normal_bone2)
        
    if outlier_index == 22: #Rheel, linked to Rbigtoe and Rankle
        normal_bone1 =  median_bones[9]
        normal_bone2 =  median_bones[8]
        inlier_from_outlier = CalcInlier_two(outlier_coor, Rbigtoe, Rankle, normal_bone1, normal_bone2)                       
    return inlier_from_outlier

def CalcInlier_one(outlier_coor, connect1,normal_bone1):
    """calculate the coor of inlier based on one connected point
    -Args:
    outlier_coor, connect1: a list in the form of [x,y,z]
    normal_bone1: constants
    
    -Returns:
    inlier_from_outlier: a list in the form of [x,y,z]
    """
    minial_dis = 0.5
    if LA.norm(np.asarray(connect1) - np.asarray(outlier_coor)) < minial_dis :
        return outlier_coor
    p1 = normal_bone1 / LA.norm(np.asarray(connect1) - np.asarray(outlier_coor))
    inlier_from_outlier = (np.asarray(connect1) * (1 - p1) + np.asarray(outlier_coor) * p1).tolist()        
    return inlier_from_outlier

def CalcInlier_two(outlier_coor, connect1, connect2, normal_bone1, normal_bone2):
    """calculate the coor of inlier based on two connected points
    -Args:
    outlier_coor, connect1, connect2: a list in the form of [x,y,z]
    normal_bone1, normal_bone2: constants
    
    -Returns:
    inlier_from_outlier: a list in the form of [x,y,z]
    """
    minial_dis = 0.5
    if LA.norm(np.asarray(connect1) - np.asarray(outlier_coor)) < minial_dis \
       or LA.norm(np.asarray(connect2) - np.asarray(outlier_coor)) < minial_dis :
        return outlier_coor
    p1 = normal_bone1 / LA.norm(np.asarray(connect1) - np.asarray(outlier_coor))
    p2 = normal_bone2 / LA.norm(np.asarray(connect2) - np.asarray(outlier_coor))
    inlier_from_outlier1 = np.asarray(connect1) * (1 - p1) + np.asarray(outlier_coor) * p1      
    inlier_from_outlier2 = np.asarray(connect2) * (1 - p2) + np.asarray(outlier_coor) * p2
    inlier_from_outlier = (0.5 * (inlier_from_outlier1 + inlier_from_outlier2)).tolist() 
    return inlier_from_outlier