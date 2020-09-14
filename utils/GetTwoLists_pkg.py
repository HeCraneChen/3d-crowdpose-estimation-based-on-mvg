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
from VisualizeAll3D import VisualizeAll3D, ThreeDTriangulate

def GetTwoLists(data_dict,anno_num,img_id,Width, Height,edge_thresh):
    """ 
    args:
    -data_dict the whole dictionary in json file
    -anno_num number of annotation
    -img_id id of annotation in the json file
    
    returns:
    -list_r a list of right foot coordinates, in the form of tuples (x,y,score) for right foot
    -list_l a list of left foot coordinates, in the form of tuples (x,y,score) for right foot
    """
    list_r = []
    list_l = []
    img_id = '{0:0=8d}.png'.format(img_id)
    for counter in range(anno_num):
        if data_dict[counter]["image_id"] == img_id:
            v = data_dict[counter]["keypoints"]
            anno_id = data_dict[counter]["id"]
            
            threshold = 0.2
            # two logics:
            #1, rule out low score points
            #2, if one side missing, copy another side
            list_r_y = [v[61],v[64],v[67]]
            list_r_y_score = [v[62], v[65], v[68]] 
            list_r_y_valid = [list_r_y[i] for i in range(3) if list_r_y_score[i] >= threshold ]
                   
            list_l_y = [v[52],v[55],v[58]]
            list_l_y_score = [v[53], v[56], v[59]]
            list_l_y_valid = [list_l_y[i] for i in range(3) if list_l_y_score[i] >= threshold ]
           
            if len(list_r_y_valid)+len(list_l_y_valid) ==0:
                continue # pass due to no valid points
            elif len(list_r_y_valid) ==0:
                traget_l_y = max(list_l_y_valid)
                traget_r_y = traget_l_y
            elif len(list_l_y_valid) ==0:
                traget_r_y = max(list_r_y_valid)
                traget_l_y = traget_r_y
            else:    
                traget_r_y = max(list_r_y_valid)
                traget_l_y = max(list_l_y_valid)
            
            ankle_r_y = Height - traget_r_y
            ankle_l_y = Height - traget_l_y
            
            if traget_r_y == v[61]:
                ankle_r_x = v[60]
            elif traget_r_y == v[64]:
                ankle_r_x = v[63] 
            elif traget_r_y == v[67]:
                ankle_r_x = v[66]
            else:
                #this means right copy left 
                ankle_r_x = -1
                                           #scores [53] [56] [59]
            if traget_l_y == v[52]:
                ankle_l_x = v[51]
            elif traget_l_y == v[55]:
                ankle_l_x = v[54] 
            elif traget_l_y == v[58]:
                ankle_l_x = v[57]
            else:
                #this means left copy right 
                ankle_l_x = -1  
                
            # copy y value for missing side 
            if ankle_l_x == -1:
                ankle_l_x = ankle_r_x
            elif ankle_r_x == -1:
                ankle_r_x = ankle_l_x
            X_p_r = (ankle_r_x,ankle_r_y,anno_id)
            X_p_l = (ankle_l_x,ankle_l_y,anno_id)
            
            if X_p_r[0] >= (0 + edge_thresh) and X_p_r[0] <= (Width - edge_thresh) and X_p_r[1] >= (0 + edge_thresh) and X_p_r[1] <= (Height - edge_thresh) and\
            X_p_l[0] >= (0 + edge_thresh) and X_p_l[0] <= (Width - edge_thresh) and X_p_l[1] >= (0 + edge_thresh) and X_p_l[1] <= (Height - edge_thresh):
                list_r.append(X_p_r)
                list_l.append(X_p_l)
    return list_r,list_l #coordinate origin is at the lower left corner

def WriteTexts(directory,counter,list_r,list_l):
    if not os.path.exists(directory):
        os.makedirs(directory)
    counter = counter + 5        
    file = open(directory + '/right.txt','w') 
    file.write(str(list_r))
    file.close()
    file = open(directory + '/left.txt','w') 
    file.write(str(list_l))
    file.close()
    #print('finish writing a folder')
    
def transFeet(ankle_x,ankle_y,H, Height):
    X = np.array([[ankle_x],[ankle_y],[1]])
    X_p = H.dot(X)  
    X_out = (0,0)
    thresh = 1e-5
    thresh = -300
    if abs(X_p[2]-0.0)>thresh:
        X_p = np.array([[X_p[0]/X_p[2]],[X_p[1]/X_p[2]],[1]])
        X_p[0] = X_p.item(0).astype(int)
        X_p[1] = Height - X_p.item(1).astype(int)
        X_p[2] = X_p.item(2)
        X_out = (int(X_p[0]),int(X_p[1]))
    return X_out #coordinate origin is at the upper left corner

def GetProjectLists(data_dict,anno_num,img_id,H,Width, Height, edge_thresh):
    list_r = []
    list_l = []
    img_id = '{0:0=8d}.png'.format(img_id)
    for counter in range(anno_num):
        if data_dict[counter]["image_id"] == img_id:
            v = data_dict[counter]["keypoints"]
            anno_id = data_dict[counter]["id"]   #scores [62] [65] [68] 
            
            threshold = 0.2
            # two logics:
            #1, rule out low score points
            #2, if one side missing, copy another side
            list_r_y = [v[61],v[64],v[67]]
            list_r_y_score = [v[62], v[65], v[68]] 
            list_r_y_valid = [list_r_y[i] for i in range(3) if list_r_y_score[i] >= threshold ]
                   
            list_l_y = [v[52],v[55],v[58]]
            list_l_y_score = [v[53], v[56], v[59]]
            list_l_y_valid = [list_l_y[i] for i in range(3) if list_l_y_score[i] >= threshold ]
           
            if len(list_r_y_valid)+len(list_l_y_valid) ==0:
                continue # pass due to no valid points
            elif len(list_r_y_valid) ==0:
                traget_l_y = max(list_l_y_valid)
                traget_r_y = traget_l_y
            elif len(list_l_y_valid) ==0:
                traget_r_y = max(list_r_y_valid)
                traget_l_y = traget_r_y
            else:    
                traget_r_y = max(list_r_y_valid)
                traget_l_y = max(list_l_y_valid)
            
            ankle_r_y = Height - traget_r_y
            ankle_l_y = Height - traget_l_y
            
            if traget_r_y == v[61]:
                ankle_r_x = v[60]
            elif traget_r_y == v[64]:
                ankle_r_x = v[63] 
            elif traget_r_y == v[67]:
                ankle_r_x = v[66]
            else:
                #this means right copy left 
                ankle_r_x = -1
                                           #scores [53] [56] [59]
            if traget_l_y == v[52]:
                ankle_l_x = v[51]
            elif traget_l_y == v[55]:
                ankle_l_x = v[54] 
            elif traget_l_y == v[58]:
                ankle_l_x = v[57]
            else:
                #this means left copy right 
                ankle_l_x = -1  
                
            # copy y value for missing side 
            if ankle_l_x == -1:
                ankle_l_x = ankle_r_x
            elif ankle_r_x == -1:
                ankle_r_x = ankle_l_x
                       
            X_p_l0 = transFeet(ankle_l_x,ankle_l_y,H, Height)
            X_p_r0 = transFeet(ankle_r_x,ankle_r_y,H, Height)
            X_p_l = (X_p_l0[0], Height - X_p_l0[1], anno_id)
            X_p_r = (X_p_r0[0], Height - X_p_r0[1], anno_id)
            #print('X_p_l0',X_p_l[0])
            if X_p_l[0] > (0 + edge_thresh) and X_p_l[0] < (Width - edge_thresh) and X_p_l[1] > (0 + edge_thresh) and X_p_l[1] < (Height - edge_thresh) and X_p_r[0] > (0 + edge_thresh) and X_p_r[0] < (Width - edge_thresh) and X_p_r[1] > (0 + edge_thresh) and X_p_r[1] < (Height - edge_thresh):
                list_l.append(X_p_l)              
                list_r.append(X_p_r)  
                #print(X_p_r)
    return list_r,list_l #coordinate origin is at the lower left corner   