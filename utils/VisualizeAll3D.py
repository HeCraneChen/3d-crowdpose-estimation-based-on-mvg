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
from all_dataset_para import Get_P_from_dataset
from pose_optimize.multiview_geo import get_distance, fund_mat, recover_rti, orth_point_line, reconstruct_point_mini

def ThreeDTriangulate(P4,P6,KP4,KP6,match_pair):
    """
        this function gets 3D points from matching points of two images
    """
    X = cv2.triangulatePoints(P4, P6, KP4, KP6)
#     X = reconstruct_point_mini(P4, P6, KP4, KP6)
    pts3D = X/X[3]
    return pts3D
             
def ConnectSkeleton(i,j,co,pts3D,ax):
    x = [pts3D[0, i], pts3D[0, j]]
    y = [pts3D[1, i], pts3D[1, j]]
    z  = [pts3D[2, i], pts3D[2, j]]
    ax.plot(x,y,z,color=co )
    return x,y,z

def SingularityElimination(p1,p2,p3,p4):
    d12 = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2
    d23 = (p2[0] - p3[0])**2 + (p2[1] - p3[1])**2 + (p2[2] - p3[2])**2
    d13 = (p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2
    d14 = (p1[0] - p4[0])**2 + (p1[1] - p4[1])**2 + (p1[2] - p4[2])**2
    d24 = (p2[0] - p4[0])**2 + (p2[1] - p4[1])**2 + (p2[2] - p4[2])**2
    d34 = (p3[0] - p4[0])**2 + (p3[1] - p4[1])**2 + (p3[2] - p4[2])**2
    d = np.array([d12,d13,d14,d23,d24,d34])
    min_ind_s = np.unravel_index(np.argmin(d, axis=None), d.shape)
    min_ind = min_ind_s[0]    
    if min_ind == 0:
        p_f = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), 0.5 * (p1[2] + p2[2]))
    if min_ind == 1:
        p_f = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]), 0.5 * (p1[2] + p3[2]))
    if min_ind == 2:
        p_f = (0.5 * (p1[0] + p4[0]), 0.5 * (p1[1] + p4[1]), 0.5 * (p1[2] + p4[2]))
    if min_ind == 3:
        p_f = (0.5 * (p2[0] + p3[0]), 0.5 * (p2[1] + p3[1]), 0.5 * (p2[2] + p3[2]))
    if min_ind == 4:
        p_f = (0.5 * (p2[0] + p4[0]), 0.5 * (p2[1] + p4[1]), 0.5 * (p2[2] + p4[2]))
    if min_ind == 5:
        p_f = (0.5 * (p3[0] + p4[0]), 0.5 * (p4[1] + p4[1]), 0.5 * (p3[2] + p4[2]))
    return p_f
    
        
def VisualizeOne3D(pts3D_person,co,ax,dataset_name, xlim, ylim, zlim):
    """
        this function visualizes one person in colorful 3D skeleton, with different colors
        represent different joints, the firsy input is 3 by 23 numpy array
        the second input is the color of skeleton
        input follows COCO convention which is
        {0,  "Nose"}, 
        {1,  "LEye"}, 
        {2,  "REye"}, 
        {3,  "LEar"}, 
        {4,  "REar"},
        
        {5,  "LShoulder"}, 
        {6,  "RShoulder"}, 
        {7,  "LElbow"}, 
        {8,  "RElbow"}, 
        {9,  "LWrist"},
        {10, "RWrist"}, 
        {11, "LHip"}, 
        {12, "RHip"}, 
        {13, "LKnee"}, 
        {14, "Rknee"},
        
        {15, "LAnkle"}, 
        {16, "RAnkle"}, 
        {17, "LBigToe"}, 
        {18, "LSmallToe"}, 
        {19, "LHeel"}, 
        {20, "RBigToe"}, 
        {21, "RSmallToe"},
        {22, "RHeel"}, 
    """ 
    pts3D = pts3D_person
    if dataset_name == 'ETHZ_dataset2' or dataset_name == 'ETHZ_dataset1':
        pts3D = np.array([pts3D[0,:],pts3D[2,:],-pts3D[1,:]])
    
    if dataset_name ==  'EPFL_RLC_MultiCamera':
        pts3D = np.array([pts3D[0,:],pts3D[1,:],-pts3D[2,:]])
        
    if dataset_name ==  'CMU':
        pts3D = np.array([pts3D[0,:],pts3D[2,:],-pts3D[1,:]])   
        
    colormap = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,140/255,0),(0,100/255,0)]   
#     colormap = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]
    kp_size = 15
    colormap_skeleton = [(66/255, 133/255, 244/255), (52/255, 168/255, 83/255), (251/255, 188/255, 5/255), (234/255, 67/255, 53/255)]
    
    bright_orange_cv2 = (255/255,79/255,0)
    co_index = np.random.randint(4)
    co = bright_orange_cv2
    
    #find neck
    neck3D_x = (pts3D[0, 5] + pts3D[0, 6])*0.5
    neck3D_y = (pts3D[1, 5] + pts3D[1, 6])*0.5
    neck3D_z = (pts3D[2, 5] + pts3D[2, 6])*0.5
    if ax != None:
        ax.scatter(neck3D_x, neck3D_y, neck3D_z, s=kp_size, c=colormap[6], marker='o')#Neck
    #find head
    p1 = ((pts3D[0, 1] + pts3D[0, 2])*0.5,(pts3D[1, 1] + pts3D[1, 2])*0.5,(pts3D[2, 1] + pts3D[2, 2])*0.5)
    p2 = ((pts3D[0, 3] + pts3D[0, 4])*0.5,(pts3D[1, 3] + pts3D[1, 4])*0.5,(pts3D[2, 3] + pts3D[2, 4])*0.5)
    p3 = (pts3D[0, 0], pts3D[1, 0],pts3D[2, 0])
    p4 = (0,0,0)
    p_h = SingularityElimination(p1,p2,p3,p4)
    if ax != None:
        ax.scatter(p_h[0], p_h[1], p_h[2], s=kp_size, c=colormap[0], marker='o')#head
        ax.scatter(pts3D[0, 5], pts3D[1, 5], pts3D[2, 5], s=kp_size, c=colormap[7], marker='o') #LShoulder
        ax.scatter(pts3D[0, 6], pts3D[1, 6], pts3D[2, 6], s=kp_size, c=colormap[7], marker='o') #Rshoulder
        ax.scatter(pts3D[0, 7], pts3D[1, 7], pts3D[2, 7], s=kp_size, c=colormap[2], marker='o') #LElbow
        ax.scatter(pts3D[0, 8], pts3D[1, 8], pts3D[2, 8], s=kp_size, c=colormap[2], marker='o') #RElbow
        ax.scatter(pts3D[0, 9], pts3D[1, 9], pts3D[2, 9], s=kp_size, c=colormap[3], marker='o') #LWrist
        ax.scatter(pts3D[0, 10], pts3D[1, 10], pts3D[2, 10], s=kp_size, c=colormap[3], marker='o') #RWrist
        ax.scatter(pts3D[0, 11], pts3D[1, 11], pts3D[2, 11], s=kp_size, c=colormap[4], marker='o') #LHip
        ax.scatter(pts3D[0, 12], pts3D[1, 12], pts3D[2, 12], s=kp_size, c=colormap[4], marker='o') #RHip
        ax.scatter(pts3D[0, 13], pts3D[1, 13], pts3D[2, 13], s=kp_size, c=colormap[5], marker='o') #Lknee
        ax.scatter(pts3D[0, 14], pts3D[1, 14], pts3D[2, 14], s=kp_size, c=colormap[5], marker='o') #Rknee

    #feet
    p_f_L = SingularityElimination(pts3D[:,[15]],pts3D[:,[17]],pts3D[:,[18]],pts3D[:,[19]])
    p_f_R = SingularityElimination(pts3D[:,[16]],pts3D[:,[20]],pts3D[:,[21]],pts3D[:,[22]])
    if ax != None:
        ax.scatter(p_f_L[0], p_f_L[1], p_f_L[2], s=kp_size, c=colormap[1], marker='o')
        ax.scatter(p_f_R[0], p_f_R[1], p_f_R[2], s=kp_size, c=colormap[1], marker='o') 
    
        #connect skeleton 
        ConnectSkeleton(11,13,co,pts3D,ax)  
        ConnectSkeleton(12,14,co,pts3D,ax) 
        ConnectSkeleton(5,6,co,pts3D,ax)
        ConnectSkeleton(5,7,co,pts3D,ax)  
        ConnectSkeleton(7,9,co,pts3D,ax) 
        ConnectSkeleton(6,8,co,pts3D,ax)
        ConnectSkeleton(8,10,co,pts3D,ax)


        x = [p_h[0], neck3D_x]
        y = [p_h[1], neck3D_y]
        z = [p_h[2], neck3D_z]
        ax.plot(x,y,z,color=co )
        x = [pts3D[0, 11], neck3D_x]
        y = [pts3D[1, 11], neck3D_y]
        z = [pts3D[2, 11], neck3D_z]
        ax.plot(x,y,z,color=co )
        x = [pts3D[0, 12], neck3D_x]
        y = [pts3D[1, 12], neck3D_y]
        z = [pts3D[2, 12], neck3D_z]
        ax.plot(x,y,z,color=co )
        x = [pts3D[0, 13], p_f_L[0]]
        y = [pts3D[1, 13], p_f_L[1]]
        z = [pts3D[2, 13], p_f_L[2]]
        ax.plot(x,y,z,color=co )
        x = [pts3D[0, 14], p_f_R[0]]
        y = [pts3D[1, 14], p_f_R[1]]
        z = [pts3D[2, 14], p_f_R[2]]
        ax.plot(x,y,z,color=co )
    
    #formulate 13 pts3D_13
    # {0,  "Head"}, 
    # {1,  "LShoulder"}, 
    # {2,  "RShoulder"},
    # {3,  "LElbow"},  
    # {4,  "RElbow"},
    # {5,  "LWrist"},
    # {6, "RWrist"}, 
    # {7, "LHip"}, 
    # {8, "RHip"}, 
    # {9, "LKnee"}, 
    # {10, "Rknee"},
    # {11, "LFoot"},
    # {12, "RFoot"},
    pts3D_oneperson_13 = np.zeros((4,13)) 
    pts3D_oneperson_13[:,0] = [p_h[0], p_h[1], p_h[2], 1]  #{0,  "Head"}
    pts3D_oneperson_13[:,1] = [pts3D[0, 5], pts3D[1, 5], pts3D[2, 5], 1] #{1,  "LShoulder"}
    pts3D_oneperson_13[:,2] = [pts3D[0, 6], pts3D[1, 6], pts3D[2, 6], 1] # {2,  "RShoulder"}
    pts3D_oneperson_13[:,3] = [pts3D[0, 7], pts3D[1, 7], pts3D[2, 7], 1] # {3,  "LElbow"}
    pts3D_oneperson_13[:,4] = [pts3D[0, 8], pts3D[1, 8], pts3D[2, 8], 1] # {4,  "RElbow"}
    pts3D_oneperson_13[:,5] = [pts3D[0, 9], pts3D[1, 9], pts3D[2, 9], 1] # {5,  "LWrist"}
    pts3D_oneperson_13[:,6] = [pts3D[0, 10], pts3D[1, 10], pts3D[2, 10], 1] # {6,  "RWrist"}
    pts3D_oneperson_13[:,7] = [pts3D[0, 11], pts3D[1, 11], pts3D[2, 11], 1] # {7,  "LHip"}
    pts3D_oneperson_13[:,8] = [pts3D[0, 12], pts3D[1, 12], pts3D[2, 12], 1] # {8,  "RHip"}
    pts3D_oneperson_13[:,9] = [pts3D[0, 13], pts3D[1, 13], pts3D[2, 13], 1] # {9,  "LKnee"}
    pts3D_oneperson_13[:,10] = [pts3D[0, 14], pts3D[1, 14], pts3D[2, 14], 1] # {10,  "RKnee"}
    pts3D_oneperson_13[:,11] = [p_f_L[0], p_f_L[1], p_f_L[2], 1] # {11,  "LFoot"}
    pts3D_oneperson_13[:,12] = [p_f_R[0], p_f_R[1], p_f_R[2], 1] # {12,  "RFoot"}
    return pts3D_oneperson_13

def VisualizeOne3D_13(pts3D_person,co,ax,dataset_name, xlim, ylim, zlim):
    """
        this function visualizes one person in colorful 3D skeleton, with different colors
        represent different joints, the firsy input is 3 by 23 numpy array
        the second input is the color of skeleton
        input follows COCO convention which is
        {0,  "Head"}, 
        {1,  "LShoulder"}, 
        {2,  "RShoulder"},
        {3,  "LElbow"},  
        {4,  "RElbow"},
        {5,  "LWrist"},
        {6, "RWrist"}, 
        {7, "LHip"}, 
        {8, "RHip"}, 
        {9, "LKnee"}, 
        {10, "Rknee"},
        {11, "LFoot"},
        {12, "RFoot"},
    """ 
    pts3D = pts3D_person
    pts3D_for_13vis = pts3D
    
    # mild orange and blue cv2
    mild_orange_cv2 = (255/255,119/255,0)
    mild_blue_cv2 = (30/255, 144/255, 255/255)
    
    #bright orange and blue cv2
    bright_orange_cv2 = (255/255,79/255,0)
    bright_blue_cv2 = (0, 229/255, 238/255)
        
    colormap = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,140/255,0),(0,100/255,0)]   
    #colormap = [mild_blue_cv2,mild_blue_cv2,mild_blue_cv2,mild_blue_cv2,mild_blue_cv2,mild_blue_cv2,mild_blue_cv2,mild_blue_cv2]
    kp_size = 10
    
    colormap_skeleton = [(66/255, 133/255, 244/255), (52/255, 168/255, 83/255), (251/255, 188/255, 5/255), (234/255, 67/255, 53/255)]
    colormap_skeleton = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,140/255,0),(0,100/255,0)] 
    
#     colormap_skeleton = [mild_orange_cv2,mild_blue_cv2,bright_orange_cv2,bright_blue_cv2]
    co_index = np.random.randint(8)
    
    co = bright_orange_cv2
    
    #find neck
    neck3D_x = (pts3D[0, 1] + pts3D[0, 2])*0.5
    neck3D_y = (pts3D[1, 1] + pts3D[1, 2])*0.5
    neck3D_z = (pts3D[2, 1] + pts3D[2, 2])*0.5
    if ax != None:
        ax.scatter(neck3D_x, neck3D_y, neck3D_z, s=kp_size, c=colormap[6], marker='o')#Neck
 
    if ax != None:
        ax.scatter(pts3D[0, 0], pts3D[1, 0], pts3D[2, 0], s=kp_size, c=colormap[0], marker='o')#head
        ax.scatter(pts3D[0, 1], pts3D[1, 1], pts3D[2, 1], s=kp_size, c=colormap[7], marker='o') #LShoulder
        ax.scatter(pts3D[0, 2], pts3D[1, 2], pts3D[2, 2], s=kp_size, c=colormap[7], marker='o') #Rshoulder
        ax.scatter(pts3D[0, 3], pts3D[1, 3], pts3D[2, 3], s=kp_size, c=colormap[2], marker='o') #LElbow
        ax.scatter(pts3D[0, 4], pts3D[1, 4], pts3D[2, 4], s=kp_size, c=colormap[2], marker='o') #RElbow
        ax.scatter(pts3D[0, 5], pts3D[1, 5], pts3D[2, 5], s=kp_size, c=colormap[3], marker='o') #LWrist
        ax.scatter(pts3D[0, 6], pts3D[1, 6], pts3D[2, 6], s=kp_size, c=colormap[3], marker='o') #RWrist
        ax.scatter(pts3D[0, 7], pts3D[1, 7], pts3D[2, 7], s=kp_size, c=colormap[4], marker='o') #LHip
        ax.scatter(pts3D[0, 8], pts3D[1, 8], pts3D[2, 8], s=kp_size, c=colormap[4], marker='o') #RHip
        ax.scatter(pts3D[0, 9], pts3D[1, 9], pts3D[2, 9], s=kp_size, c=colormap[5], marker='o') #Lknee
        ax.scatter(pts3D[0, 10], pts3D[1, 10], pts3D[2, 10], s=kp_size, c=colormap[5], marker='o') #Rknee
        ax.scatter(pts3D[0, 11], pts3D[1, 11], pts3D[2, 11], s=kp_size, c=colormap[1], marker='o') #LFoot
        ax.scatter(pts3D[0, 12], pts3D[1, 12], pts3D[2, 12], s=kp_size, c=colormap[1], marker='o') #RFoot

     
        #connect skeleton 
        ConnectSkeleton(1,2,co,pts3D,ax)  
        ConnectSkeleton(1,3,co,pts3D,ax) 
        ConnectSkeleton(2,4,co,pts3D,ax)
        ConnectSkeleton(3,5,co,pts3D,ax)  
        ConnectSkeleton(4,6,co,pts3D,ax) 
        ConnectSkeleton(7,9,co,pts3D,ax)
        ConnectSkeleton(8,10,co,pts3D,ax)
        ConnectSkeleton(9,11,co,pts3D,ax)
        ConnectSkeleton(10,12,co,pts3D,ax)

        x = [pts3D[0, 0], neck3D_x]
        y = [pts3D[1, 0], neck3D_y]
        z = [pts3D[2, 0], neck3D_z]
        ax.plot(x,y,z,color=co )
        x = [pts3D[0, 7], neck3D_x]
        y = [pts3D[1, 7], neck3D_y]
        z = [pts3D[2, 7], neck3D_z]
        ax.plot(x,y,z,color=co )
        x = [pts3D[0, 8], neck3D_x]
        y = [pts3D[1, 8], neck3D_y]
        z = [pts3D[2, 8], neck3D_z]
        ax.plot(x,y,z,color=co )
    return pts3D_for_13vis


def VisualizeAll3D(pts3D,ax,dataset_name, xlim, ylim, zlim):
    """
        this function visualizes all 3D skeletons from one image
        the form if pts3D is 4 by n numpy array, rows 0,1,2 are x,y,z, row3 is always 1
    """
    bright_orange_cv2 = (255/255,79/255,0)
    bright_blue_cv2 = (0, 229/255, 238/255)
    
    mild_orange_cv2 = (255/255,119/255,0)
    mild_blue_cv2 = (30/255, 144/255, 255/255)
    
    num_people = pts3D.shape[1] / 23
    pts3D_13 = np.zeros((4,int(13 * num_people)))
    print('pts3D_13.shape',pts3D_13.shape)
    for counter in range (0,int(num_people)):
        start = counter * 23
        end = start + 23
        pts3D_person = pts3D[0:3,start:end]
        pts3D_oneperson_13 = VisualizeOne3D(pts3D_person,mild_orange_cv2 , ax,dataset_name, xlim, ylim, zlim)
        start_13 = counter * 13
        end_13 = start_13 + 13
        pts3D_13[0:3,start_13:end_13] = pts3D_oneperson_13[0:3]
    return pts3D_13

def VisualizeAll3D_13(pts3D_for_13vis,ax,dataset_name, xlim, ylim, zlim):
    """
        this function visualizes all 3D skeletons from one image
        the form if pts3D is 4 by n numpy array, rows 0,1,2 are x,y,z, row3 is always 1
    """
    pts3D = pts3D_for_13vis
    bright_orange_cv2 = (255/255,79/255,0)
    bright_blue_cv2 = (0, 229/255, 238/255)
    
    mild_orange_cv2 = (255/255,119/255,0)
    mild_blue_cv2 = (30/255, 144/255, 255/255)
    
    num_people = pts3D.shape[1] / 13
    for counter in range (0,int(num_people)):
        start = counter * 13
        end = start + 13
        pts3D_person = pts3D[0:3,start:end]
        pts3D_oneperson_13 = VisualizeOne3D_13(pts3D_person,mild_orange_cv2 , ax,dataset_name, xlim, ylim, zlim)
    pts3D_13 = np.zeros((4,int(13 * num_people)))
    return pts3D_13