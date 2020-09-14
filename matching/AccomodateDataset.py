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
#import torch
from all_dataset_para import Get_P_from_dataset

def accomoDataset(dataset_name):
    if dataset_name == 'WildTrack':
        path_c4 = '/home/crane/Documents/RestartCVPR2020/AlphaPose_mod/result/C4/alphapose_results_update.json'
        path_c6 = '/home/crane/Documents/RestartCVPR2020/AlphaPose_mod/result/C6/alphapose_results_update.json'
        path3 = ''
        
        path_c1 = '/home/crane/Documents/RestartCVPR2020/RawImage/1740/C1_result/alphapose-results_update.json'
        path_c3 = '/home/crane/Documents/RestartCVPR2020/RawImage/1740/C3_result/alphapose-results_update.json'
        
        image_path_c4 = '/home/crane/Documents/RestartCVPR2020/RawImage/C4'
        image_path_c6 = '/home/crane/Documents/RestartCVPR2020/RawImage/C6'
        image_path3 = ''
        
        image_path_c1 = '/home/crane/Documents/RestartCVPR2020/Dataset/Wildtrack_dataset/Image_subsets/C1'
        image_path_c3 = '/home/crane/Documents/RestartCVPR2020/Dataset/Wildtrack_dataset/Image_subsets/C3'
        
        json_3d_path = '/home/crane/Documents/RestartCVPR2020/FeetMatch_py/WildTrack/mass_saving_result_1101_pfg/data_3D.json'
        
        path1,path2 = path_c4, path_c6
        image_path1,image_path2 = image_path_c4, image_path_c6
        
        P1, P2 = Get_P_from_dataset(dataset_name)
        P3 = np.zeros((3,4))
        H46 = np.array([[-2.13023655e-04, 1.66676273e-03, -9.76608752e-01],
                        [-2.08723174e-04, 3.65874368e-04, -2.15016419e-01],
                        [-2.13360215e-07, 6.64101722e-07, -5.32679554e-04]])
        
        H16 = np.array([[-0.0003,0.0015,-0.9278],
                        [ -1.4725e-4,3.3617e-4,-0.3731],
                        [-1.4483e-7, 6.1926e-7,-6.6033e-4]])
        H13 = np.array([[0.0002,-4.6449e-5,-0.4627],
                        [1.6006e-4,8.2294e-4,-0.8865],
                        [1.4821e-7,1.0299e-6,-0.0011]])
        H23 = np.zeros((3,3)) 
        rand_image = random.choice([x for x in os.listdir(image_path1 )
               if os.path.isfile(os.path.join(image_path1 , x))])
        image = cv2.imread(image_path1  + '/' + rand_image)
        Width = image.shape[1]
        Height = image.shape[0]
        start_index = 0
        end_index = 2000
        H12 = H46
        
    if dataset_name == 'ETHZ_dataset2':
        path1 = '/home/crane/Documents/RestartCVPR2020/Dataset/ETHZ_dataset2/lp-left_rename_result/alphapose-results_update.json'
        path2 = '/home/crane/Documents/RestartCVPR2020/Dataset/ETHZ_dataset2/lp-right_rename_result/alphapose-results_update.json'
        path3 = ''
        image_path1 = '/home/crane/Documents/RestartCVPR2020/Dataset/ETHZ_dataset2/lp-left_rename'
        image_path2 = '/home/crane/Documents/RestartCVPR2020/Dataset/ETHZ_dataset2/lp-right_rename'
        image_path3 = ''
        json_3d_path = '/home/crane/Documents/RestartCVPR2020/FeetMatch_py/ETHZ_dataset2/mass_saving_result_1101_pfg/data_3D.json'
        P1, P2 = Get_P_from_dataset(dataset_name)
        P3 = np.zeros((3,4))
        H12 = np.array([[0.0097,0.0039,-0.9938],
                        [-0.0002,0.0099,0.1101],
                        [-7.1368e-07,-4.3915e-07,0.0102]])
        H13 = np.zeros((3,3))  
        H23 = np.zeros((3,3)) 
        rand_image = random.choice([x for x in os.listdir(image_path1 )
               if os.path.isfile(os.path.join(image_path1 , x))])
        image = cv2.imread(image_path1  + '/' + rand_image)
        Width = image.shape[1]
        Height = image.shape[0]
        start_index = 0
        end_index = 4000
        
    if dataset_name == 'EPFL_RLC_MultiCamera':
        path1 = '/home/crane/Documents/RestartCVPR2020/Dataset/EPFL_RLC_MultiCamera/EPFL-RLC_dataset/frames/cam0_rename_result/alphapose-results_update.json'
        path2 = '/home/crane/Documents/RestartCVPR2020/Dataset/EPFL_RLC_MultiCamera/EPFL-RLC_dataset/frames/cam1_rename_result/alphapose-results_update.json'
        path3 = '/home/crane/Documents/RestartCVPR2020/Dataset/EPFL_RLC_MultiCamera/EPFL-RLC_dataset/frames/cam2_rename_result/alphapose-results_update.json'
        image_path1 = '/home/crane/Documents/RestartCVPR2020/Dataset/EPFL_RLC_MultiCamera/EPFL-RLC_dataset/frames/cam0_rename'
        image_path2 = '/home/crane/Documents/RestartCVPR2020/Dataset/EPFL_RLC_MultiCamera/EPFL-RLC_dataset/frames/cam1_rename'
        image_path3 = '/home/crane/Documents/RestartCVPR2020/Dataset/EPFL_RLC_MultiCamera/EPFL-RLC_dataset/frames/cam2_rename'
        json_3d_path = '/home/crane/Documents/RestartCVPR2020/FeetMatch_py/EPFL_RLC_MultiCamera/mass_saving_result_1101_pfg/data_3D.json'
        P1, P2, P3 = Get_P_from_dataset(dataset_name)
        H12 = np.array([[0.0011,0.0036,-0.8644],
                        [0.0003,0.0034,-0.5028],
                        [1.2447e-6,1.9132e-5,-0.0029]])
        H13 = np.array( [[0.0013,0.0099,-0.8718],
                         [-0.0014,0.0035,0.4897],
                         [ -6.0448e-6,2.4479e-06,0.0043]])
        H23 = np.array([[ 1.5776e-4,-0.0067,0.9498],
                        [9.8795e-4,-0.0032,0.3128],
                        [3.2776e-6,-1.9993e-5,0.0029]])
        rand_image = random.choice([x for x in os.listdir(image_path1)
               if os.path.isfile(os.path.join(image_path1 , x))])
        image = cv2.imread(image_path1 + '/' + rand_image)
        Width = image.shape[1]
        Height = image.shape[0]
        start_index = 8250
        end_index = 9180
        
    if dataset_name == 'ETHZ_dataset1':
        path1 = '/home/crane/Documents/RestartCVPR2020/Dataset/ETHZ_dataset1/seq03-img-left_rename_result/alphapose-results_update.json'
        path2 = '/home/crane/Documents/RestartCVPR2020/Dataset/ETHZ_dataset1/seq03-img-right_rename_result/alphapose-results_update.json'
        path3 = ''
        image_path1 = '/home/crane/Documents/RestartCVPR2020/Dataset/ETHZ_dataset1/seq03-img-left_rename'
        image_path2 = '/home/crane/Documents/RestartCVPR2020/Dataset/ETHZ_dataset1/seq03-img-right_rename'
        image_path3 = ''
        json_3d_path = ''
        P1,P2 = Get_P_from_dataset(dataset_name)
        P3 = np.zeros((3,4))
        H12 = np.array([[0.0029, 0.0031, -0.9494],
                        [0.0002, 0.0031, -0.3140],
                        [1.6123e-6, 5.3003e-6, 1.5440e-4]])
        H13 = np.zeros((3,3))  
        H23 = np.zeros((3,3)) 
        rand_image = random.choice([x for x in os.listdir(image_path1)
               if os.path.isfile(os.path.join(image_path1 , x))])
        image = cv2.imread(image_path1 + '/' + rand_image)
        Width = image.shape[1]
        Height = image.shape[0]
#         start_index = 0
#         end_index = 4995
        start_index = 910
        end_index = 1300
        
    if dataset_name == 'CMU':
        path1 = '/home/crane/Documents/RestartCVPR2020/Dataset/CMU/C2_result/alphapose-results_update.json'
        path2 = '/home/crane/Documents/RestartCVPR2020/Dataset/CMU/C17_result/alphapose-results_update.json'
        path3 = ''
        image_path1 = '/home/crane/Documents/RestartCVPR2020/Dataset/CMU/C2'
        image_path2 = '/home/crane/Documents/RestartCVPR2020/Dataset/CMU/C17'
        image_path3 = ''
        json_3d_path = ''
        #P2 P17
        P1,P2 = Get_P_from_dataset(dataset_name)
        P3 = np.zeros((3,4))
        H12 = np.array([[-6.170463354141715e-04, 0.001988328531467, -0.931736700775615],
                        [-2.326643873601426e-04, 7.088146004161090e-04, -0.363127033651097],
                        [-7.259099627489957e-07, 1.709447875088721e-06, -7.662301831096733e-04]])
        H13 = np.zeros((3,3))  
        H23 = np.zeros((3,3)) 
        rand_image = random.choice([x for x in os.listdir(image_path1)
               if os.path.isfile(os.path.join(image_path1 , x))])
        image = cv2.imread(image_path1 + '/' + rand_image)
        Width = image.shape[1]
        Height = image.shape[0]
        start_index = 4000
        end_index = 4995
    return path1, path2, path3, image_path1, image_path2, image_path3, json_3d_path, P1, P2, P3, H12, H13, H23, Width, Height, start_index, end_index