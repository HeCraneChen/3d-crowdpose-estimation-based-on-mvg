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
from all_dataset_para import Get_P_from_dataset

def GetP(I, rvec, tvec):
    """
        this function generate projection matrix of camera
    """
    R = cv2.Rodrigues(rvec)[0]
    P = np.dot(I,np.concatenate((R, tvec.T), axis=1))
    return P

def accomoAllWildtrack(view1,view2):
    dataset_name = 'WildTrack'
    if dataset_name == 'WildTrack':
        path_c4 = '../Dataset/Wildtrack_dataset/Image_subsets/C4_result/alphapose-results_update.json'
        path_c6 = '../Dataset/Wildtrack_dataset/Image_subsets/C6_result/alphapose-results_update.json'
       
        
        path3 = ''
        
        path_c1 = '../Dataset/Wildtrack_dataset/Image_subsets/C1_result/alphapose-results_update.json'
        path_c3 = '../Dataset/Wildtrack_dataset/Image_subsets/C3_result/alphapose-results_update.json'
        
        
        image_path_c4 = '../Dataset/Wildtrack_dataset/Image_subsets/C4'
        image_path_c6 = '../Dataset/Wildtrack_dataset/Image_subsets/C6'
        image_path3 = ''
        
        image_path_c1 = '../Dataset/Wildtrack_dataset/Image_subsets/C1'
        image_path_c3 = '../Dataset/Wildtrack_dataset/Image_subsets/C3'
        
        json_3d_path = '../Dataset/WildTrack/mass_saving_result_1101_pfg/data_3D.json'
        
        
        I1 = np.array([[1743.4478759765625, 0.0, 934.5202026367188], [0.0, 1735.1566162109375, 444.3987731933594], [0.0, 0.0, 1.0]])
        I2 = np.array([[1707.266845703125, 0.0, 978.1306762695312], [0.0, 1719.0408935546875, 417.01922607421875], [0.0, 0.0, 1.0]])
        I3 = np.array([[1738.7144775390625, 0.0, 906.56689453125], [0.0, 1752.8876953125, 462.0346374511719], [0.0, 0.0, 1.0]])
        I4 = np.array([[1725.2772216796875, 0.0, 995.0142211914062], [0.0, 1720.581787109375, 520.4190063476562], [0.0, 0.0, 1.0]])
        I5 = np.array([[1708.6573486328125, 0.0, 936.0921630859375], [0.0, 1737.1904296875, 465.18243408203125], [0.0, 0.0, 1.0]])
        I6 = np.array([[1742.977783203125, 0.0, 1001.0738525390625], [0.0, 1746.0140380859375, 362.4325866699219], [0.0, 0.0, 1.0]])
        I7 = np.array([[1732.4674072265625, 0.0, 931.2559204101562], [0.0, 1757.58203125, 459.43389892578125], [0.0, 0.0, 1.0]])

        #external matrix
        rvec1 = np.array([ 1.759099006652832, 0.46710100769996643, -0.331699013710022])
        tvec1 = np.array([[-525.8941650390625, 45.40763473510742, 986.7235107421875]])

        rvec2 = np.array([0.6167870163917542, -2.14595890045166, 1.6577140092849731])
        tvec2 = np.array([[1195.231201171875, -336.5144958496094, 2040.53955078125]])

        rvec3 = np.array([0.5511789917945862, 2.229501962661743, -1.7721869945526123])
        tvec3 = np.array([[55.07157897949219, -213.2444610595703, 1992.845703125]])

        rvec4 = np.array([1.6647210121154785, 0.9668620228767395, -0.6937940120697021])
        tvec4 = np.array([[42.36193084716797, -45.360652923583984, 1106.8572998046875]])

        rvec5 = np.array([1.2132920026779175, -1.4771349430084229, 1.2775369882583618])
        tvec5 = np.array([[836.6625366210938, 85.86837005615234, 600.2880859375]])

        rvec6 = np.array([1.6907379627227783, -0.3968360126018524, 0.355197012424469])
        tvec6 = np.array([[-338.5532531738281, 62.87659454345703, 1044.094482421875]])

        rvec7 = np.array([1.6439390182495117, 1.126188039779663, -0.7273139953613281])
        tvec7 = np.array([[-648.9456787109375, -57.225215911865234, 1052.767578125]])
          

        H46 = np.array([[-2.13023655e-04, 1.66676273e-03, -9.76608752e-01],
                        [-2.08723174e-04, 3.65874368e-04, -2.15016419e-01],
                        [-2.13360215e-07, 6.64101722e-07, -5.32679554e-04]])
        
        H16 = np.array([[-0.0003,0.0015,-0.9278],
                        [ -1.4725e-4,3.3617e-4,-0.3731],
                        [-1.4483e-7, 6.1926e-7,-6.6033e-4]])

        
        H13 = np.array([[0.000247762137227159,4.14702379470164e-05,-0.503796125773348],
         [0.000145188859368693,0.000819171357061377,-0.863821468145268],
         [1.40572893608001e-07,1.04167811648611e-06,-0.00108609329244425]])
        
        
        H34 = np.array([[-0.000115142207552733,0.00127483305683559,-0.990703845524425],
        [-8.33679184810137e-05,0.000214191001010202,-0.136029890767930],
        [-8.32206264462426e-08,3.37369831710750e-07,-0.000260759688223595]])
        
        H23 = np.zeros((3,3)) 
        image_path1 = image_path_c4
        
        
        rand_image = random.choice([x for x in os.listdir(image_path1 )
               if os.path.isfile(os.path.join(image_path1 , x))])
        image = cv2.imread(image_path1  + '/' + rand_image)
        Width = image.shape[1]
        Height = image.shape[0]
        start_index = 0
        end_index = 2000
        P1 = GetP(I1, rvec1, tvec1)
        P2 = GetP(I2, rvec2, tvec2)
        P3 = GetP(I3, rvec3, tvec3)
        P4 = GetP(I4, rvec4, tvec4)
        P5 = GetP(I5, rvec5, tvec5)
        P6 = GetP(I6, rvec6, tvec6)
        
        if view1 == 1 and view2 == 6:
            H = H16 
            path1,path2 = path_c1, path_c6
            image_path1,image_path2 = image_path_c1, image_path_c6
            P1,P2 = P1,P6
        if view1 == 1 and view2 == 3:
            H = H13
            path1,path2 = path_c1, path_c3
            image_path1,image_path2 = image_path_c1, image_path_c3
            P1,P2 = P1,P3
        if view1 == 4 and view2 == 6:
            H = H46
            path1,path2 = path_c4,path_c6
            image_path1,image_path2 = image_path_c4, image_path_c6
            P1,P2 = P4,P6
        if view1 == 3 and view2 == 4:
            H = H34
            path1,path2 = path_c3,path_c4
            image_path1,image_path2 = image_path_c3, image_path_c4
            P1,P2 = P3,P4
        
            
    return path1, path2, image_path1, image_path2, json_3d_path, P1, P2, H, Width, Height