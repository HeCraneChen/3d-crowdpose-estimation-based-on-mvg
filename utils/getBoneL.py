import numpy as np
import json
import cv2
import math
import statistics 

def getBones(json_3d_path):
    """get the average bone length of 3D person
    each person consists of 92 numbers, namely 23 homogeneous 3D kp
    kp order (COCO style):
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
    
    -Returns: bone lengths in the form of lists
      [nose_eye, eye_ear, LRshoulder, upperArm, lowerArm, LRhip, hip_knee, knee_ankle, ankle_heel, heel_bigtoe, heel_smalltoe, ear_ear, wrist_wrist, ankle_ankle, heel_heel, toe_toe]
    """
    person_num = 0
    all_nose_eye, all_eye_ear, all_LRshoulder, all_upperArm, all_lowerArm = 0,0,0,0,0    
    all_LRhip, all_hip_knee, all_knee_ankle, all_ankle_heel, all_heel_bigtoe, all_heel_smalltoe = 0,0,0,0,0,0
    all_ear_ear, all_wrist_wrist, all_ankle_ankle, all_heel_heel, all_toe_toe = 0,0,0,0,0
    all_nose_eye_list, all_eye_ear_list, all_LRshoulder_list, all_upperArm_list, all_lowerArm_list = [],[],[],[],[]
    all_LRhip_list, all_hip_knee_list, all_knee_ankle_list, all_ankle_heel_list, all_heel_bigtoe_list, all_heel_smalltoe_list = [],[],[],[],[],[]
    all_ear_ear_list, all_wrist_wrist_list, all_ankle_ankle_list, all_heel_heel_list, all_toe_toe_list = [],[],[],[],[]
    with open(json_3d_path) as json_data:
        data_dict = json.load(json_data)
    frame_num = len(data_dict)
    for framekey in data_dict:
        for personkey in data_dict[framekey]:
            person = data_dict[framekey][personkey]
            nose = person[0:3]
            Leye = person[4:7]
            Reye = person[8:11]
            Lear = person[12:15]
            Rear = person[16:19]
            Lshoulder = person[20:23]
            Rshoulder = person[24:27]
            Lelbow = person[28:31]
            Relbow = person[32:35]
            Lwrist = person[36:39]
            Rwrist = person[40:43]
            Lhip = person[44:47]
            Rhip = person[48:51]
            Lknee = person[52:55]
            Rknee = person[56:59]
            LAnkle = person[60:63]
            RAnkle = person[64:67]
            Lbigtoe = person[68:71]
            Lsmalltoe = person[72:75]
            Lheel = person[76:79]
            Rbigtoe = person[80:83]
            Rsmalltoe = person[84:87]
            Rheel = person[88:91]
            # bones
            nose_eye = 0.5 * (np.linalg.norm((np.asarray(nose)-np.asarray(Leye))) + np.linalg.norm((np.asarray(nose)-np.asarray(Reye))))
            eye_ear = 0.5 * (np.linalg.norm((np.asarray(Lear)-np.asarray(Leye))) + np.linalg.norm((np.asarray(Rear)-np.asarray(Reye))))              
            LRshoulder = np.linalg.norm((np.asarray(Lshoulder)-np.asarray(Rshoulder)))
            upperArm =  0.5 * (np.linalg.norm((np.asarray(Lshoulder)-np.asarray(Lelbow))) + np.linalg.norm((np.asarray(Rshoulder)-np.asarray(Relbow))))   
            lowerArm = 0.5 * (np.linalg.norm((np.asarray(Lelbow)-np.asarray(Lwrist))) + np.linalg.norm((np.asarray(Relbow)-np.asarray(Rwrist))))  
            LRhip =  np.linalg.norm((np.asarray(Lhip)-np.asarray(Rhip)))         
            hip_knee = 0.5 * (np.linalg.norm((np.asarray(Lhip)-np.asarray(Lknee))) + np.linalg.norm((np.asarray(Rhip)-np.asarray(Rknee))))  
            knee_ankle = 0.5 * (np.linalg.norm((np.asarray(Lknee)-np.asarray(LAnkle))) + np.linalg.norm((np.asarray(Rknee)-np.asarray(RAnkle))))  
            ankle_heel = 0.5 * (np.linalg.norm((np.asarray(LAnkle)-np.asarray(Lheel))) + np.linalg.norm((np.asarray(RAnkle)-np.asarray(Rheel))))
            heel_bigtoe = 0.5 * (np.linalg.norm((np.asarray(Lheel)-np.asarray(Lbigtoe))) + np.linalg.norm((np.asarray(Rheel)-np.asarray(Rbigtoe))))    
            heel_smalltoe = 0.5 * (np.linalg.norm((np.asarray(Lheel)-np.asarray(Lsmalltoe))) + np.linalg.norm((np.asarray(Rheel)-np.asarray(Rsmalltoe)))) 
            
            #other important constraints
            ear_ear = np.linalg.norm((np.asarray(Lear)-np.asarray(Rear)))
            wrist_wrist =  np.linalg.norm((np.asarray(Lwrist)-np.asarray(Rwrist)))
            ankle_ankle =  np.linalg.norm((np.asarray(LAnkle)-np.asarray(RAnkle)))    
            heel_heel =  np.linalg.norm((np.asarray(Lheel)-np.asarray(Rheel)))    
            toe_toe =  0.5 * (np.linalg.norm((np.asarray(Lbigtoe)-np.asarray(Rbigtoe))) + np.linalg.norm((np.asarray(Lsmalltoe)-np.asarray(Rsmalltoe))))                              
                                          
            
            person_num += 1
            all_nose_eye += nose_eye
            all_eye_ear += eye_ear
            all_LRshoulder += LRshoulder
            all_upperArm += upperArm
            all_lowerArm += lowerArm
            all_LRhip += LRhip
            all_hip_knee += hip_knee
            all_knee_ankle += knee_ankle
            all_ankle_heel += ankle_heel
            all_heel_bigtoe += heel_bigtoe
            all_heel_smalltoe += heel_smalltoe
            
            all_nose_eye_list.append(nose_eye)
            all_eye_ear_list.append(eye_ear)
            all_LRshoulder_list.append(LRshoulder)
            all_upperArm_list.append(upperArm)
            all_lowerArm_list.append(lowerArm)
            all_LRhip_list.append(LRhip)
            all_hip_knee_list.append(hip_knee)
            all_knee_ankle_list.append(knee_ankle)
            all_ankle_heel_list.append(ankle_heel)
            all_heel_bigtoe_list.append(heel_bigtoe)
            all_heel_smalltoe_list.append(heel_smalltoe)
                                        
            all_ear_ear += ear_ear
            all_wrist_wrist += wrist_wrist
            all_ankle_ankle += ankle_ankle
            all_heel_heel += heel_heel
            all_toe_toe += toe_toe                           
                                        
            all_ear_ear_list.append(ear_ear)
            all_wrist_wrist_list.append(wrist_wrist)
            all_ankle_ankle_list.append(ankle_ankle)  
            all_heel_heel_list.append(heel_heel) 
            all_toe_toe_list.append(toe_toe)
            
    nose_eye = all_nose_eye / person_num
    eye_ear =  all_eye_ear / person_num
    LRshoulder = all_LRshoulder / person_num
    upperArm =  all_upperArm / person_num
    lowerArm =  all_lowerArm / person_num
    LRhip =  all_LRhip / person_num
    hip_knee =  all_hip_knee / person_num
    knee_ankle = all_knee_ankle / person_num
    ankle_heel = all_ankle_heel / person_num
    heel_bigtoe = all_heel_bigtoe / person_num
    heel_smalltoe =  all_heel_smalltoe / person_num 
    
    ear_ear = all_ear_ear / person_num
    wrist_wrist =  all_wrist_wrist / person_num
    ankle_ankle = all_ankle_ankle  / person_num
    heel_heel = all_heel_heel / person_num
    toe_toe = all_toe_toe / person_num                                    
                                        
    mean_bones = [nose_eye, eye_ear, LRshoulder, upperArm, lowerArm, LRhip, hip_knee, knee_ankle, ankle_heel, heel_bigtoe, heel_smalltoe, ear_ear, wrist_wrist, ankle_ankle, heel_heel, toe_toe]
    
    all_nose_eye_list = all_nose_eye_list[0:round(len(all_nose_eye_list)*0.7)]
    all_eye_ear_list = all_eye_ear_list[0:round(len(all_eye_ear_list)*0.7)]
    all_LRshoulder_list = all_LRshoulder_list[0:round(len(all_LRshoulder_list)*0.7)]
    all_upperArm_list =  all_upperArm_list[0:round(len(all_upperArm_list)*0.7)]
    all_lowerArm_list = all_lowerArm_list[0:round(len(all_lowerArm_list)*0.7)]
    all_LRhip_list = all_LRhip_list[0:round(len(all_LRhip_list)*0.7)]
    all_hip_knee_list = all_hip_knee_list[0:round(len(all_hip_knee_list)*0.7)]
    all_knee_ankle_list = all_knee_ankle_list[0:round(len(all_knee_ankle_list)*0.7)]
    all_ankle_heel_list = all_ankle_heel_list[0:round(len(all_ankle_heel_list)*0.7)]
    all_heel_bigtoe_list = all_heel_bigtoe_list[0:round(len(all_heel_bigtoe_list)*0.7)]
    all_heel_smalltoe_list = all_heel_smalltoe_list[0:round(len(all_heel_smalltoe_list)*0.7)]
    all_ear_ear_list = all_ear_ear_list[0:round(len(all_ear_ear_list)*0.7)]
    all_wrist_wrist_list = all_wrist_wrist_list[0:round(len(all_wrist_wrist_list)*0.7)]
    all_ankle_ankle_list = all_ankle_ankle_list[0:round(len(all_ankle_ankle_list)*0.7)]
    all_heel_heel_list =all_heel_heel_list[0:round(len(all_heel_heel_list)*0.7)]
    all_toe_toe_list = all_toe_toe_list[0:round(len(all_toe_toe_list)*0.7)]
    
    median_bones = [statistics.median(all_nose_eye_list),
                    statistics.median(all_eye_ear_list),
                    statistics.median(all_LRshoulder_list),
                    statistics.median(all_upperArm_list),
                    statistics.median(all_lowerArm_list),
                    statistics.median(all_LRhip_list),
                    statistics.median(all_hip_knee_list),                   
                    statistics.median(all_knee_ankle_list),
                    statistics. median(all_ankle_heel_list),
                    statistics.median(all_heel_bigtoe_list),
                    statistics.median(all_heel_smalltoe_list),                     
                    statistics.median(all_ear_ear_list),
                    statistics.median(all_wrist_wrist_list),
                    statistics.median(all_ankle_ankle_list),
                    statistics.median(all_heel_heel_list),
                    statistics.median(all_toe_toe_list)]
    print('len(median_bones)',len(median_bones))
    return mean_bones, median_bones
