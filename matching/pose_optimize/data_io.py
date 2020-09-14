import cv2 
import numpy as np 
from os.path import join
import json 

def transloader():
    I4 = np.array([[1725.2772216796875, 0.0, 995.0142211914062], [0.0, 1720.581787109375, 520.4190063476562], [0.0, 0.0, 1.0]])
    I6 = np.array([[1742.977783203125, 0.0, 1001.0738525390625], [0.0, 1746.0140380859375, 362.4325866699219], [0.0, 0.0, 1.0]])
    rvec4 = np.array([1.6647210121154785, 0.9668620228767395, -0.6937940120697021])
    tvec4 = np.array([[42.36193084716797, -45.360652923583984, 1106.8572998046875]])
    rvec6 = np.array([1.6907379627227783, -0.3968360126018524, 0.355197012424469])
    tvec6 = np.array([[-338.5532531738281, 62.87659454345703, 1044.094482421875]])
    
    P4 = GetP(I4, rvec4, tvec4)
    P6 = GetP(I6, rvec6, tvec6)
    # r6 = Rot.from_rotvec(rvec6).as_dcm()
    # cam_intr, g6 = recover_rti(P6)
    # print(np.max(np.abs(g6[:3,:3] - r6)))
    # print(np.max(np.abs(cam_intr - I6)))
    # print(np.max(np.abs(g6[:3,3] - tvec6)))
    return P4,P6

def GetP(I, rvec, tvec):
    """get 3 by 4 projection matrix
    
    Args:
    -I intrinsic matrix
    -rvec tvec rotation and translation vector, opencv format
    
    Return:
    -P 4 by 3 numpy array
    """
    R = cv2.Rodrigues(rvec)[0]
    P = np.dot(I,np.concatenate((R, tvec.T), axis=1))
    return P

def convert_to_3d_format(new_overall_json, json_crane):
    '''
    Convert 3d data for crane's format
    '''
    #base_dir = "data"
    # new_overall_json = join(base_dir, 'data_3D_new_eval.json')
    # json_crane = join(base_dir, 'data_3D_format.json')
    with open(new_overall_json,"r") as f:
        data_3d_dict = json.load(f)
    
    frame_list = [k for k in data_3d_dict["3D"].keys()]
    frame_list.sort()

    data_3d_old = {}

    for f in frame_list:
        person_list = [k for k in data_3d_dict["3D"][f].keys()]
        person_list.sort()
        data_3d_frame = {}
        for p in person_list:
            data_3d_array = np.array(data_3d_dict["3D"][f][p])
            data_3d_array_pad = np.pad(data_3d_array, ((0,0),(0,1)), mode="constant", constant_values=1)
            data_3d_frame[p] = data_3d_array_pad.flatten().tolist()
        data_3d_old[f] = data_3d_frame
    with open(json_crane, "w") as f:
        json.dump(data_3d_old, f)

def load_3d_format_crnae(P1, P2, json_crane, new_overall_json):
    assert P1.shape == (3,4)
    assert P2.shape == (3,4)
    with open(json_crane, "r") as f:
        data_dict = {}
        data_dict_3d = json.load(f)

    data_dict_remove_homo = {}
    for frame_id in data_dict_3d.keys():
        data_dict_remove_homo[frame_id] = {}
        for person_id in data_dict_3d[frame_id].keys():
            data_3d_homo = np.array(data_dict_3d[frame_id][person_id]).reshape([-1,4])
            data_dict_remove_homo[frame_id][person_id] = data_3d_homo[:,:3].flatten().tolist()


    data_dict["3D"] = data_dict_remove_homo
    data_dict["P4"] = P1.astype(float).tolist()
    data_dict["P6"] = P2.astype(float).tolist()
    with open(new_overall_json, "w") as f:
        json.dump(data_dict, f)