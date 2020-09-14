import json 
import numpy as np 
from tqdm import tqdm

bone_name_global = ["LR Hip", "LR Shoulder", "Hip Shoulder", "Shoulder Elbow", "Elbow Wrist", "Shoulder ear", "Ear Eye", "Eye Nose", "Hip knee", "Knee Ankle", "Ankle Heel", "Heel BigToe", "Heel SmallToe"]

left_list_global = np.array([11,12, 5,6, 11,5, 5,7, 7,9, 5,3, 3,1, 1,0, 11,13, 13,15, 15,19, 19,17, 19,18]).reshape([-1,2]).tolist()
right_list_global = np.array([11,12, 5,6, 12,6, 6,8, 8,10, 6,4, 4,2, 2,0, 12,14, 14,16, 16,22, 22,20, 22,21]).reshape([-1,2]).tolist()

def collect_param(json_path_3d, left_list=left_list_global, right_list=right_list_global, bone_name_list = bone_name_global, out_json_path=None, ratio = 1.0):
    '''
    Analysis
    Args:
        json_path_3d: 
        left_list
        right_list:
        ratio: only use first `ratio` part of the data, in order to remove large ouliers 
    Return:
        bone_left: a dict contain different bones distance list
        bone_right: 
        median_bone:  
        mean_bone: 
    '''
    assert len(left_list) == len(right_list)
    num_bone = len(left_list)
    bone_left = {i:[] for i in range(num_bone)}
    bone_right = {i:[] for i in range(num_bone)}

    person_num = 0
    with open(json_path_3d, "r") as json_data:
        data_dict = json.load(json_data)["3D"]
    frame_num = len(data_dict)
    for framekey in data_dict.keys():
        for personkey in data_dict[framekey].keys():
            person = np.array(data_dict[framekey][personkey]).reshape([-1,3])
            for i, bone_pair in enumerate(left_list):
                bone_vec = person[bone_pair[1],:] - person[bone_pair[0],:]
                bone_left[i] += [np.linalg.norm(bone_vec)]
            for i, bone_pair in enumerate(right_list):
                bone_vec = person[bone_pair[1],:] - person[bone_pair[0],:]
                bone_right[i] += [np.linalg.norm(bone_vec)]
            person_num += 1
    median_bone = {}
    mean_bone = {}
    l = []
    num_obj = len(bone_left[0])
    for i in range(num_bone):
        left_array = np.array(bone_left[i])
        right_array = np.array(bone_right[i])
        # Take only a subset of bone list
        bone_sorted = np.sort(left_array + right_array)
        bone_sorted = bone_sorted[:int(ratio*num_obj)]

        median_bone[i] = float(np.median(bone_sorted)/2)
        mean_bone[i]= float(np.mean(bone_sorted)/2)
        print(bone_name_list[i])
        print(np.sqrt(np.var(bone_sorted))/median_bone[i])

    if out_json_path is not None:
        data_prior = {}
        with open(out_json_path, "w") as f:
            data_prior["left_list"] = left_list
            data_prior["right_list"] = right_list
            data_prior["median_bone"] = median_bone
            data_prior["bone_name_list"] = bone_name_list
            json.dump(data_prior, f)

    return bone_left, bone_right, median_bone, mean_bone

def loss_kpt_3d(kpt_3d_array, median_bone, left_list=left_list_global, right_list=right_list_global):

    assert kpt_3d_array.shape[1] == 3
    assert len(left_list) == len(right_list)
    assert len(left_list) == len(median_bone.keys())
    num_bone = len(left_list)
    left_error = []
    right_error = []
    for i in range(num_bone):
        bon_vec_left = kpt_3d_array[left_list[i][1],:] - kpt_3d_array[left_list[i][0],:]
        left_error_i = np.sqrt(np.dot(bon_vec_left, bon_vec_left)) - median_bone[str(i)]
        left_error += [ abs(left_error_i)]
    for i in range(num_bone):
        bon_vec_right = kpt_3d_array[right_list[i][1],:] - kpt_3d_array[right_list[i][0],:]
        right_error_i = np.sqrt(np.dot(bon_vec_right, bon_vec_right)) - median_bone[str(i)]
        right_error += [abs(right_error_i)]
    return left_error, right_error

def cal_kpt_3d_error(json_prior, json_fintune, json_prior_eval=None):
    '''
    json_prior:
    json_fintune: json 3d contain key "3D"
    json_prior_eval: json output
    '''
    with open(json_prior, 'r') as f:
        data_prior = json.load(f)
        left_list = data_prior["left_list"]
        right_list = data_prior["right_list"]
        median_bone = data_prior["median_bone"]
    
    with open(json_fintune, 'r') as f:
        data_raw_dict = json.load(f)
        data_dict = data_raw_dict["3D"].copy()
    assert len(left_list) == len(right_list)
    num_bone = len(left_list)

    left_error_dict = {str(i):[] for i in range(num_bone)}
    right_error_dict = {str(i):[] for i in range(num_bone)}

    frame_id_list = [k for k in data_dict.keys()]
    frame_id_list.sort()
    for frame_id in tqdm(frame_id_list):
        person_id_list = [k for k in data_dict[frame_id].keys()]
        person_id_list.sort()
        for person_id in person_id_list:
            kpt_3d_array = np.array(data_dict[frame_id][person_id]).reshape([-1,3])
            left_error, right_error = loss_kpt_3d(kpt_3d_array[:,:3], median_bone)
            for i in range(num_bone):
                left_error_dict[str(i)] += [left_error[i]]
                right_error_dict[str(i)] += [right_error[i]]

    data_raw_dict["left_error"] = left_error_dict
    data_raw_dict["right_error"] = right_error_dict
    if json_prior_eval is not None:
        with open(json_prior_eval, "w") as f:
            json.dump(data_raw_dict, f)
    
    return data_raw_dict
    
    print('OK')
