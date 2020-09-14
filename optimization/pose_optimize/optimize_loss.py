import numpy as np 
import json
from os.path import join
from tqdm import tqdm
from scipy.optimize import least_squares
from pose_optimize.multiview_geo import reproject_error

DEBUG=False

def reproject_error_loss(p3d, p4, p6, cam_proj_4, cam_proj_6, num_kpt=23):
    '''
    Return:
        kp4_e, kp6_e: error array, both (23,) shape
    '''
    assert p3d.shape == (num_kpt, 3)
    assert p4.shape == (num_kpt, 2)
    assert p6.shape == (num_kpt, 2)

    kp4_recon = np.dot(cam_proj_4[0:3,0:3],p3d.T) + cam_proj_4[0:3,3].reshape([-1,1])
    kp6_recon = np.dot(cam_proj_6[0:3,0:3],p3d.T) + cam_proj_6[0:3,3].reshape([-1,1])

    kp4_recon = kp4_recon[0:2,:]/kp4_recon[2,:]
    kp6_recon = kp6_recon[0:2,:]/kp6_recon[2,:]

    # kp4_e = np.linalg.norm(kp4_recon.T - p4, axis=1)
    # kp6_e = np.linalg.norm(kp6_recon.T - p6, axis=1)
    kp4_e = np.sqrt(np.sum(np.square(kp4_recon.T - p4), axis=1))
    kp6_e = np.sqrt(np.sum(np.square(kp6_recon.T - p6), axis=1))
    
    return kp4_e, kp6_e

def reproject_error_loss_score(p3d, p4, p6, cam_proj_4, cam_proj_6, num_kpt=23):
    '''
    Return:
        kp4_e, kp6_e: error array, both (23,) shape
    '''
    assert p3d.shape == (num_kpt, 3)
    assert p4.shape == (num_kpt, 3)
    assert p6.shape == (num_kpt, 3)
    
    kp4_recon = np.dot(cam_proj_4[0:3,0:3],p3d.T) + cam_proj_4[0:3,3].reshape([-1,1])
    kp6_recon = np.dot(cam_proj_6[0:3,0:3],p3d.T) + cam_proj_6[0:3,3].reshape([-1,1])

    kp4_recon = kp4_recon[0:2,:]/kp4_recon[2,:]
    kp6_recon = kp6_recon[0:2,:]/kp6_recon[2,:]

    # kp4_e = np.linalg.norm(kp4_recon.T - p4, axis=1)
    # kp6_e = np.linalg.norm(kp6_recon.T - p6, axis=1)
    kp4_e = p4[:,2]*np.sqrt(np.sum(np.square(kp4_recon.T - p4[:,:2]), axis=1))
    kp6_e = p6[:,2]*np.sqrt(np.sum(np.square(kp6_recon.T - p6[:,:2]), axis=1))
    
    return kp4_e, kp6_e

def optimze_loss_2d(p3d_faltten, p4, p6, cam_proj_4, cam_proj_6, num_kpt=23, lambda_reproj = 1):
    '''
    Only consider reprojection loss
    '''
    l1 = lambda_reproj

    p3d = p3d_faltten.reshape([-1,3])
    kp4_e, kp6_e = reproject_error_loss(p3d, p4, p6, cam_proj_4, cam_proj_6, num_kpt=23)
    
    return np.concatenate((l1*kp4_e, l1*kp6_e))

def shape_dis_loss(kpt_3d_array, median_bone, left_list, right_list, num_kpt=23):
    '''
    Shape loss given prior shape information
    '''
    assert kpt_3d_array.shape == (num_kpt, 3)
    assert len(left_list) == len(right_list)
    assert len(left_list) == len(median_bone.keys())
    num_bone = len(left_list)
    left_error = []
    right_error = []
    left_error = np.zeros(num_bone)
    right_error = np.zeros(num_bone)
    for i in range(num_bone):
        bon_vec_left = kpt_3d_array[left_list[i][1],:] - kpt_3d_array[left_list[i][0],:]
        left_error_i = np.sqrt(np.dot(bon_vec_left, bon_vec_left)) - median_bone[str(i)]
        left_error[i] = abs(left_error_i)
        
        bon_vec_right = kpt_3d_array[right_list[i][1],:] - kpt_3d_array[right_list[i][0],:]
        right_error_i = np.sqrt(np.dot(bon_vec_right, bon_vec_right)) - median_bone[str(i)]
        right_error[i] = abs(right_error_i)
        
    return left_error, right_error

def optimze_loss(p3d_faltten, p4, p6, cam_proj_4, cam_proj_6, left_list, right_list, median_bone, num_kpt=23, lambda_reproj = 0.1, lambda_shape=5.0):
    '''
    Full Loss with shape prior
    '''
    l1 = lambda_reproj
    l2 = lambda_shape

    p3d = p3d_faltten.reshape([-1,3])
    kp4_e, kp6_e = reproject_error_loss_score(p3d, p4, p6, cam_proj_4, cam_proj_6, num_kpt=23)
    left_error, right_error = shape_dis_loss(p3d, median_bone, left_list, right_list, num_kpt=23)


    return np.concatenate((l1*kp4_e, l1*kp6_e, l2*left_error, l2*right_error))

def optimze_loss_no_score(p3d_faltten, p4, p6, cam_proj_4, cam_proj_6, left_list, right_list, median_bone, num_kpt=23, lambda_reproj = 0.1, lambda_shape=1.0):
    '''
    Full Loss with shape prior
    '''
    l1 = lambda_reproj
    l2 = lambda_shape

    p3d = p3d_faltten.reshape([-1,3])
    kp4_e, kp6_e = reproject_error_loss(p3d, p4, p6, cam_proj_4, cam_proj_6, num_kpt=23)
    left_error, right_error = shape_dis_loss(p3d, median_bone, left_list, right_list, num_kpt=23)


    return np.concatenate((l1*kp4_e, l1*kp6_e, l2*left_error, l2*right_error))

def centerize_keypoint(p1, p2, norm_dst):
    '''
    Centeralize two points
    '''
    assert p1.shape == (3,)
    assert p2.shape == (3,)
    p_center = (p1+p2)/2
    p_vec = (p1-p2)
    p_dis = np.sqrt(np.dot(p_vec, p_vec))
    p1_shift = p_center + 0.5*p_vec/p_dis
    p2_shift = p_center - 0.5*p_vec/p_dis

    return p1_shift, p2_shift

def shape_initialize(left_list, right_list, median_bone, kpt_3d_array, num_kpt=23):
    '''
    Initialize human joints 3D position from shape prior
    '''
    assert kpt_3d_array.shape == (num_kpt,3)
    assert len(left_list) == len(right_list)
    assert len(left_list) == len(median_bone.keys())
    num_bone = len(left_list)
    left_ratio_list, right_ratio_list = [],[]
    vec_left_list, vec_right_list = [], []
    
    ratio_outlier = 1.5
    ratio_draw_back = 1.1
    for i in range(num_bone):
        bon_vec_left = kpt_3d_array[left_list[i][1],:] - kpt_3d_array[left_list[i][0],:] 
        ratio_left = np.sqrt(np.dot(bon_vec_left, bon_vec_left))/ median_bone[str(i)]
        left_ratio_list += [ratio_left]
        vec_left_list += [bon_vec_left]
    for i in range(num_bone):
        bon_vec_right = kpt_3d_array[right_list[i][1],:] - kpt_3d_array[right_list[i][0],:]
        ratio_right = np.sqrt(np.dot(bon_vec_right, bon_vec_right))/median_bone[str(i)]
        right_ratio_list += [ratio_right]
        vec_right_list += [bon_vec_right]
    
    kp_3d_new = np.zeros(kpt_3d_array.shape)
    # Adjust Shoulder to hip
    kp_3d_new[left_list[2][0], :], kp_3d_new[left_list[2][1], :] = centerize_keypoint(kpt_3d_array[left_list[2][0], :], kpt_3d_array[left_list[2][1], :] , median_bone["2"])
    kp_3d_new[right_list[2][0], :], kp_3d_new[right_list[2][1], :] = centerize_keypoint(kpt_3d_array[right_list[2][0], :], kpt_3d_array[right_list[2][1], :] , median_bone["2"])
    # Adjust shoulder and Hip pair
    sh_p = left_list[0]
    hi_p = left_list[1]
    kp_3d_new[sh_p[0]], kp_3d_new[sh_p[1]] = centerize_keypoint(kp_3d_new[sh_p[0]], kp_3d_new[sh_p[1]], median_bone["0"]) # shoulder
    kp_3d_new[hi_p[0]], kp_3d_new[hi_p[1]] = centerize_keypoint(kp_3d_new[hi_p[0]], kp_3d_new[hi_p[1]], median_bone["1"]) # hip

    #  left part
    for i in range(2, num_bone):
        start_indx, end_indx = tuple(left_list[i])
        if left_ratio_list[i] < ratio_outlier:
            kp_3d_new[end_indx, :] = kp_3d_new[start_indx, :] + vec_left_list[i]
        else:
            kp_3d_new[end_indx, :] = kp_3d_new[start_indx, :] + vec_left_list[i]/left_ratio_list[i]*ratio_draw_back

    for i in range(2, num_bone):
        start_indx, end_indx = tuple(right_list[i])
        if right_ratio_list[i] < ratio_outlier:
            kp_3d_new[end_indx, :] = kp_3d_new[start_indx, :] + vec_right_list[i]
        else:
            kp_3d_new[end_indx, :] = kp_3d_new[start_indx, :] + vec_right_list[i]/right_ratio_list[i]*ratio_draw_back


    
    # left_error, right_error = loss_kpt_3d(kp_3d_new, median_bone, left_list, right_list)  
    # print(left_error) 
    # print(right_error)
    # print("OK")
    return kp_3d_new

def fintune_human_keypoint_2d(P4, P6, path4, path6, path3D, path_finetune=None):
    
    with open(path3D,"r") as f:
        data_3d = json.load(f)
    with open(path4, "r") as f:
        data_dict4 = json.load(f)
    with open(path6, "r") as f:
        data_dict6 = json.load(f)   
    
    # frame_id = next(iter(data_3d["3D"].keys()))
    # person_id = next(iter(data_3d["3D"][frame_id].keys()))
    # # frame_id = "000005"
    # # person_id = "000"
    cam_proj_4 = np.array(data_3d["P4"])
    cam_proj_6 = np.array(data_3d["P6"])

    data_3d_dict = {}
    data_3d_dict["P4"] = data_3d["P4"]
    data_3d_dict["P6"] = data_3d["P6"]
    data_3d_dict["3D"] = {}
    data_3d_dict["kp4_e"] = {}
    data_3d_dict["kp6_e"] = {}

    frame_list = [k for k in data_dict4.keys()]
    frame_list.sort()
    for i, frame_id in enumerate(tqdm(frame_list)):
        frame_3d_dict = {}
        kp4_dict = {}
        kp6_dict = {}
        
        person_list = [k for k in data_dict4[frame_id].keys()]
        person_list.sort()

        for person_id in person_list:
            p3d_flatten = np.array(data_3d["3D"][frame_id][person_id]).ravel()
            p4_homo = np.array(data_dict4[frame_id][person_id]).reshape([-1,3])
            p6_homo = np.array(data_dict6[frame_id][person_id]).reshape([-1,3])

            p4 = p4_homo[:,:2]
            p6 = p6_homo[:,:2]
            
            if DEBUG: 
                loss_init = optimze_loss_2d(p3d_flatten, p4, p6, cam_proj_4, cam_proj_6)
                print("Initial error", str(np.sqrt(np.sum(np.square(loss_init)))) )
            
            res = least_squares(optimze_loss_2d, p3d_flatten, verbose=0, x_scale='jac', ftol=1e-4, method='trf',args=(p4, p6, cam_proj_4, cam_proj_6))
            
            if DEBUG: 
                loss_final = res.fun
                print("Final error", str(np.sqrt(np.sum(np.square(loss_final)))) )
                loss_final = optimze_loss_2d(res.x, p4, p6, cam_proj_4, cam_proj_6)
                print("Final error", str(np.sqrt(np.sum(np.square(loss_final)))) )

            p3d_tune = res.x.reshape([-1,3])
            
            kp4_recon, kp6_recon, kp4_e, kp6_e = reproject_error(p3d_tune, p4, p6, cam_proj_4, cam_proj_6)

            frame_3d_dict[person_id] = p3d_tune.tolist()
            kp4_dict[person_id] = kp4_e.tolist()
            kp6_dict[person_id] = kp6_e.tolist()
        
        data_3d_dict["3D"][frame_id] = frame_3d_dict
        data_3d_dict["kp4_e"][frame_id] = kp4_dict
        data_3d_dict["kp6_e"][frame_id] = kp6_dict
    
    if path_finetune is not None:
        with open(path_finetune, "w") as f:
            json.dump(data_3d_dict, f)

    return data_3d_dict

def finetune_human_3d(path_finetune_input, path4, path6, shape_prior_path, shape_prior_finetune_output, frame_list=None):
    '''
    path_finetune_input:
    path4: data_C4.json
    path6: data_C6.json
    shape_prior_path:
    shape_prior_finetune_output:
    '''
    with open(path_finetune_input,"r") as f:
        data_3d = json.load(f)
    with open(path4, "r") as f:
        data_dict4 = json.load(f)
    with open(path6, "r") as f:
        data_dict6 = json.load(f)   

    with open(shape_prior_path, 'r') as f:
        data_prior = json.load(f)
        left_list = data_prior["left_list"]
        right_list = data_prior["right_list"]
        median_bone = data_prior["median_bone"]
    
    cam_proj_4 = np.array(data_3d["P4"])
    cam_proj_6 = np.array(data_3d["P6"])

    data_3d_dict = {}
    data_3d_dict["P4"] = data_3d["P4"]
    data_3d_dict["P6"] = data_3d["P6"]
    data_3d_dict["3D"] = {}
    data_3d_dict["kp4_e"] = {}
    data_3d_dict["kp6_e"] = {}

    if frame_list:
        for f in frame_list:
            if f not in data_dict4.keys():
                print("KEY ERROR!")
                assert 0
    else:
        frame_list = [k for k in data_dict4.keys()]
        frame_list.sort()
    
    for i, frame_id in enumerate(tqdm(frame_list)):
        frame_3d_dict = {}
        kp4_dict = {}
        kp6_dict = {}
        
        person_list = [k for k in data_dict4[frame_id].keys()]
        person_list.sort()

        for person_id in person_list:

            p3d = np.array(data_3d["3D"][frame_id][person_id]).reshape([-1,3])
            p3d_init = shape_initialize(left_list, right_list, median_bone, p3d)

            p4_homo = np.array(data_dict4[frame_id][person_id]).reshape([-1,3])
            p6_homo = np.array(data_dict6[frame_id][person_id]).reshape([-1,3])

            p4 = p4_homo
            p6 = p6_homo

            p3d_flatten = p3d_init.flatten()
            # loss_init = optimze_loss(p3d_flatten, p4, p6, cam_proj_4, cam_proj_6, left_list, right_list, median_bone)
            #print(np.linalg.norm(loss_init))
            res = least_squares(optimze_loss, p3d_flatten, verbose=0, x_scale='jac', ftol=1e-2, method='trf',args=(p4, p6, cam_proj_4, cam_proj_6, left_list, right_list, median_bone))

            p3d_tune = res.x.reshape([-1,3])
            # loss_res = optimze_loss(res.x, p4, p6, cam_proj_4, cam_proj_6, left_list, right_list, median_bone)
            # print(np.linalg.norm(loss_res))
            
            kp4_recon, kp6_recon, kp4_e, kp6_e = reproject_error(p3d_tune, p4[:,:2], p6[:,:2], cam_proj_4, cam_proj_6)

            frame_3d_dict[person_id] = p3d_tune.tolist()
            kp4_dict[person_id] = kp4_e.tolist()
            kp6_dict[person_id] = kp6_e.tolist()
        
        data_3d_dict["3D"][frame_id] = frame_3d_dict
        data_3d_dict["kp4_e"][frame_id] = kp4_dict
        data_3d_dict["kp6_e"][frame_id] = kp6_dict
        
    with open(shape_prior_finetune_output, "w") as f:
        json.dump(data_3d_dict, f)

def finetune_human_3d_no_score(path_finetune_input, path4, path6, shape_prior_path, shape_prior_finetune_output, frame_list=None):
    '''
    path_finetune_input:
    path4: data_C4.json
    path6: data_C6.json
    shape_prior_path:
    shape_prior_finetune_output:
    '''
    with open(path_finetune_input,"r") as f:
        data_3d = json.load(f)
    with open(path4, "r") as f:
        data_dict4 = json.load(f)
    with open(path6, "r") as f:
        data_dict6 = json.load(f)   

    with open(shape_prior_path, 'r') as f:
        data_prior = json.load(f)
        left_list = data_prior["left_list"]
        right_list = data_prior["right_list"]
        median_bone = data_prior["median_bone"]
    
    cam_proj_4 = np.array(data_3d["P4"])
    cam_proj_6 = np.array(data_3d["P6"])

    data_3d_dict = {}
    data_3d_dict["P4"] = data_3d["P4"]
    data_3d_dict["P6"] = data_3d["P6"]
    data_3d_dict["3D"] = {}
    data_3d_dict["kp4_e"] = {}
    data_3d_dict["kp6_e"] = {}

    if frame_list:
        for f in frame_list:
            if f not in data_dict4.keys():
                print("KEY ERROR!")
                assert 0
    else:
        frame_list = [k for k in data_dict4.keys()]
        frame_list.sort()
    
    for i, frame_id in enumerate(tqdm(frame_list)):
        if i > 300:
            import sys 
            sys.exit()
        frame_3d_dict = {}
        kp4_dict = {}
        kp6_dict = {}
        
        person_list = [k for k in data_dict4[frame_id].keys()]
        person_list.sort()

        for person_id in person_list:
            try:
                p3d = np.array(data_3d["3D"][frame_id][person_id]).reshape([-1,3])
                p3d_init = shape_initialize(left_list, right_list, median_bone, p3d)

                p4_homo = np.array(data_dict4[frame_id][person_id]).reshape([-1,3])
                p6_homo = np.array(data_dict6[frame_id][person_id]).reshape([-1,3])

                p4 = p4_homo[:,:2]
                p6 = p6_homo[:,:2]

                p3d_flatten = p3d_init.flatten()
                # loss_init = optimze_loss(p3d_flatten, p4, p6, cam_proj_4, cam_proj_6, left_list, right_list, median_bone)
                #print(np.linalg.norm(loss_init))
                res = least_squares(optimze_loss_no_score, p3d_flatten, verbose=2, x_scale='jac', ftol=1e-2, method='trf',args=(p4, p6, cam_proj_4, cam_proj_6, left_list, right_list, median_bone))

                p3d_tune = res.x.reshape([-1,3])
                # loss_res = optimze_loss(res.x, p4, p6, cam_proj_4, cam_proj_6, left_list, right_list, median_bone)
                # print(np.linalg.norm(loss_res))
                
                kp4_recon, kp6_recon, kp4_e, kp6_e = reproject_error(p3d_tune, p4[:,:2], p6[:,:2], cam_proj_4, cam_proj_6)

                frame_3d_dict[person_id] = p3d_tune.tolist()
                kp4_dict[person_id] = kp4_e.tolist()
                kp6_dict[person_id] = kp6_e.tolist()
            except:
                print("Error")
        data_3d_dict["3D"][frame_id] = frame_3d_dict
        data_3d_dict["kp4_e"][frame_id] = kp4_dict
        data_3d_dict["kp6_e"][frame_id] = kp6_dict
        
    with open(shape_prior_finetune_output, "w") as f:
        json.dump(data_3d_dict, f)
