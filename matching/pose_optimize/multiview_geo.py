import numpy as np
import scipy
from scipy.spatial.transform import Rotation as Rot
import cv2 
import json
from tqdm import tqdm
DEBUG = False
def get_distance(p4, p6, cam_proj_4, cam_proj_6):
    '''
    calculate minimum distance with epipolar constraint
    Args:
        p4: key point on camera 4 -- shape: (3,) 
        p6: key point on camera 4 -- shape: (3,) 
        cam_proj_4: projection matrix of camera 4
        cam_proj_6: projection matrix of camera 6
    Return:
        dis4, dis6
    '''
    norm_line = lambda line : np.sqrt(np.sum(np.square(line)[0,0:2]))
    assert cam_proj_4.shape == (3,4)
    assert cam_proj_6.shape == (3,4)
    assert p4.shape == (3,)
    assert p6.shape == (3,)
    
    p4 = p4.reshape([-1,1])
    p6 = p6.reshape([-1,1])

    F64 = fund_mat(cam_proj_4, cam_proj_6)
    epo_4 = np.dot(p6.T, F64)
    epo_4 = epo_4/norm_line(epo_4)
    epo_6 = np.dot(F64, p4).T
    epo_6 = epo_6/norm_line(epo_6)

    dis4, p4_correct_i = orth_point_line(epo_4, p4)
    dis6, p6_correct_i = orth_point_line(epo_6, p6)
    
    return dis4, dis6

def fund_mat(P1, P2):
    '''
    Fundamental matrix that satisfies x2'*F * x1 = 0
    F = (K2.inv)' * E * (K1.inv), E = T_21^ * R_21
    Args;
        P1: projection matrix of camera 1 (3,4)
        P2: projection matrix of camera 2 (3,4)
    Return :
        F_mat: fundamental matrix (3,3)
    '''
    assert P1.shape == (3,4)
    assert P2.shape == (3,4)

    cam_intr_1, g_10 = recover_rti(P1)
    cam_intr_2, g_20 = recover_rti(P2)
    g_21 = np.dot(g_20, np.linalg.inv(g_10) )

    t_hat = np.zeros([3,3])
    t_hat[0,1] = - g_21[2,3] # a3
    t_hat[0,2] =   g_21[1,3] # a2
    t_hat[1,2] = - g_21[0,3] # a1
    t_hat += - t_hat.T
    rot_21 = g_21[0:3,0:3]

    E_mat = np.dot(t_hat, rot_21)
    K_inv_2 = np.linalg.inv(cam_intr_2)
    K_inv_1 = np.linalg.inv(cam_intr_1)
    F_mat_21 = np.dot(K_inv_2.T, E_mat)
    F_mat_21 = np.dot(F_mat_21, K_inv_1)

    return F_mat_21

def recover_rti(extrinsic_matrix):
    '''
    Recover camera rotation and translation from extrinsic matrix
    Args;
        extrinsic_matrix: (3,4) matrix 
    Return:
        cam_intr: (3,3) upper trainguler matrix
        rot_m: (3,3) orthgonal matrix
        trans_vec: (3,) translation vector
    '''
    assert extrinsic_matrix.shape == (3,4)
    cam_intr, rot_m = scipy.linalg.rq(extrinsic_matrix[:, 0:3])
    trans_vec = np.dot(np.linalg.inv(cam_intr), extrinsic_matrix[:, 3])
    g = np.eye(4)
    g[0:3, 0:3] = rot_m
    g[0:3, 3] = trans_vec
    return cam_intr, g

def orth_point_line(line, point):
    '''
    Calculate minimum distance between line and point, return distance and point on the line
    Args:
        line: (1,3) unit vector for (a,b)
        point: (3,1) homogenous representation
    Return:
        dis: distance in pixel 
        x1: point on the line

    '''
    assert line.shape == (1,3)
    assert point.shape == (3,1)
    norm_line = lambda line : np.sqrt(np.sum(np.square(line)[0,0:2]))
    assert abs(norm_line(line)-1) < 1e-9
    dis = np.dot(line, point)[0,0]
    norm_vec = np.zeros([3,1])
    norm_vec[0:2,0] = line[0,:2]
    x1 = point - dis*norm_vec
    return dis, x1

def reconstruct_point_mini(cam_proj_4, cam_proj_6, p4, p6, fund_64 = None):
    '''
    Args:
        p4: single person keypoint in camera 4
        p6: single person keypoint in camrea 6
        cam_proj_4: camera projection matrix for camera 4
        cam_proj_6: camera projection matrix for camera 6
    Return:
        p3d: 3d point in the world, numpy array, shape is (3, num_kpt)
    '''
    norm_line = lambda line : np.sqrt(np.sum(np.square(line)[0,0:2]))
    assert cam_proj_4.shape == (3,4)
    assert cam_proj_6.shape == (3,4)

    num_kpt = p6.shape[1]
    assert p4.shape == (2, num_kpt)
    assert p6.shape == (2, num_kpt)
    
    p4 = p4.T
    p6 = p6.T
    if fund_64 is None:
        F64 = fund_mat(cam_proj_4, cam_proj_6)
    else:
        assert fund_64.shape == (3,3)
        F64 = fund_64
    
    p4_update = np.zeros(p4.shape)
    p6_update = np.zeros(p6.shape)

    for i in range(num_kpt):
        p6_i = np.ones([3,1])
        p6_i[0:2,0] = p6[i, :]
        p4_i = np.ones([3,1])
        p4_i[0:2,0] = p4[i, :]

        epo_4 = np.dot(p6_i.T, F64)
        epo_4 = epo_4/norm_line(epo_4)
        epo_6 = np.dot(F64, p4_i).T
        epo_6 = epo_6/norm_line(epo_6)

        dis4, p4_correct_i = orth_point_line(epo_4, p4_i)
        dis6, p6_correct_i = orth_point_line(epo_6, p6_i)

        #dis4, dis6 = get_distance(p4_i.squeeze(), p6_i.squeeze(), cam_proj_4, cam_proj_6)

        if dis4 > dis6:
            p4_update[i, :] = p4[i, :]
            p6_update[i, :] = p6_correct_i[:2,0]
        else:
            p6_update[i, :] = p6[i, :]
            p4_update[i, :] = p4_correct_i[:2,0]
    #remove_homo
    point4D = cv2.triangulatePoints(cam_proj_4, cam_proj_6, p4_update.T, p6_update.T)
    #p3d = (point4D[0:3,:]/point4D[3,:]).T

    return point4D

def reproject_error(p3d, p4, p6, cam_proj_4, cam_proj_6, num_kpt=23):
    '''
    calculate reprojection error for each image 
    Args:
        p3d, p4, p6, cam_proj_4, cam_proj_6 
    Return:
        kp4_recon, kp6_recon, kp4_e, kp6_e
    '''
    assert p3d.shape == (num_kpt, 3)
    assert p4.shape == (num_kpt, 2)
    assert p6.shape == (num_kpt, 2)
    kp4_recon = compute_projection(p3d, cam_proj_4)
    kp6_recon = compute_projection(p3d, cam_proj_6)

    kp4_e = np.linalg.norm(kp4_recon - p4, axis=1)
    kp6_e = np.linalg.norm(kp6_recon - p6, axis=1)

    return kp4_recon, kp6_recon, kp4_e, kp6_e

def compute_projection(kp_3d, proj_matrix):
    '''
    Project 3d keypoints to 2d image with projection matrix 
    '''
    assert kp_3d.shape[1] == 3
    assert proj_matrix.shape == (3,4)
    kp_3d_homo = np.pad(kp_3d, ((0,0), (0,1)), mode="constant", constant_values = 1)
    kp_2d_homo = np.dot(proj_matrix, kp_3d_homo.T)
    kp_2d = kp_2d_homo[0:2,:]/kp_2d_homo[2,:]
    return kp_2d.T

def trainglate_point(P4, P6, path4, path6, out_3d_path, opencv=False):
    '''
    Triangluate point in different camera
    Return:
        data_3d_dict
    '''
    with open(path4, "r") as f:
        data_dict4 = json.load(f)
    with open(path6, "r") as f:
        data_dict6 = json.load(f)   
    data_3d = {}
    frame_list = [k for k in data_dict4.keys()]
    frame_list.sort()
    
    obj_list = []

    kp4_e_array = np.zeros([0])
    kp6_e_array = np.zeros([0])
    kp3d_array = np.zeros([0,3])
    
    F64 = fund_mat(P4, P6)
    data_3d_dict = {}
    data_3d_dict["P4"] = P4.tolist()
    data_3d_dict["P6"] = P6.tolist()
    data_3d_dict["3D"] = {}
    data_3d_dict["kp4_e"] = {}
    data_3d_dict["kp6_e"] = {}

    for f in tqdm(frame_list):
        frame_data_4 = data_dict4[f]
        frame_data_6 = data_dict6[f]
        frame_data_3d = {}

        frame_3d_dict = {}
        kp4_dict = {}
        kp6_dict = {}
        person_list = [k for k in frame_data_4.keys()]
        person_list.sort()

        for p in person_list:
            point4 = frame_data_4[p]
            point6 = frame_data_6[p]

            p4 = np.array(point4).reshape([-1,3])[:,:2]
            p6 = np.array(point6).reshape([-1,3])[:,:2]

            # p3d_org = reconstruct_point(p4, p6, P4, P6, num_kpt=23)
            # kp4_recon_org, kp6_recon_org, kp4_e_org, kp6_e_org= reproject_error(p3d_org, p4, p6, P4, P6)
            if opencv:
                p3d_mini = cv2.triangulatePoints(P4, P6, p4.T, p6.T)
            else:
                p3d_mini = reconstruct_point_mini(P4, P6, p4.T, p6.T, fund_64=F64)
            p3d_mini = (p3d_mini[0:3,:]/p3d_mini[3,:]).T
            kp4_recon_mini, kp6_recon_mini, kp4_e_mini, kp6_e_mini = reproject_error(p3d_mini, p4, p6, P4, P6)
            
            if True:
                kp4_recon = kp4_recon_mini
                kp6_recon = kp6_recon_mini
                kp4_e = kp4_e_mini
                kp6_e = kp6_e_mini
                p3d = p3d_mini

            # kp4_e_array = np.concatenate([kp4_e_array, kp4_e.reshape([-1])])
            # kp6_e_array = np.concatenate([kp6_e_array, kp6_e.reshape([-1])])
            # kp3d_array = np.concatenate([kp3d_array, p3d])
            
            frame_3d_dict[p] = p3d.tolist()
            kp4_dict[p] = kp4_e.tolist()
            kp6_dict[p] = kp6_e.tolist()
            if DEBUG:
                for obj in obj_list: obj.remove()
                obj_list = []
                obj_list += show_res(axes[0], img4, p4, kp4_recon)
                obj_list += show_res(axes[1], img6, p6, kp6_recon)
                fig.show()
                plt.waitforbuttonpress()
        
        data_3d_dict["3D"][f] = frame_3d_dict
        data_3d_dict["kp4_e"][f] = kp4_dict
        data_3d_dict["kp6_e"][f] = kp6_dict
    
    with open(out_3d_path, "w") as f:
        json.dump(data_3d_dict, f)
        print("Data 3d file saved to {}".format(out_3d_path))

    return data_3d_dict
