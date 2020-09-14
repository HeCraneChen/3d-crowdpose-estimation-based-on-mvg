import numpy as np
import math
import json
from numpy import linalg as LA

def get_gt_template(gt):
    """get template in the form of 13 keypoints groundtruth
    
    Args:
    -gt: a list of length 76 [x1,y1,z1,c1,x2,y2,z2,c2,......]
    
    Returns:
    gt_form: 3 by 13 numpy array
    """
    gt_0 = np.asarray(gt).reshape((-1,4)).T[0:3,:] # 3 by 19
    
    gt_temp1 = gt_0[:,3:15].copy()
    gt_temp2 = (gt_0[:,1].copy() + (gt_0[:,15].copy() + gt_0[:,16].copy()) * 0.5 +  (gt_0[:,17].copy() + gt_0[:,18].copy()) * 0.5)/3  
    gt_temp2 = gt_temp2.reshape((3,1))
    gt_form = np.concatenate((gt_temp2,gt_temp1),axis = 1)    
    return gt_form

def get_our_template(ours):
    """get template in the form of 23 keypoints ours
    
    Args:
    -ours: a list of length 92 [x1,y1,z1,c1,x2,y2,z2,c2,......]
    
    Returns:
    ours_form: 3 by 13 numpy array
    """
    ours_0 = np.asarray(ours).reshape((-1,4)).T[0:3,:] # 3 by 23
    ours_head = (ours_0[:,0].copy() + (ours_0[:,1].copy() + ours_0[:,2].copy())*0.5 + (ours_0[:,3].copy() + ours_0[:,4].copy())*0.5)/3
#     ours_Lfoot = (ours_0[:,15].copy() + ours_0[:,17].copy() + ours_0[:,18].copy() + ours_0[:,19].copy()) * 0.25
#     ours_Rfoot = (ours_0[:,16].copy() + ours_0[:,20].copy() + ours_0[:,21].copy() + ours_0[:,22].copy()) * 0.25
    ours_Lfoot = ours_0[:,15].copy()
    ours_Rfoot = ours_0[:,16].copy()
    
    our_temp = np.concatenate((ours_0[:,5].copy().reshape((3,1)),ours_0[:,7].copy().reshape((3,1))),axis = 1)
    our_temp = np.concatenate((our_temp.copy(),ours_0[:,9].copy().reshape((3,1))),axis = 1)
    our_temp = np.concatenate((our_temp.copy(),ours_0[:,11].copy().reshape((3,1))),axis = 1)
    our_temp = np.concatenate((our_temp.copy(),ours_0[:,13].copy().reshape((3,1))),axis = 1)
    our_temp = np.concatenate((our_temp.copy(),ours_Lfoot.reshape((3,1))),axis = 1)
    
    our_temp = np.concatenate((our_temp.copy(),ours_0[:,6].copy().reshape((3,1))),axis = 1)
    our_temp = np.concatenate((our_temp.copy(),ours_0[:,8].copy().reshape((3,1))),axis = 1)
    our_temp = np.concatenate((our_temp.copy(),ours_0[:,10].copy().reshape((3,1))),axis = 1)
    our_temp = np.concatenate((our_temp.copy(),ours_0[:,12].copy().reshape((3,1))),axis = 1)
    our_temp = np.concatenate((our_temp.copy(),ours_0[:,14].copy().reshape((3,1))),axis = 1)
    our_temp = np.concatenate((our_temp.copy(),ours_Rfoot.reshape((3,1))),axis = 1)
    
    our_form = np.concatenate((ours_head.reshape((3,1)),our_temp.copy()),axis = 1)
    
    return our_form

def MPJPE_singleperson(gt_form, our_form):
    """evaluates MPJPE between ground truth and ours
    """
    center_gt = (gt_form[:,4] + gt_form[:,10]) * 0.5
    center_our = (our_form[:,4] + our_form[:,10]) * 0.5
    print('center_gt',center_gt)
    print('center_our',center_our)
    for counter in range(13):
        gt_form[:,counter] = gt_form[:,counter] - center_gt
        our_form[:,counter] = our_form[:,counter] - center_our
    a = gt_form - our_form
    MPJPE_sp_all_joint = np.array([[LA.norm(a[:, 0]),LA.norm(a[:, 1]),LA.norm(a[:, 2]),LA.norm(a[:, 3]), \
                                   LA.norm(a[:, 4]),LA.norm(a[:, 5]),LA.norm(a[:, 6]),LA.norm(a[:, 7]),\
                                   LA.norm(a[:, 8]),LA.norm(a[:, 9]),LA.norm(a[:, 10]),LA.norm(a[:, 11]),LA.norm(a[:, 12])]])
    MPJPE_sp = np.sum(MPJPE_sp_all_joint) / 13
    return MPJPE_sp, MPJPE_sp_all_joint


