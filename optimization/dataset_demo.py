import json 
import os 
import numpy as np 
from os.path import join
from pose_optimize.shape_3d_info import collect_param, loss_kpt_3d, cal_kpt_3d_error
from pose_optimize.data_io import load_3d_format_crnae, convert_to_3d_format
from pose_optimize.multiview_geo import trainglate_point
from pose_optimize.optimize_loss import fintune_human_keypoint_2d, finetune_human_3d, finetune_human_3d_no_score
from AccomodateDataset import accomoDataset

def main(base_dir):
    json_input = join(base_dir, "data_3D.json")
    json_path4 = join(base_dir, "data_C4.json")
    json_path6 = join(base_dir, "data_C6.json")

    json_3d_path_init_opencv = join(base_dir,'data_3D_init_opencv.json')
    json_3d_path_init = join(base_dir,'data_3D_init.json')
    json_3d_path_fine_2d = join(base_dir,'data_3D_fine_2d.json')
    json_3d_path_fine_3d = join(base_dir,'data_3D_fine_3d.json')
    json_crane_output = join(base_dir,'data_3D_format_final.json')
    shape_prior_path = join(base_dir,'data_shape_prior.json')
    json_prior_eval = join(base_dir,'data_3D_prior_eval.json')
    
    _, _, _, _, _,_, _, P1, P2, _, _, _, _, _, _, _, _ = accomoDataset('ETHZ_dataset2')
    data_3d_dict_init = trainglate_point(P1, P2, json_path4,json_path6, json_3d_path_init_opencv, opencv=True)
    data_3d_dict_init = trainglate_point(P1, P2, json_path4,json_path6, json_3d_path_init)

    #data_3d_dict_finetune = fintune_human_keypoint_2d(P1, P2, json_path4, json_path6, json_3d_path_init_opencv,json_3d_path_fine_2d)
    
    bone_left, bone_right, median_bone, mean_bone = collect_param(json_3d_path_fine_2d, out_json_path=shape_prior_path, ratio=0.7)
    
    data_error_dict = cal_kpt_3d_error(shape_prior_path, json_3d_path_fine_2d, json_prior_eval)

    finetune_human_3d_no_score(json_3d_path_fine_2d, json_path4, json_path6, shape_prior_path, json_3d_path_fine_3d)
    convert_to_3d_format(json_3d_path_fine_3d, json_crane_output)
    
    
if __name__ == "__main__":
    print("OK")
    # base_dir = "/home/crane/Documents/RestartCVPR2020/person_keypoints_match/data/EPFL_RLC_MultiCamera_2"
    #base_dir = "/home/crane/Documents/RestartCVPR2020/person_keypoints_match/data/no_score_exp/ETHZ_dataset2"
    # base_dir = "/home/crane/Documents/RestartCVPR2020/person_keypoints_match/data/score_exp/ETHZ_dataset2"

    # main(base_dir)
    