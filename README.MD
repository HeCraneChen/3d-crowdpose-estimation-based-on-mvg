

## Multi-person 3D Pose Estimation in Crowded Scenes Based on Multi-View Geometry



Created by [He (Crane) Chen](https://www.cranehechen.com/)\*, [Pengfei Guo](https://engineering.jhu.edu/vpatel36/sciencex_teams/pengfei-guo/)\* and [Pengfei Li](https://sites.google.com/view/lipengfei) when working at the Johns Hopkins University.

<img src="https://github.com/HeCraneChen/Multi-person-3D-Pose-Estimation-in-Crowded-Scenes-Based-on-Multi-View-Geometry/blob/master/doc/wild_track.png" width="960">

### Citation
Please cite this paper if you find the repository helpful:

@article{chen2020multi,
  title={Multi-person 3D Pose Estimation in Crowded Scenes Based on Multi-View Geometry},
  author={Chen, He and Guo, Pengfei and Li, Pengfei and Lee, Gim Hee and Chirikjian, Gregory},
  journal={arXiv preprint arXiv:2007.10986},
  year={2020}
}

### Introduction
This work is based on our paper Multi-person 3D Pose Estimation in Crowded Scenes Based on Multi-View Geometry, which appeared at European Conference on Computer Vision (ECCV) 2020 Spotlight.

<img src="https://github.com/HeCraneChen/Multi-person-3D-Pose-Estimation-in-Crowded-Scenes-Based-on-Multi-View-Geometry/blob/master/doc/pipeline.png" width="960">

You can check our paper for furtuer details.

We propose a 3D crowd human pose estimation method based on multi-view geometry. Specifically, we target at overcoming the bottlenecks when departing from multi-person 3D pose estimation problem and pushing it further to dense crowd 3D pose estimation problem. Epipolar constraint is at the core for key point matching across views. However, effectiveness of this formulation is frequently challenged for denser crowd. Based on this observation, we proposed our method.

The pipeline takes images from multiple calibrated RGB cameras as input. Firstly, a human detector is used to produce bounding boxes. Secondly, a modified SPPE network, which keeps multiple peaks in one heatmap to predict occluded joints, is used to estimate 2D joints, with attention placed on feet joints. Thirdly, a combinatorial optimization problem is solved to achieve people matching across views. Finally, MLE is applied to reach a good initialization of 3D human pose estimation before the final result is obtained from MAP optimization.

In the pipeline, cross-view correspondence problem is the bottleneck. A graphical model is developed for fast cross-view matching. Instead of exhaustive searching on pixel space, matching is carried out for 2D joints. In fact, we push it further by focusing on feet joints so that matching process is significantly sped up.
The idea is to use homography matrix to warp the ground among views, so that everything on the ground surface would also be warped to another view together with the ground, which include the `feet' joints. To this light, the problem of people matching boils down to feet assignment. The metric we defined to calculate the cost on each edge is related to three elements, foot location, stride size, and stride direction.

### Note
Our method is a 3-step approach. (1) 2D pose detection, (2) matching, (3) 3D reconstruction. Data is passed through these three steps by JSON files. Each step can also be used separately or plugged into other pipelines.


### Prerequisites

- Ubuntu 16.04
- Python 3.6
- Pytorch 1.0.1

### Acknowledgement
For 2D pose detection, our proposed loss function is plugged into [CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark](https://github.com/Jeff-sjtu/CrowdPose)\. Please check their repo for installation and further information.
