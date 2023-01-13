# Monocular Depth Estimation for ExAug
 
**Summary**: Our method, ExAug can realize the navigation with obstacle avoidance by only using an RGB image. Our control policy is trained from the synthetic images from multiple datasets by minimizing our proposed objective. In our objective and algorithm to generate the synthetic images, we use estimated depth image. Here we release our code and trainied model for monocular depth estimation.
(We are planning to release our whole system of ExAug later...)

Please see the [website](https://sites.google.com/view/exaug-nav/) for more technical details.

#### Paper
**["ExAug: Robot-Conditioned Navigation Policies via Geometric Experience Augmentation"](https://arxiv.org/abs/2210.07450)**


System Requirement
=================
Ubuntu 18.04

Pytorch 1.8.0

ROS MELODIC(http://wiki.ros.org/melodic)

Nvidia GPU

How to estimate the depth image
=================

We are providing the code and the trained model to estimate the depth image. In ExAug, we use this estimated depth to calculate our proposed objectives to avoid the collision and to generate the synthetic images for the target camera.

#### Step1: Download1 (our trained model and code)
git clone https://github.com/NHirose/Monocular-depth-esatimation-in-ExAug.git
Download our trained model from here(https://drive.google.com/file/d/1De_sFYgYWtjkzNq4T7KsLTZYgdv8ZGy6/view?usp=share_link) and unzip it in the same folder.

#### Step2: Download2 (public codes, monodepth2, which is used in our codes)
We trained our model by applying our learnable camera model by Depth360(https://arxiv.org/abs/2110.10415) and pose loss into monodepth2. Thank you, monodepth2!! By having pose lose, our model can esimate the depth with scale.
cd ./Monocular-depth-esatimation-in-ExAug
git clone https://github.com/nianticlabs/monodepth2.git
cp ./monodepth2/layers.py .
cp ./monodepth2/networks/resnet_encoder.py ./networks/

#### Step3: Camera Setup
Our model for monocular depth estimation is trained on GO Stanford dataset, Recon dataset and KITTI odometry dataset.
Hence, you can feed the corresponding camera images for our model. In this example, we feed the 360-degree camera image to capture the environment around the robot.
We highly recommend to use RICOH THETA S.(https://theta360.com/en/about/theta/s.html)
Please put the camera in front of your device(robot) at the height 0.460 m not to caputure your robot itself and connect with your PC by USB cable.

#### Step4: Image Capturing
To turn on RICOH THETA S as the live streaming mode, please hold the bottom buttom at side for about 5 senconds and push the top buttom.(Detail is shown in the instrunction sheet of RICOH THETA S.)

To capture the image from RICOH THETA S, we used the open source in ROS, cv_camera_node(http://wiki.ros.org/cv_camera).
The subscribed topic name of the image is "/cv_camera_node/image".

#### Step5: Runing our depth estimation
The last process is just to run our algorithm.

python3 depth_est_exaug.py

We publish the depth image as '/estimated_depth' for the visualization.
If you need the point clouds, you can use "cam_points_f" and "cam_points_b" in line 215 and 216.
"cam_points_f" is the estimated point clouds from front side image (416x128) on the front camera image coordinate. And "cam_points_b" is the estimated point clouds from back side image (416x128) on the back camera image coordinate.

License
=================
The codes provided on this page are published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License(https://creativecommons.org/licenses/by-nc-sa/3.0/). This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. If you are interested in commercial usage you can contact us for further options. 

Citation
=================

If you use ExAug's software, please cite:

@article{hirose2022exaug,  
  title={ExAug: Robot-Conditioned Navigation Policies via Geometric Experience Augmentation},  
  author={Hirose, Noriaki and Shah, Dhruv and Sridhar, Ajay and Levine, Sergey},  
  journal={arXiv preprint arXiv:2210.07450},  
  year={2022}  
}  

@inproceedings{hirose2022depth360,  
  title={Depth360: Self-supervised Learning for Monocular Depth Estimation using Learnable Camera Distortion Model},  
  author={Hirose, Noriaki and Tahara, Kosuke},  
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},  
  pages={317--324},  
  year={2022},  
  organization={IEEE}  
}  


