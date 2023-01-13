#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

#ROS
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

#PIL
from PIL import ImageDraw
from PIL import Image as PILImage

#torch
import torch
import torch.nn.functional as F
import torchvision.transforms as T

#matplotlib
import matplotlib as mpl
import matplotlib.cm as cm

import cv2
import numpy as np

import networks

#from monodepth2
import networks
from layers import *

#center of picture
xc = 310
yc = 321

yoffset = 310 
xoffset = 310
xyoffset = 280
xplus = 661
XYf = [(xc-xyoffset, yc-xyoffset), (xc+xyoffset, yc+xyoffset)]
XYb = [(xc+xplus-xyoffset, yc-xyoffset), (xc+xplus+xyoffset, yc+xyoffset)]

# image size for depth estimation
hsize = 128
wsize = 416

#bin size of our camera model
bsize = 16

#device(CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Lmin = 100
i = 0
j = 0

# for masking
mask = torch.zeros([1, hsize,wsize])
for i in range(wsize):
    for j in range(hsize):
        if ((i - (wsize/2))**2)/(wsize/2)**2 + ((j - (hsize/2))**2)/(hsize/2)**2 < 1.0:
            mask[:, j, i] = 1.0
mask_gpu = mask.repeat(3,1,1).unsqueeze(dim=0).cuda()

class BackprojectDepth_learned_camera(nn.Module):
    """Backproject function"""
    def __init__(self, batch_size, height, width, bin_size, eps=1e-7):
        super(BackprojectDepth_learned_camera, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

        if self.height == self.width:
            hw = self.height

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.ones_hw = nn.Parameter(torch.ones(self.batch_size), requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = nn.Parameter(self.pix_coords.repeat(batch_size, 1, 1).view(self.batch_size, 2, self.height, self.width), requires_grad=False)
        self.x_range_k = nn.Parameter(self.pix_coords[:,0:1,:,:], requires_grad=False)
        self.y_range_k = nn.Parameter(self.pix_coords[:,1:2,:,:], requires_grad=False)

        self.bin_size = bin_size
        self.sfmax = nn.Softmax(dim=2)

        self.cx = nn.Parameter(torch.ones((self.batch_size))*(self.width-1)/2.0, requires_grad=False)
        self.cy = nn.Parameter(torch.ones((self.batch_size))*(self.height-1)/2.0, requires_grad=False)

        self.relu = nn.ReLU()

    def forward(self, depth, alpha, beta, cam_range, cam_offset):

        index_x = cam_range[:,0:1] + self.eps
        index_y = cam_range[:,1:2] + self.eps

        offset_x = cam_offset[:,0:1] + self.eps
        offset_y = cam_offset[:,1:2] + self.eps

        self.x_range = offset_x.view(self.batch_size, 1, 1, 1) + index_x.view(self.batch_size, 1, 1, 1)*(self.x_range_k - self.cx.view(self.batch_size, 1, 1, 1))/self.cx.view(self.batch_size, 1, 1, 1)
        self.y_range = offset_y.view(self.batch_size, 1, 1, 1) + index_y.view(self.batch_size, 1, 1, 1)*(self.y_range_k - self.cy.view(self.batch_size, 1, 1, 1))/self.cy.view(self.batch_size, 1, 1, 1)

        xy_t = torch.sqrt(self.x_range**2 + self.y_range**2 + self.eps)

        bin_height_group = (alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1))*(xy_t.unsqueeze(4)) + beta.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        bin_height_c, _ = torch.min(bin_height_group, dim=4)
        bin_height = torch.clamp(bin_height_c, min=-0.0, max=1.0)

        XY_t = xy_t/(bin_height + self.eps)*depth
          
        cos_td = self.x_range/(xy_t + self.eps)
        sin_td = self.y_range/(xy_t + self.eps)

        X_t = XY_t*cos_td
        Y_t = XY_t*sin_td
        Z_t = depth

        cam_points = torch.cat([X_t, Y_t, Z_t], dim = 1).view(self.batch_size, 3, -1)
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

def preprocess_image(msg):
    ## image preprocess for Ricoh THETA S
    cv2_msg_img = bridge.imgmsg_to_cv2(msg)
    pil_msg_img = cv2.cvtColor(cv2_msg_img, cv2.COLOR_BGR2RGB)
    pil_msg_imgx = PILImage.fromarray(pil_msg_img)

    fg_img = PILImage.new('RGBA', pil_msg_imgx.size, (0, 0, 0, 255))    
    draw=ImageDraw.Draw(fg_img)
    draw.ellipse(XYf, fill = (0, 0, 0, 0))
    draw.ellipse(XYb, fill = (0, 0, 0, 0))

    pil_msg_imgx.paste(fg_img, (0, 0), fg_img.split()[3])
    cv2_img = cv2.cvtColor(pil_msg_img, cv2.COLOR_RGB2BGR)
    cv_cutimg_F = cv2_img[yc-xyoffset:yc+xyoffset, xc-xyoffset:xc+xyoffset]
    cv_cutimg_B = cv2_img[yc-xyoffset:yc+xyoffset, xc+xplus-xyoffset:xc+xplus+xyoffset]

    cv_cutimg_FF = cv2.transpose(cv_cutimg_F)
    cv_cutimg_F = cv2.flip(cv_cutimg_FF, 1)
    cv_cutimg_Bt = cv2.transpose(cv_cutimg_B)
    cv_cutimg_B = cv2.flip(cv_cutimg_Bt, 0)
    cv_cutimg_BF = cv2.flip(cv_cutimg_Bt, -1)

    cv_cutimg_n = np.concatenate((cv_cutimg_F, cv_cutimg_B), axis=1)
    return cv_cutimg_n

def callback_depthest(msg):
    cv_img = preprocess_image(msg)
    cv_trans = cv_img.transpose(2, 0, 1)
    cv_trans_np = np.array([cv_trans], dtype=np.float32)
        
    double_fisheye_gpu = torch.from_numpy(cv_trans_np).float().cuda()

    image_d = torch.cat((mask_gpu*transform(double_fisheye_gpu[:,:,:,0:2*xyoffset]).clone(), mask_gpu*transform(double_fisheye_gpu[:,:,:,2*xyoffset:4*xyoffset]).clone()), dim=0)/255.0
    image_d_flip = torch.flip(image_d, [3])
    image_dc = torch.cat((image_d, image_d_flip), dim=0)

    #depth estimation
    with torch.no_grad():
        features = enc_depth(image_dc)   
        outputs_c, camera_param_c, binwidth_c, camera_range_c, camera_offset_c = dec_depth(features)                 
        
        outputs = (outputs_c[("disp", 0)][0:2,:,:,:] + torch.flip(outputs_c[("disp", 0)][2:4],[3]))*0.5
        camera_param = (camera_param_c[0:2] + camera_param_c[2:4])*0.5
        binwidth = (binwidth_c[0:2] + binwidth_c[2:4])*0.5
        camera_range = (camera_range_c[0:2] + camera_range_c[2:4])*0.5                                        
        camera_offset = (camera_offset_c[0:2] + camera_offset_c[2:4])*0.5

    #camera model process
    cam_lens_x = []
    bdepth, _, _, _ = image_d.size()
    lens_zero = torch.zeros((bdepth, 1)).to(device)
    binwidth_zero = torch.zeros((bdepth, 1)).to(device)
    for i in range(bsize):
        lens_height = torch.zeros(bdepth, 1, device=device)
        for j in range(0, i+1):
            lens_height += camera_param[:, j:j+1]
        cam_lens_x.append(lens_height)
    cam_lens_c = torch.cat(cam_lens_x, dim=1)
    cam_lens = 1.0 - torch.cat([lens_zero, cam_lens_c], dim=1)

    lens_bincenter_x = []
    for i in range(bsize):
        bin_center = torch.zeros(bdepth, 1, device=device)
        for j in range(0, i+1):
            bin_center += binwidth[:, j:j+1]
        lens_bincenter_x.append(bin_center)
    lens_bincenter_c = torch.cat(lens_bincenter_x, dim=1)
    lens_bincenter = torch.cat([binwidth_zero, lens_bincenter_c], dim=1)
                
    lens_alpha = (cam_lens[:,1:bsize+1] - cam_lens[:,0:bsize])/(lens_bincenter[:,1:bsize+1] - lens_bincenter[:,0:bsize] + 1e-7)
    lens_beta = (-cam_lens[:,1:bsize+1]*lens_bincenter[:,0:bsize] + cam_lens[:,0:bsize]*lens_bincenter[:,1:bsize+1] + 1e-7)/(lens_bincenter[:,1:bsize+1] - lens_bincenter[:,0:bsize] + 1e-7)                
                
    double_disp = torch.cat((outputs[0:1], outputs[1:2]), dim=3)
    pred_disp_pano, pred_depth = disp_to_depth(double_disp, 0.1, 100.0)
                
    # backprojection to have point clouds
    # cam_points_f: estimated point clouds from front fisheye image on the front camera image coordinate
    # cam_points_b: estimated point clouds from back fisheye image on the back camera image coordinate
    cam_points_f = backproject_depth(pred_depth[:,:,:,0:wsize], lens_alpha[0:1], lens_beta[0:1], camera_range[0:1], camera_offset[0:1])
    cam_points_b = backproject_depth(pred_depth[:,:,:,wsize:2*wsize], lens_alpha[1:2], lens_beta[1:2], camera_range[1:2], camera_offset[1:2])

    disp_0 = 1.0/(double_disp)                
    disp_resized_np = (disp_0).squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    depth_rgb = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = bridge.cv2_to_imgmsg(depth_rgb, 'rgb8')  
    pub_depth.publish(im)

### for depth estimation ###
print("Define and Load depth models.")
enc_depth = networks.ResnetEncoder(18, True, num_input_images = 1)
path = os.path.join("./models/", "encoder.pth")
model_dict = enc_depth.state_dict()

pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
enc_depth.load_state_dict(model_dict)
enc_depth.eval().to(device)

dec_depth = networks.DepthDecoder_learned_camera(enc_depth.num_ch_enc, [0, 1, 2, 3], 16)
path = os.path.join("./models/", "depth.pth")
model_dict = dec_depth.state_dict()

pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
dec_depth.load_state_dict(model_dict)
dec_depth.eval().to(device)

### Back-projection function of our learned camera model ###
backproject_depth = BackprojectDepth_learned_camera(1, hsize, wsize, bsize)
backproject_depth.to(device)

transform = T.Resize(size = (hsize,wsize))
transform_raw = T.Resize(size = (2*xyoffset,4*xyoffset))

# main function
if __name__ == '__main__':
    # CvBridge
    bridge = CvBridge()

    #initialize node
    rospy.init_node('DepthEst_ExAug', anonymous=False)

    #subscriber of topics
    msg_pedest = rospy.Subscriber('/cv_camera_node/image_raw', Image, callback_depthest, queue_size=1)

    #publisher of topics
    pub_depth = rospy.Publisher('/estimated_depth',Image,queue_size=1)   #estimated depth

    print('waiting message .....')
    rospy.spin()
