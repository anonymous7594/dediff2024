#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, DeformModel, DeformPredict, FeatureEnhancement #, Deform_Predict_ver_2#, Deform_Predict_ver_3
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
#import timm
import torchvision.transforms as transforms


gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) #if not self.args.use_multi_gpu else self.args.devices
device = torch.device('cuda:{}'.format(gpu))


#preprocess = transforms.Compose([
#    transforms.Resize((224, 224)),  # Resize the image to 224x224
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
#])

def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform, deform_predict, using_latent=True): #pretrained_model, using_latent, 
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame_number = 0
    total_frame = len(views)
    #print('Lenght of views: ',len(views))
    h = 3 ### <--------------------------------------------------------------------------------------------------------------- MANUALLY UPDATED

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        '''
        # Previous key frame
        key_frame_before =  max(frame_number-1,0)
        viewpoint_cam_before = viewpoint_stack[key_frame_before]
        # Proceeding key frame
        key_frame_after =  min(frame_number+1,total_frame-1) # every h-th frame
        viewpoint_cam_after = viewpoint_stack[key_frame_after]
        '''
        
        # Previous key frame
        key_frame_before =  (frame_number//h)*h
        viewpoint_cam_before = views[key_frame_before]
        # Proceeding key frame
        key_frame_after =  min((frame_number//h + 1)*h,total_frame-1)
        viewpoint_cam_after = views[key_frame_after]

        if load2gpu_on_the_fly:
            view.load2device()
            viewpoint_cam_before.load2device()
            viewpoint_cam_after.load2device()

        # Temporal features
        fid = view.fid
        fid_before = viewpoint_cam_before.fid
        fid_after = viewpoint_cam_after.fid

        # Gaussian features
        xyz = gaussians.get_xyz
        #rotation = gaussians.get_rotation
        #scaling = gaussians.get_scaling

        # Change shapre for temporal features        
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        time_input_before = fid_before.unsqueeze(0).expand(xyz.shape[0], -1)
        time_input_after = fid_after.unsqueeze(0).expand(xyz.shape[0], -1)

        # Get the deformable values for key frames
        d_xyz_before, d_rotation_before, d_scaling_before = deform.step(xyz.detach(), time_input_before)
        d_xyz_after, d_rotation_after, d_scaling_after = deform.step(xyz.detach(),time_input_after)
        d_xyz_pre, d_rotation_pre, d_scaling_pre = deform.step(xyz.detach(),time_input)
        #d_xyz_pre, d_rotation_pre, d_scaling_pre = [], [], []

        # Get the Gaussians features for key frames
        # Before
        #image_before = render(viewpoint_cam_before, gaussians, pipeline, background, d_xyz_before, d_rotation_before, d_scaling_before, is_6dof)["render"]
        d_xyz_before = xyz + d_xyz_before
        #d_rotation_before = rotation + d_rotation_before
        #d_scaling_before = scaling + d_scaling_before
        # After
        #image_after = render(viewpoint_cam_after, gaussians, pipeline, background, d_xyz_after, d_rotation_after, d_scaling_after, is_6dof)["render"]
        d_xyz_after = xyz + d_xyz_after
        #d_rotation_after = rotation + d_rotation_after
        #d_scaling_after = scaling + d_scaling_after
        ## Image Embedding
        # Before
        #image_before = preprocess(image_before).unsqueeze(0)
        #features_before = pretrained_model.forward_features(image_before)
        #features_before = image_before
        # After
        #image_after = preprocess(image_after).unsqueeze(0)
        #eatures_after = pretrained_model.forward_features(image_after)
        #features_after = image_after
            
        ## Prediction
        #if using_latent:
        '''
            # using key frames only
            #print('USING LATENT............... SUCCESSFULLY')
            latent_dict = torch.load(os.path.join(args.model_path, "latent_dict.pth"))
            new_feature_latent_data = latent_dict[f'key_frame_{key_frame_before}_and_key_frame_{key_frame_after}']
            new_feature_latent_data = new_feature_latent_data.to(device)
            #print('LOADING LATENT DATA..................SUCCESSFULLY')
            '''

        new_feature_latent_data_all = torch.load(os.path.join(args.model_path, "latent_dict_compiled.pth"))
        new_feature_latent_data_all = new_feature_latent_data_all.to(device)
            
        # using all frames
        latent_dict = torch.load(os.path.join(args.model_path, "latent_dict.pth"))
        new_feature_latent_data = latent_dict[f'frame_{frame_number}']
            #print(new_feature_latent_data.size())
        #else:
            # using all frames
        #    latent_dict = torch.load(os.path.join(args.model_path, "latent_dict.pth"))
        #    new_feature_latent_data = latent_dict[f'frame_{frame_number}']

        #    new_feature_latent_data_all = []
            #print('NO LATENT DATA..................SUCCESSFULLY')

        '''
        ## Rendering the image data from key frames -> Saving image
        image_before = render(viewpoint_cam_before, gaussians, pipeline, background, d_xyz_before, d_rotation_before, d_scaling_before, is_6dof)["render"].to(device)
        image_after = render(viewpoint_cam_after, gaussians, pipeline, background, d_xyz_after, d_rotation_after, d_scaling_after, is_6dof)["render"].to(device)                
        features_before = image_before.to(device)
        features_after = image_after.to(device)
        time_dim = time_input[0,:]
        new_feature_latent_data = feature_enhancement.step(features_before,features_after,time_dim)
        #new_feature_latent_data = new_feature_latent_data.unsqueeze(0)
        new_feature_latent_data = new_feature_latent_data.to(device)
        '''
        d_xyz, d_rotation, d_scaling, latent_loss = deform_predict.step(time_input, time_input_before, time_input_after, xyz.detach(),
                                                           d_xyz_before, d_xyz_after, new_feature_latent_data, new_feature_latent_data_all,
                                                           d_xyz_pre, d_rotation_pre, d_scaling_pre) #(features_before, features_after)
        #d_xyz, d_rotation, d_scaling = deform_predict.step(time_input+ast_noise, time_input_before, time_input_after, d_xyz_before, d_xyz_after, new_feature_latent_data) 
        #d_xyz, d_rotation, d_scaling = deform_predict.step(time_input, xyz.detach(), new_feature_latent_data)
        #d_xyz, d_rotation, d_scaling, h_input = deform.step(xyz.detach(), time_input)   
        #d_xyz, d_rotation, d_scaling = deform_predict.step(time_input, h_input, new_feature_latent_data)

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

        if  abs(frame_number-total_frame) == 1:
            frame_number = 0
        else:
            frame_number += 1


### ------------------------------------------------- NOT UPDATED YET ---------------------------------------------------- ###
def interpolate_time(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_view(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, timer):
    render_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    frame = 150
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    render_poses = torch.stack(render_wander_path(view), 0)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
    #                            0)

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = view.fid

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        # acc = results["acc"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))
        # torchvision.utils.save_image(acc, os.path.join(acc_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_all(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 150
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_poses(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, timer):
    render_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)
    frame = 520
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view_begin = views[0]  # Choose a specific time for rendering
    view_end = views[-1]
    view = views[idx]

    R_begin = view_begin.R
    R_end = view_end.R
    t_begin = view_begin.T
    t_end = view_end.T

    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = view.fid

        ratio = i / (frame - 1)

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def interpolate_view_original(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background,
                              timer):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 1000
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for view in views:
        R.append(view.R)
        T.append(view.T)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        R_begin = view_begin.R
        R_end = view_end.R
        t_begin = view_begin.T
        t_end = view_end.T

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


#### OVERALL FUNCTION ------------------------------------------------------------------------------------------------------------------------------------- ####
def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # Deformable 3D-GS
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)
        # Predict 
        deform_predict = DeformPredict(dataset.is_blender, dataset.is_6dof)
        #deform_predict = Deform_Predict_ver_3(dataset.is_blender, dataset.is_6dof)
        deform_predict.load_weights(dataset.model_path)
        # Foundation model
        #pretrained_model = timm.create_model('vit_base_patch16_224_dino', pretrained=True).to(device)
        # Feature Enhancement
        #feature_enhancement = FeatureEnhancement()
        #feature_enhancement.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "view":
            render_func = interpolate_view
        elif mode == "pose":
            render_func = interpolate_poses
        elif mode == "original":
            render_func = interpolate_view_original
        else:
            render_func = interpolate_all

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, deform, deform_predict, using_latent=True) #pretrained_model,feature_enhancement

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform,deform_predict, using_latent=True) #pretrained_model, feature_enhancement


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode)
