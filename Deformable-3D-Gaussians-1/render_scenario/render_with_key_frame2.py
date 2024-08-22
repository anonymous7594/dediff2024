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
from scene import Scene, DeformModel
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

gpu = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) #if not self.args.use_multi_gpu else self.args.devices
device = torch.device('cuda:{}'.format(gpu))

def render_set(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    print('NEW RENDERING 21/05/2024 ver 4')
    frame_number = 0
    total_frame = len(views)
    key_frame_deformable_value = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if (frame_number == 0) or (frame_number == (total_frame-1)) or (frame_number % 5 == 0):
            if load2gpu_on_the_fly:
                view.load2device()
            fid = view.fid
            xyz = gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            # Save deformable values
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
            dict_tensor = {'time_input':fid, 'd_xyz': d_xyz, 'd_rotation': d_rotation, 'd_scaling':d_scaling}
            key_frame_deformable_value.append(dict_tensor)
            # -----------------------
            results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
            rendering = results["render"]
            depth = results["depth"]
            depth = depth / (depth.max() + 1e-5)

            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

            if  abs(frame_number-total_frame) == 1:
                frame_number = 1     
            elif frame_number + 5 > total_frame - 1:
                frame_number = total_frame - 1       
            else:
                frame_number += 5
        
        else:
            # Load first and last frames
            #viewpoint_cam_0 = views[0]
            #viewpoint_cam_last = views[total_frame-1]
            # Load 2 nearest frames
            # Previous key frame
            key_frame_before =  (frame_number//5)*5
            #viewpoint_cam_before = views[key_frame_before]
            index_before = key_frame_before // 5
            # Proceeding key frame
            key_frame_after =  min((frame_number//5 + 1)*5,total_frame-1)
            #viewpoint_cam_after = views[key_frame_after]
            index_after = key_frame_after // 5 + min(key_frame_after % 5,1)

            if load2gpu_on_the_fly:
                view.load2device()
                #viewpoint_cam_before.load2device()
                #viewpoint_cam_after.load2device()
                #viewpoint_cam_0.load2device()
                #viewpoint_cam_last.load2device()

            fid = view.fid
            #fid_before = viewpoint_cam_before.fid
            #fid_after = viewpoint_cam_after.fid
            #fid_0 = viewpoint_cam_0.fid
            #fid_last = viewpoint_cam_last.fid
            fid_before = key_frame_deformable_value[index_before]['time_input']
            fid_after = key_frame_deformable_value[index_after]['time_input']

            xyz = gaussians.get_xyz
                
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            time_input_before = fid_before.unsqueeze(0).expand(xyz.shape[0], -1)
            time_input_after = fid_after.unsqueeze(0).expand(xyz.shape[0], -1)
            #time_input_0 = fid_0.unsqueeze(0).expand(xyz.shape[0], -1)
            #time_input_last = fid_last.unsqueeze(0).expand(xyz.shape[0], -1)
                
            #time_diff_0 = abs(time_input-time_input_0)
            time_diff_before = abs(time_input-time_input_before)
            time_diff_after = abs(time_input-time_input_after)
            #time_diff_last = abs(time_input-time_input_last)
            #perc_before = time_diff_before/(time_diff_before+time_diff_after+time_diff_0+time_diff_last)
            #perc_after = time_diff_after/(time_diff_before+time_diff_after+time_diff_0+time_diff_last)
            #perc_0 = time_diff_0/(time_diff_before+time_diff_after+time_diff_0+time_diff_last)
            #perc_last = time_diff_last/(time_diff_before+time_diff_after+time_diff_0+time_diff_last)
            perc_before = time_diff_before/(time_diff_before+time_diff_after)
            perc_after = time_diff_after/(time_diff_before+time_diff_after)

            #d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
            #d_xyz_before, d_rotation_before, d_scaling_before = deform.step(gaussians.get_xyz.detach(), time_input_before)
            #d_xyz_after, d_rotation_after, d_scaling_after = deform.step(gaussians.get_xyz.detach(), time_input_after)
            #d_xyz_0, d_rotation_0, d_scaling_0 = deform.step(gaussians.get_xyz.detach(), time_input_0)
            #d_xyz_last, d_rotation_last, d_scaling_last = deform.step(gaussians.get_xyz.detach(), time_input_last)
            # Before
            d_xyz_before = key_frame_deformable_value[index_before]['d_xyz']
            d_rotation_before = key_frame_deformable_value[index_before]['d_rotation']
            d_scaling_before = key_frame_deformable_value[index_before]['d_scaling']   
            # After
            d_xyz_after = key_frame_deformable_value[index_after]['d_xyz']
            d_rotation_after = key_frame_deformable_value[index_after]['d_rotation']
            d_scaling_after = key_frame_deformable_value[index_after]['d_scaling']
            
            # Naive average
            d_xyz = d_xyz_before*perc_before + d_xyz_after*perc_after #+ d_xyz_0*perc_0 + d_xyz_last*perc_last
            d_rotation = d_rotation_before*perc_before + d_rotation_after*perc_after #+ d_rotation_0*perc_0 + d_rotation_last*perc_last
            d_scaling = d_scaling_before*perc_before + d_scaling_after*perc_after #+ d_scaling_0*perc_0 + d_scaling_last*perc_last

            results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
            rendering = results["render"]
            depth = results["depth"]
            depth = depth / (depth.max() + 1e-5)

            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

            if (frame_number + 1) % 5 == 0:
                if (frame_number + 2) == (total_frame-1):
                    frame_number = 0
                    key_frame_deformable_value = []
                else: 
                    frame_number += 2
            elif (frame_number + 1) == (total_frame - 1):
                frame_number = 0
                key_frame_deformable_value = []
            else:
                frame_number += 1


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


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

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
                        background, deform)

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform)


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
