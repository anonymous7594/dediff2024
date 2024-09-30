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

import os
import torch
from random import randint, choice
from utils.loss_utils import l1_loss, ssim, calculate_motion_loss, calculate_rotation_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel, DeformPredict
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import random
import timm
import torchvision.transforms as transforms


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)
    # Predict
    deform_predict = DeformPredict(dataset.is_blender, dataset.is_6dof)
    deform_predict.train_setting(opt)
    # Pretrained Model
    pretrained_model = timm.create_model('vit_base_patch16_224_dino', pretrained=True).to(device)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)


    ## UPDATE: BY FRAME ORDER
    # FRAME INTERVAL <--------------------------------------------- UPDATED
    h = 5
    # WARM-UP INTERATIONS <--------------------------------------------- UPDATED
    iteration_predict = 10000 
    # Pick a random Camera
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    ## Pick frame by order
    total_frame = len(viewpoint_stack)
    number_of_frames = list(range(0,total_frame))
    ## Key frames list
    key_frame_list = []
    for i in number_of_frames:
        if (i == 0) or (i % h == 0) or (i == len(number_of_frames)-1):
            key_frame_list.append(i)
    # for key frames only
    key_frame_list_in_loop = key_frame_list.copy()
    # for all frames
    key_frame_list_in_loop2 = key_frame_list.copy()
    key_frame_list_in_loop2.remove(total_frame - 1) # REMOVE LAST FRAME

    print('BASIN-35 EXP WITH ViT and Multiple frames at once')
    
    for iteration in range(1, opt.iterations + 1):
        if iteration <= iteration_predict:
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                                0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            ## Pick frame by order
            if not key_frame_list_in_loop:
                key_frame_list_in_loop = key_frame_list.copy()
            time_interval = 1 / len(key_frame_list_in_loop)
            frame_number = key_frame_list_in_loop.pop(randint(0, len(key_frame_list_in_loop) - 1))
        
            ## UPDATE: BY FRAME ORDER
            # Load views
            viewpoint_cam = viewpoint_stack[frame_number]

            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()
                
            # Temporal dimensionality
            fid = viewpoint_cam.fid

            if iteration < opt.warm_up:
                d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
            else:
                N = gaussians.get_xyz.shape[0]
                time_input = fid.unsqueeze(0).expand(N, -1) 
                ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device=device).expand(N, -1) * time_interval * smooth_term(iteration)
                d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
                        
            # Render
            render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
                    "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]


            # Loss
            if iteration < opt.warm_up:
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                loss.backward()
            else:
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                # overall loss
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 
                loss.backward()

            iter_end.record()

            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device('cpu')

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                        radii[visibility_filter])

                # Log and save
                cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
                if iteration in testing_iterations:
                    if cur_psnr.item() > best_psnr:
                        best_psnr = cur_psnr.item()
                        best_iteration = iteration

                if iteration in saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    deform.save_weights(args.model_path, iteration)
                    #deform_predict.save_weights(args.model_path, iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.update_learning_rate(iteration)
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    deform.optimizer.step()
                    deform.optimizer.zero_grad()
                    deform.update_learning_rate(iteration)

        ### Apply Prediction after Prediction Iterations 
        else:
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                                0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()


            ## Pick frame by order
            if not key_frame_list_in_loop2:
                key_frame_list_in_loop2 = key_frame_list.copy()
                key_frame_list_in_loop2.remove(total_frame - 1) # REMOVE LAST FRAME
            #time_interval = 1 / len(key_frame_list_in_loop)
            key_frame_before = choice(key_frame_list_in_loop2)
            key_frame_list_in_loop2.remove(key_frame_before)
            #print('LENGTH: ',len(key_frame_list_in_loop2))

            ## UPDATE: BY FRAME ORDER
            ## Load views
            viewpoint_cam_before = viewpoint_stack[key_frame_before]
            # Previous key frame
            # Nearest
            #key_frame_before =  (frame_number//h)*h # every h-th frame
            #viewpoint_cam_before = viewpoint_stack[key_frame_before]
            # Nearest
            key_frame_after =  min((key_frame_before//h + 1)*h,total_frame-1) # every h-th frame
            viewpoint_cam_after = viewpoint_stack[key_frame_after]
            #print('Successful key_frame_before: ',key_frame_before)  
            #print('Key frame after: ',key_frame_after)
            
            if key_frame_after == total_frame-1:
                frame_1 = key_frame_after - 5
                frame_2 = key_frame_after - 4
                frame_3 = key_frame_after - 3
                frame_4 = key_frame_after - 2
                frame_5 = key_frame_after - 1
                frame_6 = key_frame_after
                
                viewpoint_cam_1 = viewpoint_stack[frame_1]
                viewpoint_cam_2 = viewpoint_stack[frame_2]
                viewpoint_cam_3 = viewpoint_stack[frame_3]
                viewpoint_cam_4 = viewpoint_stack[frame_4]
                viewpoint_cam_5 = viewpoint_stack[frame_5]
                viewpoint_cam_6 = viewpoint_stack[frame_6]
            else:
                frame_1 = key_frame_before
                frame_2 = key_frame_before + 1
                frame_3 = key_frame_before + 2
                frame_4 = key_frame_before + 3
                frame_5 = key_frame_before + 4
                frame_6 = key_frame_before + 5
                viewpoint_cam_1 = viewpoint_stack[frame_1]
                viewpoint_cam_2 = viewpoint_stack[frame_2]
                viewpoint_cam_3 = viewpoint_stack[frame_3]
                viewpoint_cam_4 = viewpoint_stack[frame_4]
                viewpoint_cam_5 = viewpoint_stack[frame_5]
                viewpoint_cam_6 = viewpoint_stack[frame_6]
                

            if dataset.load2gpu_on_the_fly:
                # Reference frames
                viewpoint_cam_before.load2device()
                viewpoint_cam_after.load2device()
                # Predicted frames
                viewpoint_cam_1.load2device()
                viewpoint_cam_2.load2device()
                viewpoint_cam_3.load2device()
                viewpoint_cam_4.load2device()
                viewpoint_cam_5.load2device()
                viewpoint_cam_6.load2device()

            ## Temporal dimensionality
            # Reference frames
            fid_before = viewpoint_cam_before.fid
            fid_after = viewpoint_cam_after.fid
            # Predicted frames
            fid_1 = viewpoint_cam_1.fid
            fid_2 = viewpoint_cam_2.fid
            fid_3 = viewpoint_cam_3.fid
            fid_4 = viewpoint_cam_4.fid
            fid_5 = viewpoint_cam_5.fid
            fid_6 = viewpoint_cam_6.fid

            ## Gaussians config
            N = gaussians.get_xyz.shape[0]

            ## Expand Temporal dimension
            # Reference frames
            time_input_before = fid_before.unsqueeze(0).expand(N, -1) 
            time_input_after = fid_after.unsqueeze(0).expand(N, -1) 
            # Predicted frames
            time_input_1 = fid_1.unsqueeze(0).expand(N, -1)
            time_input_2 = fid_2.unsqueeze(0).expand(N, -1)
            time_input_3 = fid_3.unsqueeze(0).expand(N, -1)
            time_input_4 = fid_4.unsqueeze(0).expand(N, -1)
            time_input_5 = fid_5.unsqueeze(0).expand(N, -1)
            time_input_6 = fid_6.unsqueeze(0).expand(N, -1)

            #ast_noise = torch.randn(1, 1, device=device).expand(N, -1) * time_interval * smooth_term(iteration)

            with torch.no_grad():
                ### Reference frames
                ## Preceeding Key Frame  -------------------------------------------
                d_xyz_before, d_rotation_before, d_scaling_before = deform.step(gaussians.get_xyz.detach(), time_input_before)
                # Render - FIX ERROR INPUT FOR RENDER() ON JULY-3 (PLATE-51): should use deformable values for render() input rather than updated coordination values 
                image_before = render(viewpoint_cam_before, gaussians, pipe, background, d_xyz_before, d_rotation_before, d_scaling_before, dataset.is_6dof)["render"]
                d_xyz_before = gaussians.get_xyz.detach() + d_xyz_before
                #d_rotation_before = gaussians.get_rotation.detach() + d_rotation_before
                #d_scaling_before = gaussians.get_scaling.detach() + d_scaling_before
                ## Proceeding Key Frame -------------------------------------------
                d_xyz_after, d_rotation_after, d_scaling_after = deform.step(gaussians.get_xyz.detach(),time_input_after)
                # Render
                image_after = render(viewpoint_cam_after, gaussians, pipe, background, d_xyz_after, d_rotation_after, d_scaling_after, dataset.is_6dof)["render"]
                d_xyz_after = gaussians.get_xyz.detach() + d_xyz_after
                #d_rotation_after = gaussians.get_rotation.detach() + d_rotation_after
                #d_scaling_after = gaussians.get_scaling.detach() + d_scaling_after
                ## Image Embedding
                # Preceeding Key Frame
                image_before = preprocess(image_before).unsqueeze(0)
                features_before = pretrained_model.forward_features(image_before)
                #features_before = image_before
                # Proceeding Key Frame
                image_after = preprocess(image_after).unsqueeze(0)
                features_after = pretrained_model.forward_features(image_after)
                #features_after = image_after
                

            ## Prediction for predicted frames
            d_xyz_1, d_rotation_1, d_scaling_1 = deform_predict.step(time_input_1, time_input_before, time_input_after, 
                                                               d_xyz_before, d_xyz_after, features_before, features_after) 
            d_xyz_2, d_rotation_2, d_scaling_2 = deform_predict.step(time_input_2, time_input_before, time_input_after, 
                                                               d_xyz_before, d_xyz_after, features_before, features_after) 
            d_xyz_3, d_rotation_3, d_scaling_3 = deform_predict.step(time_input_3, time_input_before, time_input_after, 
                                                               d_xyz_before, d_xyz_after, features_before, features_after) 
            d_xyz_4, d_rotation_4, d_scaling_4 = deform_predict.step(time_input_4, time_input_before, time_input_after, 
                                                               d_xyz_before, d_xyz_after, features_before, features_after) 
            d_xyz_5, d_rotation_5, d_scaling_5 = deform_predict.step(time_input_5, time_input_before, time_input_after, 
                                                               d_xyz_before, d_xyz_after, features_before, features_after) 
            d_xyz_6, d_rotation_6, d_scaling_6 = deform_predict.step(time_input_6, time_input_before, time_input_after, 
                                                               d_xyz_before, d_xyz_after, features_before, features_after) 
            
            ## Render
            # Frame 1
            render_pkg_re_1 = render(viewpoint_cam_1, gaussians, pipe, background, d_xyz_1, d_rotation_1, d_scaling_1, dataset.is_6dof)
            image_1, viewspace_point_tensor_1, visibility_filter_1, radii_1 = render_pkg_re_1["render"], render_pkg_re_1[
                    "viewspace_points"], render_pkg_re_1["visibility_filter"], render_pkg_re_1["radii"]
            # Frame 2
            render_pkg_re_2 = render(viewpoint_cam_2, gaussians, pipe, background, d_xyz_2, d_rotation_2, d_scaling_2, dataset.is_6dof)
            image_2, viewspace_point_tensor_2, visibility_filter_2, radii_2 = render_pkg_re_2["render"], render_pkg_re_2[
                    "viewspace_points"], render_pkg_re_2["visibility_filter"], render_pkg_re_2["radii"]
            # Frame 3
            render_pkg_re_3 = render(viewpoint_cam_3, gaussians, pipe, background, d_xyz_3, d_rotation_3, d_scaling_3, dataset.is_6dof)
            image_3, viewspace_point_tensor_3, visibility_filter_3, radii_3 = render_pkg_re_3["render"], render_pkg_re_3[
                    "viewspace_points"], render_pkg_re_3["visibility_filter"], render_pkg_re_3["radii"]
            # Frame 4
            render_pkg_re_4 = render(viewpoint_cam_4, gaussians, pipe, background, d_xyz_4, d_rotation_4, d_scaling_4, dataset.is_6dof)
            image_4, viewspace_point_tensor_4, visibility_filter_4, radii_4 = render_pkg_re_4["render"], render_pkg_re_4[
                    "viewspace_points"], render_pkg_re_4["visibility_filter"], render_pkg_re_4["radii"]
            # Frame 5
            render_pkg_re_5 = render(viewpoint_cam_5, gaussians, pipe, background, d_xyz_5, d_rotation_5, d_scaling_5, dataset.is_6dof)
            image_5, viewspace_point_tensor_5, visibility_filter_5, radii_5 = render_pkg_re_5["render"], render_pkg_re_5[
                    "viewspace_points"], render_pkg_re_5["visibility_filter"], render_pkg_re_5["radii"]
            # Frame 6
            render_pkg_re_6 = render(viewpoint_cam_6, gaussians, pipe, background, d_xyz_6, d_rotation_6, d_scaling_6, dataset.is_6dof)
            image_6, viewspace_point_tensor_6, visibility_filter_6, radii_6 = render_pkg_re_6["render"], render_pkg_re_6[
                    "viewspace_points"], render_pkg_re_6["visibility_filter"], render_pkg_re_6["radii"]

            ### Loss
            ## Image loss
            # Predicted frames
            # Frame 1
            gt_image_1 = viewpoint_cam_1.original_image.cuda()
            Ll1_1 = l1_loss(image_1, gt_image_1)
            # Frame 2
            gt_image_2 = viewpoint_cam_2.original_image.cuda()
            Ll1_2 = l1_loss(image_2, gt_image_2)
            # Frame 3
            gt_image_3 = viewpoint_cam_3.original_image.cuda()
            Ll1_3 = l1_loss(image_3, gt_image_3)
            # Frame 4
            gt_image_4 = viewpoint_cam_4.original_image.cuda()
            Ll1_4 = l1_loss(image_4, gt_image_4)
            # Frame 5
            gt_image_5 = viewpoint_cam_5.original_image.cuda()
            Ll1_5 = l1_loss(image_5, gt_image_5)
            # Frame 6
            gt_image_6 = viewpoint_cam_6.original_image.cuda()
            Ll1_6 = l1_loss(image_6, gt_image_6)
            # All
            Ll1 = Ll1_1 + Ll1_2 + Ll1_3 + Ll1_4 + Ll1_5 + Ll1_6

            ## DSSIM Loss
            # Frame 1
            dssim_1 = 1.0 - ssim(image_1, gt_image_1)
            # Frame 2
            dssim_2 = 1.0 - ssim(image_2, gt_image_2)
            # Frame 3
            dssim_3 = 1.0 - ssim(image_3, gt_image_3)
            # Frame 4
            dssim_4 = 1.0 - ssim(image_4, gt_image_4)
            # Frame 5
            dssim_5 = 1.0 - ssim(image_5, gt_image_5)
            # Frame 6
            dssim_6 = 1.0 - ssim(image_6, gt_image_6)
            # All
            dssim = dssim_1 + dssim_2 + dssim_3 + dssim_4 + dssim_5 + dssim_6
             
            ## Motion Loss
            #d_xyz_pred = gaussians.get_xyz.detach() + d_xyz
            #motion_loss = calculate_motion_loss(d_xyz_before, d_xyz_pred, d_xyz_after)
            lambda_motion = 0
            loss = (1.0 - opt.lambda_dssim - lambda_motion) * Ll1 + opt.lambda_dssim * dssim 
            #loss = loss + lambda_motion*motion_loss  # Motion loss
            loss.backward()

            iter_end.record()

            if dataset.load2gpu_on_the_fly:
                # Reference frames
                viewpoint_cam_before.load2device('cpu')
                viewpoint_cam_after.load2device('cpu')
                # Predicted frames
                viewpoint_cam_1.load2device('cpu')
                viewpoint_cam_2.load2device('cpu')
                viewpoint_cam_3.load2device('cpu')
                viewpoint_cam_4.load2device('cpu')
                viewpoint_cam_5.load2device('cpu')
                viewpoint_cam_6.load2device('cpu')

            ## Gaussians Config
            # visibility_filter
            visibility_filter = torch.stack([visibility_filter_1,visibility_filter_2,visibility_filter_3,
                                                            visibility_filter_4,visibility_filter_5,visibility_filter_6])
            sum_visibility_filter = visibility_filter.sum(dim=0)
            visibility_filter = sum_visibility_filter >= (len(visibility_filter) // 2 + 1)
            # radii
            radii = torch.stack([radii_1,radii_2,radii_3,radii_4,radii_5,radii_6])
            radii = torch.mean(radii.float(), dim=0).to(torch.int32)
            #viewspace_point_tensor = torch.stack([viewspace_point_tensor_1, viewspace_point_tensor_2, viewspace_point_tensor_3
            #                                  , viewspace_point_tensor_4, viewspace_point_tensor_5, viewspace_point_tensor_6])
            #viewspace_point_tensor = torch.mean(viewspace_point_tensor, dim=0)
            #viewspace_point_tensor.retain_grad()


            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                        radii[visibility_filter])

                # Log and save
                cur_psnr = training_report_with_prediction(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                            testing_iterations, scene, render, (pipe, background), deform, deform_predict,
                                            dataset.load2gpu_on_the_fly, pretrained_model, dataset.is_6dof, h = 5) #pretrained_model
                if iteration in testing_iterations:
                    if cur_psnr.item() > best_psnr:
                        best_psnr = cur_psnr.item()
                        best_iteration = iteration

                if iteration in saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    deform_predict.save_weights(args.model_path, iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    gaussians.add_densification_stats(viewspace_point_tensor_1, visibility_filter_1)
                    gaussians.add_densification_stats(viewspace_point_tensor_2, visibility_filter_2)
                    gaussians.add_densification_stats(viewspace_point_tensor_3, visibility_filter_3)
                    gaussians.add_densification_stats(viewspace_point_tensor_4, visibility_filter_4)
                    gaussians.add_densification_stats(viewspace_point_tensor_5, visibility_filter_5)
                    gaussians.add_densification_stats(viewspace_point_tensor_6, visibility_filter_6)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.update_learning_rate(iteration)
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    deform_predict.optimizer.step()
                    deform_predict.optimizer.zero_grad()
                    deform_predict.update_learning_rate(iteration)  


    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device=device)
                gts = torch.tensor([], device=device)
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


def training_report_with_prediction(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, deform_predict, load2gpu_on_the_fly, pretrained_model, is_6dof=False, h = 5): # <------------------------------------------ UPDATED, pretrained_model
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device=device)
                gts = torch.tensor([], device=device)

                frame_number = 0
                total_frame = len(config['cameras'])

                for idx, viewpoint in enumerate(config['cameras']):
                    ## Preceeding key frame
                    key_frame_before =  (frame_number//h)*h
                    viewpoint_cam_before = config['cameras'][key_frame_before]
                    ## Proceeding key frame
                    key_frame_after =  min((frame_number//h + 1)*h,total_frame-1)
                    viewpoint_cam_after = config['cameras'][key_frame_after]

                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                        viewpoint_cam_before.load2device()
                        viewpoint_cam_after.load2device()

                    fid = viewpoint.fid
                    fid_before = viewpoint_cam_before.fid
                    fid_after = viewpoint_cam_after.fid

                    # Current Gaussians
                    xyz = scene.gaussians.get_xyz
                    rotation = scene.gaussians.get_rotation
                    scaling = scene.gaussians.get_scaling

                    # Temporal dimension
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    time_input_before = fid_before.unsqueeze(0).expand(xyz.shape[0], -1)
                    time_input_after = fid_after.unsqueeze(0).expand(xyz.shape[0], -1)

                    # Predict Deformable values for key frames
                    d_xyz_before, d_rotation_before, d_scaling_before = deform.step(xyz.detach(), time_input_before)
                    d_xyz_after, d_rotation_after, d_scaling_after = deform.step(xyz.detach(), time_input_after)
                    
                    ### Update Gaussian values for key frames
                    # before
                    image_before = render(viewpoint_cam_before, scene.gaussians, *renderArgs, d_xyz_before, d_rotation_before, d_scaling_before, is_6dof)["render"]
                    d_xyz_before = xyz + d_xyz_before
                    #d_rotation_before = rotation + d_rotation_before
                    #d_scaling_before = scaling + d_scaling_before
                    # after
                    image_after = render(viewpoint_cam_after, scene.gaussians, *renderArgs, d_xyz_after, d_rotation_after, d_scaling_after, is_6dof)["render"]            
                    d_xyz_after = xyz + d_xyz_after
                    #d_rotation_after = rotation + d_rotation_after
                    #d_scaling_after = scaling + d_scaling_after
  
                    ## Image Embedding
                    # Before
                    image_before = preprocess(image_before).unsqueeze(0)
                    features_before = pretrained_model.forward_features(image_before)
                    #features_before = image_before
                    # After
                    image_after = preprocess(image_after).unsqueeze(0)
                    features_after = pretrained_model.forward_features(image_after)
                    #features_after = image_after

                    ### Predict
                    d_xyz, d_rotation, d_scaling = deform_predict.step(time_input,time_input_before,time_input_after,d_xyz_before,d_xyz_after,features_before,features_after)                                                                    
                    # kl_loss, contrastive_loss

                    image = torch.clamp(
                            renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                            0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(device), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                        viewpoint_cam_before.load2device('cpu')
                        viewpoint_cam_after.load2device('cpu')

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                                image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                    gt_image[None], global_step=iteration)
                        
                    if  abs(frame_number-total_frame) == 1:
                        frame_number = 0
                    else:
                        frame_number += 1

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")