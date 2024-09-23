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
from utils.loss_utils import l1_loss, ssim, calculate_motion_loss#, l2_loss#, calculate_motion_loss_ver_2
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel, DeformPredict, FeatureEnhancement
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import random
import torch.nn.functional as F
import lpips



try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)
    # Predict
    deform_predict = DeformPredict(dataset.is_blender, dataset.is_6dof)
    deform_predict.train_setting(opt)
    # Diffusion Model for Feature Enhancement
    feature_enhancement = FeatureEnhancement()
    feature_enhancement.train_setting(opt)

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
    h = 3
    ## WARM-UP INTERATIONS <--------------------------------------------- UPDATED
    #iteration_predict = 10000 
    ## LOAD ALL TRAINING FRAMES
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    ## ALL TRAINING FRAMES LENGTH
    total_frame = len(viewpoint_stack)
    number_of_frames = list(range(0,total_frame))
    number_of_frames_dm = list(range(0,total_frame))
    number_of_frames_dp = list(range(0,total_frame))
    ## ALL KEY FRAMES
    key_frame_list = []
    for i in number_of_frames:
        if (i == 0) or (i % h == 0) or (i == len(number_of_frames)-1):
            key_frame_list.append(i)
    key_frame_list_in_loop = key_frame_list.copy()
    ## SAVING OUTPUT FROM DEFORM() AND FEATURE_ENHANCEMENT()
    deform_dict = {}
    latent_dict = {}
    print('BASIN-107 EXP WITH Diffusion Model')
    

    ### DEFINE LOOP TIMESTEP
    #m_1 = 2 # to determine initial training loops for deform()
    m_1 = 10000
    m_2 = 1 # to determine training loops for feature_enhancement()
    m_3 = 1
    # Only key frames
    #loop_1_end = len(key_frame_list)*10*m_1 + len(key_frame_list)*m_2 
    #loop_1_end = total_frame*10*m_1 + len(key_frame_list)*m_2 
    # All frames
    #loop_1_end = len(key_frame_list)*10*m_1 + total_frame*m_2 
    #loop_1_end = total_frame*10*m_1 + total_frame*m_2 
    loop_1_end = m_1 + total_frame*m_2 
    loop_2_end = loop_1_end +total_frame*m_3
    #loop_2_start = loop_1_end + len(key_frame_list)*10*m_1
    #loop_2_end = loop_2_start + len(key_frame_list)*m_2 
    
    print('Number of iterations for training deform() until: ',m_1)
    print('Number of iterations for training feature_enhancement() until: ',loop_1_end)
    print('Number of iterations for training feature_enhancement() and deform_predict() with latent compilation until: ',loop_2_end)
    #print('Loop 2 starts: ',loop_2_start)
    #print('Loop 2 ends: ',loop_2_end)

    ## LATENT COMPILATION
    # Only key frames
    #latent_compiled = torch.zeros((len(key_frame_list), 4, 33, 60)).to(device)
    # All frames
    latent_compiled = torch.zeros((total_frame, 4, 33, 60)).to(device)
    #k_th = 0 # update latent_compilation index
    
    for iteration in range(1, opt.iterations + 1):
        ### STAGE 1: Training Deform() for key frames
        #if iteration < len(key_frame_list)*10*m_1:
        if iteration <= m_1: #total_frame*10*m_1:
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

            '''
            ## Pick key frame by order
            if not key_frame_list_in_loop:
                key_frame_list_in_loop = key_frame_list.copy()
            time_interval = 1 / len(key_frame_list_in_loop)
            frame_number = key_frame_list_in_loop.pop(randint(0, len(key_frame_list_in_loop) - 1))
            '''
            ## Pick all frame by order
            if not number_of_frames_dm:
                number_of_frames_dm = list(range(0,total_frame))
            time_interval = 1 / len(number_of_frames_dm)
            frame_number = number_of_frames_dm.pop(randint(0, len(number_of_frames_dm) - 1))
            
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
        
        ### STAGE 2: Apply deform() and train to get latent data for feature enhancement
        elif (iteration <= loop_1_end): # or ((iteration > loop_2_start) and (iteration < loop_2_end)):
        #if (iteration <= loop_1_end):
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

            '''
            ## Pick key frame randomly
            if not key_frame_list_in_loop:
                key_frame_list_in_loop = key_frame_list.copy()
            time_interval = 1 / len(key_frame_list_in_loop)
            frame_number = key_frame_list_in_loop.pop(randint(0, len(key_frame_list_in_loop) - 1))
            '''
            
            ## Pick frame by order
            if not number_of_frames:
                number_of_frames = list(range(0,total_frame))
            time_interval = 1 / len(number_of_frames)
            frame_number = number_of_frames.pop(randint(0, len(number_of_frames) - 1))
            #print('frame_number: ',frame_number)

            
            ## Load views
            viewpoint_cam = viewpoint_stack[frame_number]
            # Previous key frame
            key_frame_before =  (frame_number//h)*h # every h-th frame
            viewpoint_cam_before = viewpoint_stack[key_frame_before]
            # Proceeding key frame
            key_frame_after =  min((frame_number//h + 1)*h,total_frame-1) # every h-th frame
            viewpoint_cam_after = viewpoint_stack[key_frame_after]
            '''
            # Previous key frame
            key_frame_before =  max(frame_number-1,0)
            viewpoint_cam_before = viewpoint_stack[key_frame_before]
            # Proceeding key frame
            key_frame_after =  min(frame_number+1,total_frame-1) # every h-th frame
            viewpoint_cam_after = viewpoint_stack[key_frame_after]
            '''

            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()
                viewpoint_cam_before.load2device()
                viewpoint_cam_after.load2device()

            # Temporal dimensionality
            fid = viewpoint_cam.fid
            fid_before = viewpoint_cam_before.fid
            fid_after = viewpoint_cam_after.fid

            # Number of existing gaussians
            N = gaussians.get_xyz.shape[0]

            # Time dimensionality
            time_input = fid.unsqueeze(0).expand(N, -1) 
            time_input_before = fid_before.unsqueeze(0).expand(N, -1) 
            time_input_after = fid_after.unsqueeze(0).expand(N, -1) 

            ast_noise = torch.randn(1, 1, device=device).expand(N, -1) * time_interval * smooth_term(iteration)

            #if iteration < opt.warm_up:
            #    d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
            #else:
            with torch.no_grad():
                ## Prior Key  -------------------------------------------
                d_xyz_before, d_rotation_before, d_scaling_before = deform.step(gaussians.get_xyz.detach(), time_input_before)
                ## Proceeding Key Frame -------------------------------------------
                d_xyz_after, d_rotation_after, d_scaling_after = deform.step(gaussians.get_xyz.detach(),time_input_after)
                    ## Rendering the image data from key frames -> Saving image
                    #image_before = render(viewpoint_cam_before, gaussians, pipe, background, d_xyz_before, d_rotation_before, d_scaling_before, dataset.is_6dof)["render"].to(device)
                    #image_after = render(viewpoint_cam_after, gaussians, pipe, background, d_xyz_after, d_rotation_after, d_scaling_after, dataset.is_6dof)["render"].to(device)                
                    #features_before = image_before.to(device)
                    #features_after = image_after.to(device)
                    ## Saving image data from deform_dict()
                    #deform_dict[f'key_frame_{key_frame_before}'] = features_before
                    #deform_dict[f'key_frame_{key_frame_after}'] = features_after
                    #torch.save(deform_dict, os.path.join(args.model_path, 'deform_dict.pth'))
                ## Current time
                d_xyz_pre, d_rotation_pre, d_scaling_pre = deform.step(gaussians.get_xyz.detach(),time_input)
                #d_xyz_pre, d_rotation_pre, d_scaling_pre = [], [], []
                
            features_before = viewpoint_cam_before.original_image.cuda()
            features_after = viewpoint_cam_after.original_image.cuda()
                    
            ## Apply Diffusion Model to train feature_enhancement()
            time_dim = time_input[0,:]
            new_feature_latent_data = feature_enhancement.step(features_before, features_after, time_dim, viewpoint_cam)                            
            new_feature_latent_data = new_feature_latent_data.to(device)
            ## Saving a compilation of multiple latent data
            latent_compiled[frame_number] = new_feature_latent_data
            torch.save(latent_compiled, os.path.join(args.model_path, "latent_dict_compiled.pth"))
            new_feature_latent_data_all = []

            '''
                    ### Apply latent data on key frames only
                    ## Saving latent data from key frames
                    latent_dict[f'key_frame_{key_frame_before}_and_key_frame_{key_frame_after}'] = new_feature_latent_data
                    torch.save(latent_dict, os.path.join(args.model_path, "latent_dict.pth"))
                    ## Saving a compilation of multiple latent data
                    latent_compiled[k_th] = new_feature_latent_data
                    torch.save(latent_compiled, os.path.join(args.model_path, "latent_dict_compiled.pth"))
                    new_feature_latent_data_all = []
                    # update the next order for latent data 
                    if k_th + 1 < len(key_frame_list):
                        k_th += 1
                    else:
                        k_th = 0
            '''
                    
            ### Apply latent data on all frames
            ## Saving latent data from all frames
            latent_dict[f'frame_{frame_number}'] = new_feature_latent_data
            torch.save(latent_dict, os.path.join(args.model_path, "latent_dict.pth"))


            #### Prediction
            ## Update Gaussians config in key frames
            with torch.no_grad():
                ## Preceeding Key Frame
                d_xyz_before = gaussians.get_xyz.detach() + d_xyz_before
                ## Proceeding Key Frame
                d_xyz_after = gaussians.get_xyz.detach() + d_xyz_after
                #d_xyz_before = 0.0
                #d_xyz_after = 0.0
                ## Apply Prediction model
            d_xyz, d_rotation, d_scaling, loss_image_latent_encoding = deform_predict.step(time_input, time_input_before, time_input_after, gaussians.get_xyz.detach(), 
                                                                                                    d_xyz_before, d_xyz_after, new_feature_latent_data, new_feature_latent_data_all,
                                                                                                    d_xyz_pre, d_rotation_pre, d_scaling_pre) 
                    #d_xyz, d_rotation, d_scaling = deform_predict.step(time_input, gaussians.get_xyz.detach(), new_feature_latent_data)
                
            ## Render
            render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
                    "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

            ## Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            #Ll2 = l2_loss(image, gt_image)
            lambda_l2 = 0
            # Motion Loss & Latent Loss
            #if iteration < opt.warm_up:
            #    motion_loss = 0
            #    lambda_motion = 0
            #    loss_image_latent_encoding = 0
            #    lambda_latent = 0
            #else:
            d_xyz_pred = gaussians.get_xyz.detach() + d_xyz
            #t_diff_minus_h = abs(time_input-time_input_before)
            #t_diff_plus_h = abs(time_input-time_input_after)
            motion_loss = calculate_motion_loss(d_xyz_before, d_xyz_pred, d_xyz_after) #, t_diff_minus_h, t_diff_plus_h
            #motion_loss = 0
            lambda_motion = 0.2
            lambda_latent = 0
            # KL Loss
            #if frame_number == key_frame_before:
            #    image1 = image_before
            #else:
            #    image1 = image_after
            #image2 = image
            # Ensure the images are non-negative and normalized to sum to 1
            #image1 = torch.clamp(image1, min=1e-10) 
            #image2 = torch.clamp(image2, min=1e-10)
            #image1 = image1 / image1.sum()
            #image2 = image2 / image2.sum()
            # Flatten the images
            #image1_flat = image1.view(-1)
            #image2_flat = image2.view(-1)
            # Calculate KL Divergence
            #kl_loss = F.kl_div(image1_flat.log(), image2_flat, reduction='sum')
            kl_loss = 0
            lambda_kl = 0
            lpips_loss = loss_fn_vgg(image, gt_image).reshape(-1)
            lambda_lpips = 0.05
            # Overall loss
            loss = (1.0 - opt.lambda_dssim - lambda_motion - lambda_kl - lambda_l2) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) # L1 loss & dssim loss
            loss = loss + lambda_motion*motion_loss  # Motion loss
            loss = loss + lambda_kl*kl_loss # KL Loss
            #loss = loss + lambda_l2*Ll2 # L2 loss
            loss = loss + loss_image_latent_encoding*lambda_latent # Latent Loss
            loss = loss + lpips_loss*lambda_lpips # LPIPS loss

            loss.backward()

            iter_end.record()

            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device('cpu')
                viewpoint_cam_before.load2device('cpu')
                viewpoint_cam_after.load2device('cpu')


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
                #if iteration > len(key_frame_list)*10*m_1 + total_frame: 
                if iteration > m_1 + total_frame:
                    cur_psnr = training_report_with_prediction(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                            testing_iterations, scene, render, (pipe, background), deform, deform_predict,
                                            dataset.load2gpu_on_the_fly, dataset.is_6dof, h = 3,using_latent=True,pretrain_latent=True) #pretrained_model  <--------------------------------------------- UPDATED
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
                    deform_predict.optimizer.step()
                    deform_predict.optimizer.zero_grad()
                    deform_predict.update_learning_rate(iteration)    
                    #if iteration < len(key_frame_list)*10*m_1 + total_frame*m_2: 
                    feature_enhancement.optimizer.step()
                    feature_enhancement.optimizer.zero_grad()
                    feature_enhancement.update_learning_rate(iteration)
        
        
        ### STAGE 3: Training Feature_Enhancement() and Deform_Predict() with latent data all
        elif (iteration <= loop_2_end):
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

            '''
            ## Pick key frame randomly
            if not key_frame_list_in_loop:
                key_frame_list_in_loop = key_frame_list.copy()
            time_interval = 1 / len(key_frame_list_in_loop)
            frame_number = key_frame_list_in_loop.pop(randint(0, len(key_frame_list_in_loop) - 1))
            '''
            
            ## Pick frame by order
            if not number_of_frames:
                #print('IT IS THE END')
                number_of_frames = list(range(0,total_frame))
            time_interval = 1 / len(number_of_frames)
            frame_number = number_of_frames.pop(randint(0, len(number_of_frames) - 1))
            #print('frame_number: ',frame_number)

            ## Load views
            viewpoint_cam = viewpoint_stack[frame_number]
            # Previous key frame
            key_frame_before =  (frame_number//h)*h # every h-th frame
            viewpoint_cam_before = viewpoint_stack[key_frame_before]
            # Proceeding key frame
            key_frame_after =  min((frame_number//h + 1)*h,total_frame-1) # every h-th frame
            viewpoint_cam_after = viewpoint_stack[key_frame_after]
            '''
            # Previous key frame
            key_frame_before =  max(frame_number-1,0)
            viewpoint_cam_before = viewpoint_stack[key_frame_before]
            # Proceeding key frame
            key_frame_after =  min(frame_number+1,total_frame-1) # every h-th frame
            viewpoint_cam_after = viewpoint_stack[key_frame_after]
            '''

            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()
                viewpoint_cam_before.load2device()
                viewpoint_cam_after.load2device()

            # Temporal dimensionality
            fid = viewpoint_cam.fid
            fid_before = viewpoint_cam_before.fid
            fid_after = viewpoint_cam_after.fid

            # Number of existing gaussians
            N = gaussians.get_xyz.shape[0]

            # Time dimensionality
            time_input = fid.unsqueeze(0).expand(N, -1) 
            time_input_before = fid_before.unsqueeze(0).expand(N, -1) 
            time_input_after = fid_after.unsqueeze(0).expand(N, -1) 

            ast_noise = torch.randn(1, 1, device=device).expand(N, -1) * time_interval * smooth_term(iteration)

            with torch.no_grad():
            ## Prior Key  -------------------------------------------
                d_xyz_before, d_rotation_before, d_scaling_before = deform.step(gaussians.get_xyz.detach(), time_input_before)
            ## Proceeding Key Frame -------------------------------------------
                d_xyz_after, d_rotation_after, d_scaling_after = deform.step(gaussians.get_xyz.detach(),time_input_after)
            
                ## Rendering the image data from key frames -> Saving image
                #image_before = render(viewpoint_cam_before, gaussians, pipe, background, d_xyz_before, d_rotation_before, d_scaling_before, dataset.is_6dof)["render"].to(device)
                #image_after = render(viewpoint_cam_after, gaussians, pipe, background, d_xyz_after, d_rotation_after, d_scaling_after, dataset.is_6dof)["render"].to(device)                
                #features_before = image_before.to(device)
                #features_after = image_after.to(device)
                ## Saving image data from deform_dict()
                #deform_dict[f'key_frame_{key_frame_before}'] = features_before
                #deform_dict[f'key_frame_{key_frame_after}'] = features_after
                #torch.save(deform_dict, os.path.join(args.model_path, 'deform_dict.pth'))
            ## Current timestep
                d_xyz_pre, d_rotation_pre, d_scaling_pre = deform.step(gaussians.get_xyz.detach(),time_input)
                #d_xyz_pre, d_rotation_pre, d_scaling_pre = [], [], []

            features_before = viewpoint_cam_before.original_image.cuda()
            features_after = viewpoint_cam_after.original_image.cuda()
            
            ## Apply Diffusion Model to train feature_enhancement()
            time_dim = time_input[0,:]
            new_feature_latent_data = feature_enhancement.step(features_before, features_after, time_dim, viewpoint_cam)                                                                                 
            new_feature_latent_data = new_feature_latent_data.to(device)
            ## Saving a compilation of multiple latent data
            latent_compiled[frame_number] = new_feature_latent_data
            torch.save(latent_compiled, os.path.join(args.model_path, "latent_dict_compiled.pth"))
            


            '''
            ### Apply latent data on key frames only
            ## Saving latent data from key frames
            latent_dict[f'key_frame_{key_frame_before}_and_key_frame_{key_frame_after}'] = new_feature_latent_data
            torch.save(latent_dict, os.path.join(args.model_path, "latent_dict.pth"))
            ## Saving a compilation of multiple latent data
            latent_compiled[k_th] = new_feature_latent_data
            torch.save(latent_compiled, os.path.join(args.model_path, "latent_dict_compiled.pth"))
            new_feature_latent_data_all = []
            # update the next order for latent data 
            if k_th + 1 < len(key_frame_list):
                k_th += 1
            else:
                k_th = 0
            '''
            
            
            ### Apply latent data on all frames
            ## Saving latent data from all frames
            latent_dict[f'frame_{frame_number}'] = new_feature_latent_data
            torch.save(latent_dict, os.path.join(args.model_path, "latent_dict.pth"))
            
            
            ## Saving a compilation of multiple latent data
            new_feature_latent_data_all = torch.load(os.path.join(args.model_path, "latent_dict_compiled.pth"))
            new_feature_latent_data_all = new_feature_latent_data_all.to(device)
            

            #### Prediction
            ## Update Gaussians config in key frames
            with torch.no_grad():
                ## Preceeding Key Frame
                d_xyz_before = gaussians.get_xyz.detach() + d_xyz_before
                ## Proceeding Key Frame
                d_xyz_after = gaussians.get_xyz.detach() + d_xyz_after
            ## Apply Prediction model
            d_xyz, d_rotation, d_scaling, loss_image_latent_encoding = deform_predict.step(time_input, time_input_before, time_input_after, gaussians.get_xyz.detach(), 
                                                               d_xyz_before, d_xyz_after, new_feature_latent_data, new_feature_latent_data_all,
                                                               d_xyz_pre, d_rotation_pre, d_scaling_pre) 
            #d_xyz, d_rotation, d_scaling = deform_predict.step(time_input, gaussians.get_xyz.detach(), new_feature_latent_data)
            
            ## Render
            render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
                    "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

            ## Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            #Ll2 = l2_loss(image, gt_image)
            lambda_l2 = 0
            # Motion Loss
            d_xyz_pred = gaussians.get_xyz.detach() + d_xyz
            #t_diff_minus_h = abs(time_input-time_input_before)
            #t_diff_plus_h = abs(time_input-time_input_after)
            motion_loss = calculate_motion_loss(d_xyz_before, d_xyz_pred, d_xyz_after) #, t_diff_minus_h, t_diff_plus_h
            #motion_loss = 0.2
            lambda_motion = 0.2
            # KL Loss
            #if frame_number == key_frame_before:
            #    image1 = image_before
            #else:
            #    image1 = image_after
            #image2 = image
            # Ensure the images are non-negative and normalized to sum to 1
            #image1 = torch.clamp(image1, min=1e-10) 
            #image2 = torch.clamp(image2, min=1e-10)
            #image1 = image1 / image1.sum()
            #image2 = image2 / image2.sum()
            # Flatten the images
            #image1_flat = image1.view(-1)
            #image2_flat = image2.view(-1)
            # Calculate KL Divergence
            #kl_loss = F.kl_div(image1_flat.log(), image2_flat, reduction='sum')
            kl_loss = 0
            lambda_kl = 0
            lambda_latent = 0
            lpips_loss = loss_fn_vgg(image, gt_image).reshape(-1)
            lambda_lpips = 0.05
            # Overall loss
            loss = (1.0 - opt.lambda_dssim - lambda_motion - lambda_kl - lambda_l2) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) # L1 loss & dssim loss
            loss = loss + lambda_motion*motion_loss  # Motion loss
            loss = loss + lambda_kl*kl_loss # KL Loss
            #loss = loss + lambda_l2*Ll2 # L2 loss
            loss = loss + loss_image_latent_encoding*lambda_latent # Latent loss
            loss = loss + lambda_lpips*lpips_loss # LPIPS loss
            loss.backward()

            iter_end.record()

            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device('cpu')
                viewpoint_cam_before.load2device('cpu')
                viewpoint_cam_after.load2device('cpu')


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
                #if iteration > len(key_frame_list)*10*m_1 + total_frame: 
                if iteration > m_1 + total_frame:
                    cur_psnr = training_report_with_prediction(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                            testing_iterations, scene, render, (pipe, background), deform, deform_predict,
                                            dataset.load2gpu_on_the_fly, dataset.is_6dof, h = 3,using_latent=True,pretrain_latent=False) #pretrained_model  <--------------------------------------------- UPDATED
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
                    deform_predict.optimizer.step()
                    deform_predict.optimizer.zero_grad()
                    deform_predict.update_learning_rate(iteration)    
                    #if iteration < len(key_frame_list)*10*m_1 + total_frame*m_2: 
                    feature_enhancement.optimizer.step()
                    feature_enhancement.optimizer.zero_grad()
                    feature_enhancement.update_learning_rate(iteration)

        
        ### STAGE 4: Training Defomn_Predict() with compiled latent data
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
            if not number_of_frames_dp:
                number_of_frames_dp = list(range(0,total_frame))
            frame_number = number_of_frames_dp.pop(randint(0, len(number_of_frames_dp) - 1))


            viewpoint_cam = viewpoint_stack[frame_number]

            
            ## Load views
            # Previous key frame
            key_frame_before =  (frame_number//h)*h # every h-th frame
            viewpoint_cam_before = viewpoint_stack[key_frame_before]
            # Proceeding key frame
            key_frame_after =  min((frame_number//h + 1)*h,total_frame-1) # every h-th frame
            viewpoint_cam_after = viewpoint_stack[key_frame_after]
            '''
            # Previous key frame
            key_frame_before =  max(frame_number-1,0)
            viewpoint_cam_before = viewpoint_stack[key_frame_before]
            # Proceeding key frame
            key_frame_after =  min(frame_number+1,total_frame-1) # every h-th frame
            viewpoint_cam_after = viewpoint_stack[key_frame_after]
            '''
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()
                viewpoint_cam_before.load2device()
                viewpoint_cam_after.load2device()

            # Temporal dimensionality
            fid = viewpoint_cam.fid
            fid_before = viewpoint_cam_before.fid
            fid_after = viewpoint_cam_after.fid

            # Number of existing gaussians
            N = gaussians.get_xyz.shape[0]

            # Temporal data
            time_input = fid.unsqueeze(0).expand(N, -1) 
            time_input_before = fid_before.unsqueeze(0).expand(N, -1) 
            time_input_after = fid_after.unsqueeze(0).expand(N, -1) 

            ast_noise = torch.randn(1, 1, device=device).expand(N, -1) * time_interval * smooth_term(iteration)

            with torch.no_grad():
                ## Prior Key  -------------------------------------------
                d_xyz_before, d_rotation_before, d_scaling_before = deform.step(gaussians.get_xyz.detach(), time_input_before)
                ## Proceeding Key Frame -------------------------------------------
                d_xyz_after, d_rotation_after, d_scaling_after = deform.step(gaussians.get_xyz.detach(),time_input_after)
                ## Current timestep
                d_xyz_pre, d_rotation_pre, d_scaling_pre = deform.step(gaussians.get_xyz.detach(),time_input)
                #d_xyz_pre, d_rotation_pre, d_scaling_pre = [], [], []

            ## Loading latent data
            
            #### Existing one
            ## all frames
            latent_dict = torch.load(os.path.join(args.model_path, "latent_dict.pth"))
            new_feature_latent_data = latent_dict[f'frame_{frame_number}']
            ## key frames
            #new_feature_latent_data = torch.load(os.path.join(args.model_path, "latent_dict.pth"))
            #new_feature_latent_data = new_feature_latent_data[f'key_frame_{key_frame_before}_and_key_frame_{key_frame_after}']
            # If using all latent data
            new_feature_latent_data_all = torch.load(os.path.join(args.model_path, "latent_dict_compiled.pth"))
            new_feature_latent_data_all = new_feature_latent_data_all.to(device)
            

            '''
            #### Apply function
            ## Apply Diffusion Model to train feature_enhancement()
            with torch.no_grad():
                time_dim = time_input[0,:]
                new_feature_latent_data = feature_enhancement.step(features_before, features_after, time_dim, viewpoint_cam)
            new_feature_latent_data_all = torch.load(os.path.join(args.model_path, "latent_dict_compiled.pth"))
            new_feature_latent_data_all = new_feature_latent_data_all.to(device)
            '''


            #### Prediction
            ## Update Gaussians config in key frames
            with torch.no_grad():
                ## Preceeding Key Frame
                d_xyz_before = gaussians.get_xyz.detach() + d_xyz_before
                ## Proceeding Key Frame
                d_xyz_after = gaussians.get_xyz.detach() + d_xyz_after
            ## Apply Prediction model
            d_xyz, d_rotation, d_scaling, loss_image_latent_encoding = deform_predict.step(time_input, time_input_before, time_input_after,  gaussians.get_xyz.detach(), 
                                                               d_xyz_before, d_xyz_after, new_feature_latent_data, new_feature_latent_data_all,
                                                               d_xyz_pre, d_rotation_pre, d_scaling_pre) 
            #d_xyz, d_rotation, d_scaling = deform_predict.step(time_input, gaussians.get_xyz.detach(), new_feature_latent_data)

            # Render
            render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
                    "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

            ## Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            #Ll2 = l2_loss(image,gt_image)
            lambda_l2 = 0
            # Motion Loss
            d_xyz_pred = gaussians.get_xyz.detach() + d_xyz
            #t_diff_minus_h = abs(time_input-time_input_before)
            #t_diff_plus_h = abs(time_input-time_input_after)
            motion_loss = calculate_motion_loss(d_xyz_before, d_xyz_pred, d_xyz_after) #, t_diff_minus_h, t_diff_plus_h
            #motion_loss = 0
            lambda_motion = 0.2
            lambda_latent = 0
            lpips_loss = loss_fn_vgg(image, gt_image).reshape(-1)
            lambda_lpips = 0.05
            # Overall loss
            loss = (1.0 - opt.lambda_dssim - lambda_motion - lambda_l2) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) # L1 loss and dssim loss
            loss = loss + lambda_motion*motion_loss  # Motion loss
            #loss = loss + lambda_l2*Ll2
            loss = loss + loss_image_latent_encoding*lambda_latent # Latent loss
            loss = loss + lpips_loss*lambda_lpips # LPIPS loss
            loss.backward()

            iter_end.record()

            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device('cpu')
                viewpoint_cam_before.load2device('cpu')
                viewpoint_cam_after.load2device('cpu')


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
                #if iteration > len(key_frame_list)*10*m_1 + total_frame: 
                cur_psnr = training_report_with_prediction(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                            testing_iterations, scene, render, (pipe, background), deform, deform_predict,
                                            dataset.load2gpu_on_the_fly, dataset.is_6dof, h = 3,using_latent=True,pretrain_latent=False) #pretrained_model  <--------------------------------------------- UPDATED
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
                    deform_predict.optimizer.step()
                    deform_predict.optimizer.zero_grad()
                    deform_predict.update_learning_rate(iteration)    
                    #feature_enhancement.optimizer.step()
                    #feature_enhancement.optimizer.zero_grad()
                    #feature_enhancement.update_learning_rate(iteration)

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
                    gt_image = torch.clamp(viewpoint.original_image.to(device), 0.0, 1.0)
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
                    renderArgs, deform, deform_predict, load2gpu_on_the_fly, is_6dof=False, h = 3, using_latent=True, pretrain_latent=True): #UPDATED: h, using_latent, 
    
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
                                           range(0, 6, 1)]}) #----------------------------------------------------------------------------------> UPDATE

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device=device)
                gts = torch.tensor([], device=device)

                frame_number = 0
                total_frame = len(config['cameras'])

                for idx, viewpoint in enumerate(config['cameras']):
                    
                    if frame_number == 10 and config['name'] == 'train': #----------------------------------------------------------------------------------> UPDATE
                        continue

                    
                    ## Previous key frame
                    key_frame_before =  (frame_number//h)*h
                    viewpoint_cam_before = config['cameras'][key_frame_before]
                    ## Proceeding key frame
                    key_frame_after =  min((frame_number//h + 1)*h,total_frame-1)
                    viewpoint_cam_after = config['cameras'][key_frame_after]
                    '''
                    # Previous key frame
                    key_frame_before =  max(frame_number-1,0)
                    viewpoint_cam_before = config['cameras'][key_frame_before]
                    # Proceeding key frame
                    key_frame_after =  min(frame_number+1,total_frame-1) # every h-th frame
                    viewpoint_cam_after = config['cameras'][key_frame_after]
                    '''
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

                    # Time dimension
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    time_input_before = fid_before.unsqueeze(0).expand(xyz.shape[0], -1)
                    time_input_after = fid_after.unsqueeze(0).expand(xyz.shape[0], -1)


                    # Predict Deformable values for key frames
                    d_xyz_before, d_rotation_before, d_scaling_before = deform.step(xyz.detach(), time_input_before)
                    d_xyz_after, d_rotation_after, d_scaling_after = deform.step(xyz.detach(), time_input_after)
                    d_xyz_pre, d_rotation_pre, d_scaling_pre = deform.step(xyz.detach(), time_input)
                    
                    ### Update Gaussian values for key frames
                    ## predetermined
                    # before
                    d_xyz_before = xyz + d_xyz_before
                    # after
                    d_xyz_after = xyz + d_xyz_after

                    ### Prediction
                    if using_latent:
                        if pretrain_latent:
                            '''
                            # Using key frame
                            latent_dict = torch.load(os.path.join(args.model_path, "latent_dict.pth"))
                            new_feature_latent_data = latent_dict[f'key_frame_{key_frame_before}_and_key_frame_{key_frame_after}']
                            new_feature_latent_data = new_feature_latent_data.to(device)
                            new_feature_latent_data_all = []
                            '''

                            # Using all frames
                            latent_dict = torch.load(os.path.join(args.model_path, "latent_dict.pth"))
                            new_feature_latent_data = latent_dict[f'frame_{frame_number}']
                            new_feature_latent_data = new_feature_latent_data.to(device)
                            new_feature_latent_data_all = []
                            
                        else:
                            '''
                            # Using key frame
                            latent_dict = torch.load(os.path.join(args.model_path, "latent_dict.pth"))
                            new_feature_latent_data = latent_dict[f'key_frame_{key_frame_before}_and_key_frame_{key_frame_after}']
                            new_feature_latent_data = new_feature_latent_data.to(device)
                            '''
                        
                    
                            # Using all frames
                            latent_dict = torch.load(os.path.join(args.model_path, "latent_dict.pth"))
                            new_feature_latent_data = latent_dict[f'frame_{frame_number}']
                            new_feature_latent_data = new_feature_latent_data.to(device)
                            
                            new_feature_latent_data_all = torch.load(os.path.join(args.model_path, "latent_dict_compiled.pth"))
                            new_feature_latent_data_all = new_feature_latent_data_all.to(device)
                    else:
                        new_feature_latent_data = torch.tensor([]).to(device)
                    d_xyz, d_rotation, d_scaling, loss_image_latent_encoding = deform_predict.step(time_input, time_input_before, time_input_after, xyz.detach(),
                                                                                                    d_xyz_before, d_xyz_after, new_feature_latent_data, new_feature_latent_data_all,
                                                                                                    d_xyz_pre, d_rotation_pre, d_scaling_pre )        
                    #d_xyz, d_rotation, d_scaling = deform_predict.step(time_input, xyz.detach(), new_feature_latent_data)                                                            

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
