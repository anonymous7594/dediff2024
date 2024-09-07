import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import Deform_Predict#, Deform_Predict_with_Gaussians_distribution, Deform_Predict_ver2#, Deform_Predict_ver3
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformPredict:
    def __init__(self, is_blender=False, is_6dof=False):
        self.deform = Deform_Predict(is_blender=is_blender, is_6dof=is_6dof).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, time_input, time_input_before, time_input_after, xyz, 
             d_xyz_before, d_xyz_after, new_feature_latent_data, new_feature_latent_data_all): #features_before, features_after): #, d_xyz_current):
        return self.deform(time_input, time_input_before, time_input_after, xyz, 
                           d_xyz_before, d_xyz_after, new_feature_latent_data, new_feature_latent_data_all) # features_before, features_after)
    #def step(self, time_input,time_input_before,time_input_after,d_xyz_before, d_xyz_after
    #            ,random_time_input_before,random_time_input_after,random_d_xyz_before, random_d_xyz_after):
    #    return self.deform(time_input,time_input_before,time_input_after,d_xyz_before, d_xyz_after
    #            ,random_time_input_before,random_time_input_after,random_d_xyz_before, random_d_xyz_after)
    #def step(self, time_input, xyz, new_feature_latent_data): 
    #    return self.deform(time_input, xyz, new_feature_latent_data)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform_predict"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init_predict * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final_predict,
                                                       lr_delay_mult=training_args.position_lr_delay_mult_predict,
                                                       max_steps=training_args.predict_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform_predict/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform_predict.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform_predict"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform_predict/iteration_{}/deform_predict.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform_predict":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            

class Deform_Predict_ver_2:
    def __init__(self, is_blender=False, is_6dof=False):
        self.deform = Deform_Predict_ver2(is_blender=is_blender, is_6dof=is_6dof).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, time_input,time_input_before,time_input_after,d_xyz_before, d_xyz_after):
        return self.deform(time_input,time_input_before,time_input_after,d_xyz_before, d_xyz_after)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform_predict"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init_predict * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final_predict,
                                                       lr_delay_mult=training_args.position_lr_delay_mult_predict,
                                                       max_steps=training_args.predict_lr_max_steps,
                                                       start_step = training_args.predict_lr_start_step)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform_predict/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform_predict.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform_predict"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform_predict/iteration_{}/deform_predict.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform_predict":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            

class Deform_Predict_ver_3:
    def __init__(self, is_blender=False, is_6dof=False):
        self.deform = Deform_Predict_ver3(is_blender=is_blender, is_6dof=is_6dof).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, time_input,time_input_before,time_input_after,d_xyz_before, d_xyz_after):
        return self.deform(time_input,time_input_before,time_input_after,d_xyz_before, d_xyz_after)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform_predict"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init_predict * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final_predict,
                                                       lr_delay_mult=training_args.position_lr_delay_mult_predict,
                                                       max_steps=training_args.predict_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform_predict/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform_predict.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform_predict"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform_predict/iteration_{}/deform_predict.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform_predict":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


## ---------------------------------------------------------- Prediction by Gaussians ---------------------------------------------------------------- ##
class DeformPredict_with_Gaussian_dist:
    def __init__(self, is_blender=False, is_6dof=False):
        self.deform = Deform_Predict_with_Gaussians_distribution().cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, d_xyz_before, d_xyz_after):
        return self.deform(d_xyz_before, d_xyz_after)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform_predict"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init_predict * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final_predict,
                                                       lr_delay_mult=training_args.position_lr_delay_mult_predict,
                                                       max_steps=training_args.predict_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform_predict/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform_predict.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform_predict"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform_predict/iteration_{}/deform_predict.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform_predict":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

