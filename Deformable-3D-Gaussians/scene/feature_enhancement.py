import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DiffusionModelLatent #, Deform_Predict_ver3
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func

gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) #if not self.args.use_multi_gpu else self.args.devices
device = torch.device('cuda:{}'.format(gpu))
    
class FeatureEnhancement:
    def __init__(self):
        self.dm_latent = DiffusionModelLatent().to(device)
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, features_before, features_after, time_dim, viewpoint_cam): #, d_xyz_current):
        return self.dm_latent(features_before, features_after, time_dim, viewpoint_cam)

    def train_setting(self, training_args):
        #fixed_lr = 0.01
        l = [
            {'params': list(self.dm_latent.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale, #fixed_lr, #
             "name": "diffusion_model_latent"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15) #fixed_lr

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init_fe * self.spatial_lr_scale,
                                                      lr_final=training_args.position_lr_final_fe,
                                                       lr_delay_mult=training_args.position_lr_delay_mult_fe,
                                                      max_steps=training_args.fe_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "dm_latent/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.dm_latent.state_dict(), os.path.join(out_weights_path, 'diffusion_model_latent.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "diffusion_model_latent"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "diffusion_model_latent/iteration_{}/diffusion_model_latent.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "diffusion_model_latent":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            

