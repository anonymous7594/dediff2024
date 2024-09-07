import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3
from utils.loss_utils import kl_divergence_loss, kl_divergence, contrastive_loss, l1_loss, l2_loss, cos_loss
import os
# If using ViT Transformer
#import timm
#import torchvision.transforms as transforms
# If using Stable Diffusion
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionXLImg2ImgPipeline
import math


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=4, W=32, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False):  #D=8,W=256
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10 #t_multires?
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out))

            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender
        self.is_6dof = is_6dof

        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling

    

#### ---------------------------------------------------------- Prediction Model ---------------------------------------------------------- ####
## Image Pre-processing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

## Time embedding by Linear
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(1, d_model, bias=False)
        #self.embed = nn.Embedding(d_inp,d_model).to(self.args.device)    

    def forward(self, x):
        #x = self.elu(x)
        #x = np.concatenate((x,x_vab),axis=2)
        return self.embed(x)


'''    
 Feature Enhancement
'''
class CameraPoseEmbedding(nn.Module):
    def __init__(self, input_dim=37, output_dim=1280):
        super(CameraPoseEmbedding, self).__init__()
        self.fc_prompt = nn.Linear(input_dim, output_dim)
        self.fc_pooled_prompt = nn.Linear(input_dim, output_dim)

    def forward(self, camera_center, world_view_transform, full_proj_transform, fovx, fovy):
        ## Image Prompt Embedding
        # Flatten the [4, 4] tensors to [16]
        world_view_flat = world_view_transform.reshape(-1)  # [16]
        proj_transform_flat = full_proj_transform.reshape(-1)  # [16]

        # Concatenate all the components
        pose_vector = torch.cat([
            camera_center,              # [3]
            world_view_flat,            # [16]
            proj_transform_flat,        # [16]
            fovx.view(-1),              # [1]
            fovy.view(-1)               # [1]
        ], dim=-1)  # [37]

        # Map to the desired size [1280]
        embedded_pose = self.fc_prompt(pose_vector)  # [1280]

        # Repeat or reshape to match [1, 77, 1280]
        embedded_pose = embedded_pose.unsqueeze(0).repeat(77, 1)  # [77, 1280]

        # Add batch dimension [1, 77, 1280]
        embedded_pose = embedded_pose.unsqueeze(0)  # [1, 77, 1280]

        ## Pooled Image Prompt Embedding
        # Flatten the [4, 4] tensors to [16]
        world_view_flat = world_view_transform.reshape(-1)  # [16]
        proj_transform_flat = full_proj_transform.reshape(-1)  # [16]

        # Convert FoVx and FoVy to tensors if they are floats
        #fovx_tensor = torch.tensor([fovx], dtype=torch.float32)
        #fovy_tensor = torch.tensor([fovy], dtype=torch.float32)

        # Concatenate all components into a single vector
        pose_vector = torch.cat([
            camera_center,              # [3]
            world_view_flat,            # [16]
            proj_transform_flat,        # [16]
            fovx.view(-1),                # [1]
            fovy.view(-1)                 # [1]
        ], dim=-1)  # [37]

        # Map to the desired size [1280]
        embedded_pose_pooled = self.fc_pooled_prompt(pose_vector)  # [1280]

        # Add batch dimension [1, 1280]
        embedded_pose_pooled = embedded_pose_pooled.unsqueeze(0)  # [1, 1280]

        return embedded_pose, embedded_pose_pooled

class DiffusionModelLatent(nn.Module):
    def __init__(self, input_size=768, output_size=1280): ###-------------------------------------------------------------------------------------------------------------------> Manually Input
        super(DiffusionModelLatent, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # Input layers
        ### Diffusion Model
        ## Foundation Model
        self.model_embedding = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.pipe_dm = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0").to(device) #, torch_dtype=torch.float16
        self.pipe_dm.set_progress_bar_config(disable=True)
        ## Input processing for 2 conditioning imges
        #self.linear_layer = torch.nn.Linear(self.input_image_bedding, self.output_image_embedding).to(device)
        #self.linear_pooled_prompt = torch.nn.Linear(self.input_image_bedding*2, self.output_image_embedding).to(device)
        ## Input processing for 1 conditioning imge
        self.linear_layer = nn.Linear(768, 1280).to(device)
        ## final_pooled_image_embedding
        self.pooled_linear_layer = nn.Linear(768, 1280).to(device)

        ### Additonal Embedding
        self.temporal_embed = nn.Linear(1, output_size, bias=False).to(device)
        self.camera_pose_embed = CameraPoseEmbedding(input_dim=37, output_dim=1280) ###-------------------------------------------------------------------------------------------> Manually Input



    def forward(self, features_before, features_after, time_dim #):
                ,viewpoint_cam):
    #def forward(self, features_before, features_after,time_start,time_end,num_frames):
        features_before = features_before.to(device)
        features_after = features_after.to(device)
        time_dim = time_dim.to(device)
        viewpoint_cam = viewpoint_cam.to(device) 
        ### IF USING STABLE DIFFUSION
        # Process the image
        #image_ref_1 = preprocess(features_before).unsqueeze(0)
        image_ref_2 = preprocess(features_after).unsqueeze(0)
        with torch.no_grad():
            # Image 1
            #output_1 = self.model_embedding.vision_model(image_ref_1) #(**input_1)
            #image_embedding_1 = output_1.last_hidden_state  # Token-level embeddings
            #pooled_image_embed_1 = output_1.pooler_output  # Pooled embedding
            # Image 2
            output_2 = self.model_embedding.vision_model(image_ref_2) #(**input_2)
            image_embedding_2 = output_2.last_hidden_state  # Token-level embeddings
            pooled_image_embed_2 = output_2.pooler_output  # Pooled embedding

        ## If conditioning on 2 images
        # Convert Embedding Size
        #image_embedding = torch.cat([image_embedding_1,image_embedding_2],dim=1)
        #print(image_embedding.shape)
        #interpolated_tensor = F.interpolate(image_embedding.permute(0, 2, 1), size=77, mode='linear', align_corners=True).permute(0, 2, 1)  # Shape: [1, 77, 768]
        #print(interpolated_tensor.shape)
        # Step 3: Linear layer to transform for image prompt embedding
        #final_image_embedding = self.linear_layer(interpolated_tensor)  # Shape: [1, 77, 1280]
        # Check the shape of the final tensor
        #print(final_image_embedding.shape)  # Output should be torch.Size([1, 77, 1280])
        #pooled_image_embedding = torch.cat([pooled_image_embed_1,pooled_image_embed_2],dim=1)
        # Linear layer to transform for pooled image prompt embedding
        #final_pooled_image_embedding = self.linear_pooled_prompt(pooled_image_embedding)
        #print(final_pooled_image_embedding.shape)

        ## If conditioning on future image only
            final_image_embedding = torch.nn.functional.pad(image_embedding_2, (0, 0, 0, 27))  
        # Apply the linear layer to transform the feature dimension
        final_image_embedding = self.linear_layer(final_image_embedding)
        final_pooled_image_embedding = self.pooled_linear_layer(pooled_image_embed_2)
        # Temporal embedding
        pooled_time_embedding = self.temporal_embed(time_dim) #for pooled image embedding with 2 dims
        time_embedding = pooled_time_embedding.unsqueeze(0) #for image embedding with 3 
        
        # Camera Pose embedding
        camera_center = viewpoint_cam.camera_center
        world_view_transform = viewpoint_cam.world_view_transform
        full_proj_transform = viewpoint_cam.full_proj_transform
        fovx = torch.tensor(math.tan(viewpoint_cam.FoVx * 0.5),device=device)
        fovy = torch.tensor(math.tan(viewpoint_cam.FoVy * 0.5),device=device)
        camera_pose_embedding, pooled_camera_pose_embedding = self.camera_pose_embed(camera_center, world_view_transform, full_proj_transform, fovx, fovy)
        
        ## Apply Diffusion 
        #with torch.no_grad():
            #image_latent_data = self.pipe_dm(prompt_embeds = final_image_embedding, pooled_prompt_embeds = final_pooled_image_embedding,  
            #                             image=features_before, output_type="latent",num_inference_steps=50).images[0] #currently using Preceeding Image
        image_latent_data = self.pipe_dm(prompt_embeds = final_image_embedding + time_embedding + camera_pose_embedding, #+ camera_pose_embedding, 
                                             pooled_prompt_embeds = final_pooled_image_embedding + pooled_time_embedding + pooled_camera_pose_embedding, #+ pooled_camera_pose_embedding,  
                                         image=features_before, 
                                         output_type="latent",
                                         num_inference_steps=50).images[0] #currently using Preceeding Image
            
        return image_latent_data
            
## Motion Prediction
class Deform_Predict(nn.Module):
    def __init__(self, D=4, W=32, Dd=4, Wd=32, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False
                 ,embed_dim = 128, num_heads = 8 # 3D Self-Attention, embed_dim = latent_dim 
                 ,input_channels = 3, hidden_dim = 256, latent_dim = 128, output_channels = 3 # VAE
                 ,input_image_bedding=768, output_image_embedding=1280, Wi = 32, Ddi = 4 # MLP to decode latent compilation from Diffusion Model
                 ):
        super(Deform_Predict, self).__init__()
        # Input layers
        self.D = D
        self.W = W
        # Output layers
        self.Dd = Dd
        self.Wd = Wd
        # Skip layers
        self.skips = [D // 2]
        self.skips_d = [Dd // 2]
        self.t_multires = 6 if is_blender else 5
        #self.input_ch = input_ch
        ## Stable Diffusion
        #self.input_image_bedding = input_image_bedding
        #self.output_image_embedding = output_image_embedding
        self.Wi = Wi
        self.Ddi = Ddi

        ## Static-Dynamic Masking
        self.linear_filter = nn.Linear(3,1)

        # Current Input
        # time embedding
        self.embed_time_fn_before, time_input_ch_before = get_embedder(self.t_multires, 1)
        self.embed_time_fn_after, time_input_ch_after = get_embedder(self.t_multires, 1)
        # x_embedding
        self.embed_fn_before, xyz_input_ch_before = get_embedder(multires, 3)
        self.embed_fn_after, xyz_input_ch_after = get_embedder(multires, 3)
        # Number of Input
        self.input_ch_before = time_input_ch_before + xyz_input_ch_before
        self.input_ch_after = time_input_ch_after + xyz_input_ch_after
        self.embed_time_fn_current, time_input_ch_current = get_embedder(self.t_multires, 1)
        # Deform()
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch_current

        
        ## Deform Network
        self.linear_deform = nn.ModuleList(
                [nn.Linear(self.input_ch, Wd)] + [
                    nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + self.input_ch, Wd)
                    for i in range(Dd - 1)]
            )
        '''
        self.norms_linear_deform = nn.ModuleList(
            [nn.BatchNorm1d(Wd) for _ in range(Dd)]
        )
        '''
        
        ## ENCODING
        # pre-determined
        # before
        self.linear_dxyz_past = nn.ModuleList(
                [nn.Linear(self.input_ch_before, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch_before, W)
                    for i in range(D - 1)]
            )
        '''
        self.norms_dxyz_past = nn.ModuleList(
            [nn.BatchNorm1d(W) for _ in range(D)]
        )
        '''
        # after
        self.linear_dxyz_future = nn.ModuleList(
                [nn.Linear(self.input_ch_after, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch_after, W)
                    for i in range(D - 1)]
            )
        '''
        self.norms_dxyz_future = nn.ModuleList(
            [nn.BatchNorm1d(W) for _ in range(D)]
        )
        '''
        ## DECODING
        # dxyz
        self.linear_dxyz = nn.ModuleList(
                [nn.Linear(W*2 + time_input_ch_current, Wd)] + [
                    nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + W*2 + time_input_ch_current, Wd)
                    for i in range(Dd - 1)])
        '''
        self.norms_linear_dxyz = nn.ModuleList(
            [nn.BatchNorm1d(Wd) for _ in range(Dd)]
        )
        '''

        ## Image Embedding - MLP - Single Latent
        '''
        self.linear_image = nn.ModuleList(
                [nn.Linear(self.Wi + time_input_ch_current, Wd)] + [
                    nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + self.Wi + time_input_ch_current, Wd)
                    for i in range(self.Ddi - 1)])
        '''
        '''
        self.norms_linear_image = nn.ModuleList(
            [nn.BatchNorm1d(Wd) for _ in range(Ddi)]
        )
        '''
        self.linear_image = nn.ModuleList(
                [nn.Linear(self.Wi, Wd)] + [
                    nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + self.Wi, Wd)
                    for i in range(self.Ddi - 1)])
        # Decode the image latent data
        self.linear_image_decode = nn.ModuleList(
            [nn.Linear(self.Wd, self.Wd)] + [
                nn.Linear(self.Wd, self.Wd) if i not in self.skips_d else nn.Linear(self.Wd + self.Wd, self.Wd)
                for i in range(self.Ddi - 1)
            ]
        )
        # Final layer to map back to the original dimension
        self.output_layer = nn.Linear(self.Wd, self.Wi)

        

        ## Image Embedding - MLP - Multi Latent
        self.decoder_latent_compilation = ConvDecoderSelfAttention(in_channels=4)
        self.linear_image_multi = nn.ModuleList(
                [nn.Linear(self.Wi + time_input_ch_current, Wd)] + [
                    nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + self.Wi + time_input_ch_current, Wd)
                    for i in range(self.Ddi - 1)])
        '''
        self.norms_linear_image_multi = nn.ModuleList(
            [nn.BatchNorm1d(Wd) for _ in range(Ddi)]
        )
        '''
        ### OUTPUT LAYER FOR D-XYZ, D-ROTATION, D-SCALING
        ## IF USING LINEAR LAYER
        # d_xyz
        self.gaussian_warp_final = nn.Linear(3, 3)
        self.gaussian_warp = nn.Linear(Wd, 3)
        self.gaussian_warp_image = nn.Linear(Wd, 3)
        # d_rotation
        self.gaussian_rotation_final = nn.Linear(4, 4)
        self.gaussian_rotation = nn.Linear(Wd, 4)
        self.gaussian_rotation_image = nn.Linear(Wd, 4)
        # d_scaling
        self.gaussian_scaling_final = nn.Linear(3, 3)
        self.gaussian_scaling = nn.Linear(Wd, 3)
        self.gaussian_scaling_image = nn.Linear(Wd, 3)

    def forward(self, time_input, time_input_before, time_input_after, xyz, 
                d_xyz_before, d_xyz_after, new_feature_latent_data, new_feature_latent_data_all):
    #def forward(self, time_input, xyz, new_feature_latent_data):    
        ### Apply Static-Dynamic mask
        combined_mask_threshold = 0.2
        diff_xyz = abs(d_xyz_before - d_xyz_after)
        diff_xyz_linear = self.linear_filter(diff_xyz)
        condition = (torch.sigmoid(diff_xyz_linear) >  combined_mask_threshold).any(dim=1)
        # Split the tensor based on the custom condition
        # Before
        d_xyz_before_dynamic = d_xyz_before[condition].to(device)
        d_xyz_before_static = d_xyz_before[~condition].to(device)
        time_input_before = time_input_before[condition].to(device)
        # After
        d_xyz_after_dynamic = d_xyz_after[condition].to(device)
        d_xyz_after_static = d_xyz_after[~condition].to(device)
        time_input_after = time_input_after[condition].to(device)

        ### EMBEDDING
        ## predetermined
        # before
        time_diff_before_emb = self.embed_time_fn_before(time_input_before)
        d_xyz_before_emb = self.embed_fn_before(d_xyz_before_dynamic)
        d_xyz_before_embed_wt = torch.cat([time_diff_before_emb, d_xyz_before_emb], dim=-1)
        # after
        time_diff_after_emb = self.embed_time_fn_after(time_input_after)
        d_xyz_after_emb = self.embed_fn_after(d_xyz_after_dynamic)
        d_xyz_after_embed_wt = torch.cat([time_diff_after_emb, d_xyz_after_emb], dim=-1)
        # current
        time_input = time_input[condition]
        time_current_embed = self.embed_time_fn_current(time_input)     
        xyz_embed = self.embed_fn(xyz)


        #### Decoding + Add feature enhancement if needed
        if new_feature_latent_data.numel() == 0:
            #### Encoding data from before-after frames
            ### d_xyz before
            d_xyz_before_wt = d_xyz_before_embed_wt
            for i, l in enumerate(self.linear_dxyz_past):
            #for i, (l, norm_layer) in enumerate(zip(self.linear_dxyz_past, self.norm_layers)):
                d_xyz_before_wt = self.linear_dxyz_past[i](d_xyz_before_wt)                
                #d_xyz_before_wt = self.norms_dxyz_past[i](d_xyz_before_wt)
                d_xyz_before_wt = F.relu(d_xyz_before_wt)
                #if i == (self.D//2 - 1) :
                #    d_xyz_before_wt = self.drop_out(d_xyz_before_wt)
                if i in self.skips:
                    d_xyz_before_wt = torch.cat([d_xyz_before_embed_wt, d_xyz_before_wt], -1)
            #print(d_xyz_before_wt.shape)
            ### d_xyz after
            d_xyz_after_wt = d_xyz_after_embed_wt
            for i, l in enumerate(self.linear_dxyz_future):
                d_xyz_after_wt = self.linear_dxyz_future[i](d_xyz_after_wt)
                #d_xyz_after_wt = self.norms_dxyz_future[i](d_xyz_after_wt)
                d_xyz_after_wt = F.relu(d_xyz_after_wt)
                #if i == (self.D//2 - 1) :
                #    d_xyz_after_wt = self.drop_out(d_xyz_after_wt)
                if i in self.skips:
                    d_xyz_after_wt = torch.cat([d_xyz_after_embed_wt, d_xyz_after_wt], -1)

            ### deformation field for current timestep
            combined_input_wt = torch.cat([xyz_embed, time_current_embed], dim=-1)
            for i, l in enumerate(self.linear_deform):
                combined_input_wt = self.linear_deform[i](combined_input_wt)
                #combined_input_wt = self.norms_linear_deform[i](combined_input_wt)
                combined_input_wt = F.relu(combined_input_wt)
                if i in self.skips_d:
                    combined_input_wt = torch.cat([xyz_embed, time_current_embed, combined_input_wt], -1)
            
            ### decoding combination between past and future encoded values
            d_xyz_r = torch.cat([d_xyz_before_wt, d_xyz_after_wt, time_current_embed], dim=-1)
            ### d_xyz convert to output  
            d_xyz_r_wt = d_xyz_r
            for i, l in enumerate(self.linear_dxyz):
                d_xyz_r_wt = self.linear_dxyz[i](d_xyz_r_wt)
                #d_xyz_r_wt = self.norms_linear_dxyz[i](d_xyz_r_wt)
                d_xyz_r_wt = F.relu(d_xyz_r_wt)
                #if i == (self.Dd//2 - 1) :
                #    d_xyz_r_wt = self.drop_out(d_xyz_r_wt)
                if i in self.skips_d:
                    d_xyz_r_wt = torch.cat([d_xyz_r, d_xyz_r_wt], -1) 
            
            combined_input_wt = d_xyz_r_wt + combined_input_wt
                
        else: 
            
            combined_input_wt = torch.cat([xyz_embed, time_current_embed], dim=-1)
            for i, l in enumerate(self.linear_deform):
                combined_input_wt = self.linear_deform[i](combined_input_wt)
                #combined_input_wt = self.norms_linear_deform[i](combined_input_wt)
                combined_input_wt = F.relu(combined_input_wt)
                if i in self.skips_d:
                    combined_input_wt = torch.cat([xyz_embed, time_current_embed, combined_input_wt], -1)
            

            ### UPDATE NEW FEATURE ENHANCEMENT DATA
            #if new_feature_latent_data.ndimension() == 3:
            if new_feature_latent_data_all == []:
                #print('THIS LOOP')
                #### IMAGE DECODER
                ## Convert to d-xyz latent
                # Flatten the tensor to 1D
                flattened_image_latent_data = new_feature_latent_data.flatten()
                flattened_image_latent_data = flattened_image_latent_data.to(device)

                '''
                ## CONCATENATE TO THE INPUT BEFORE DECODER
                # Calculate the number of elements needed for the target shape
                num_elements_needed = d_xyz_before_dynamic.shape[0] * self.W
                # If the flattened tensor has fewer elements than needed, pad with zeros (or replicate)
                if flattened_image_latent_data.numel() < num_elements_needed:
                    repeats = (num_elements_needed // flattened_image_latent_data.numel()) + 1
                    padded_image_latent_data = flattened_image_latent_data.repeat(repeats)[:num_elements_needed]
                else:
                    padded_image_latent_data = flattened_image_latent_data[:num_elements_needed]
                # Reshape to the desired shape [50000, 64]
                reshaped_image_latent_data = padded_image_latent_data.reshape(d_xyz_before_dynamic.shape[0], self.W)
                image_combined_encoded = reshaped_image_latent_data
                '''
                ## CONCATENATE TO THE COMBINED OUTPUT
                # Calculate the number of elements needed for the target shape
                num_elements_needed = d_xyz_before_dynamic.shape[0] * self.Wi
                # If the flattened tensor has fewer elements than needed, pad with zeros (or replicate)
                if flattened_image_latent_data.numel() < num_elements_needed:
                    repeats = (num_elements_needed // flattened_image_latent_data.numel()) + 1
                    padded_image_latent_data = flattened_image_latent_data.repeat(repeats)[:num_elements_needed]
                else:
                    padded_image_latent_data = flattened_image_latent_data[:num_elements_needed]
                # Reshape to the desired shape [50000, 64]
                reshaped_image_latent_data = padded_image_latent_data.reshape(d_xyz_before_dynamic.shape[0], self.Wi)
                reshaped_image_latent_data = reshaped_image_latent_data.to(device)
                #image_combined_wt = reshaped_image_latent_data

                ## Apply Image Embedding Decoder
                image_combined = torch.cat([reshaped_image_latent_data], dim=-1) #torch.cat([reshaped_image_latent_data, time_current_embed], dim=-1)
                ### image convert to output
                image_combined_wt = image_combined
                for i, l in enumerate(self.linear_image):
                    image_combined_wt = self.linear_image[i](image_combined_wt)
                    image_combined_wt = F.relu(image_combined_wt)
                    #    #if i == (self.Dd//2 - 1) :
                    #    #    image_combined_wt = self.drop_out(image_combined_wt)
                    if i in self.skips_d:
                        image_combined_wt = torch.cat([image_combined, image_combined_wt], -1) 
                # Decode the image back to the original latent
                image_decoded =image_combined_wt
                for i, l in enumerate(self.linear_image_decode):
                    image_decoded = self.linear_image_decode[i](image_decoded)
                    image_decoded = F.relu(image_decoded)
                    #    #if i == (self.Dd//2 - 1) :
                    #    #    image_combined_wt = self.drop_out(image_combined_wt)
                    if i in self.skips_d:
                        image_decoded = torch.cat([image_decoded, image_combined_wt], -1) 
                image_decoded = self.output_layer(image_decoded)
                loss_image_latent_encoding = l2_loss(image_decoded, reshaped_image_latent_data) + cos_loss(image_decoded, reshaped_image_latent_data)*0.001
                
                

                #### Encoding data from before-after frames
                ### d_xyz before
                d_xyz_before_wt = d_xyz_before_embed_wt
                for i, l in enumerate(self.linear_dxyz_past):
                #for i, (l, norm_layer) in enumerate(zip(self.linear_dxyz_past, self.norm_layers)):
                    d_xyz_before_wt = self.linear_dxyz_past[i](d_xyz_before_wt)                
                    #d_xyz_before_wt = self.norms_dxyz_past[i](d_xyz_before_wt)
                    d_xyz_before_wt = F.relu(d_xyz_before_wt)
                    #if i == (self.D//2 - 1) :
                    #    d_xyz_before_wt = self.drop_out(d_xyz_before_wt)
                    if i in self.skips:
                        d_xyz_before_wt = torch.cat([d_xyz_before_embed_wt, d_xyz_before_wt], -1)
                #print(d_xyz_before_wt.shape)
                ### d_xyz after
                d_xyz_after_wt = d_xyz_after_embed_wt
                for i, l in enumerate(self.linear_dxyz_future):
                    d_xyz_after_wt = self.linear_dxyz_future[i](d_xyz_after_wt)
                    #d_xyz_after_wt = self.norms_dxyz_future[i](d_xyz_after_wt)
                    d_xyz_after_wt = F.relu(d_xyz_after_wt)
                    #if i == (self.D//2 - 1) :
                    #    d_xyz_after_wt = self.drop_out(d_xyz_after_wt)
                    if i in self.skips:
                        d_xyz_after_wt = torch.cat([d_xyz_after_embed_wt, d_xyz_after_wt], -1)
                        
                d_xyz_r = torch.cat([d_xyz_before_wt, d_xyz_after_wt, time_current_embed], dim=-1)
                #d_xyz_r = torch.cat([d_xyz_before_wt, d_xyz_after_wt, time_current_embed, image_combined_encoded], dim=-1)
                ### d_xyz convert to output
                d_xyz_r_wt = d_xyz_r
                for i, l in enumerate(self.linear_dxyz):
                    d_xyz_r_wt = self.linear_dxyz[i](d_xyz_r_wt)
                    #d_xyz_r_wt = self.norms_linear_dxyz[i](d_xyz_r_wt)
                    d_xyz_r_wt = F.relu(d_xyz_r_wt)
                    #if i == (self.Dd//2 - 1) :
                    #    d_xyz_r_wt = self.drop_out(d_xyz_r_wt)
                    if i in self.skips_d:
                        d_xyz_r_wt = torch.cat([d_xyz_r, d_xyz_r_wt], -1) 

                combined_input_wt = d_xyz_r_wt + combined_input_wt

                ## COMBINED BY ADDITION
                #combined_input_wt = combined_input_wt + image_combined_wt
                #combined_input_wt = torch.cat([combined_input_wt, image_combined_wt], -1) 

            else:
                #print('THAT LOOP')
                #### IMAGE DECODER - Single Image ------------------------------------------------------------------------------------------------------------------
                ## Convert to d-xyz latent
                # Flatten the tensor to 1D
                flattened_image_latent_data = new_feature_latent_data.flatten()
                flattened_image_latent_data = flattened_image_latent_data.to(device)

                '''
                ## CONCATENATE TO THE INPUT BEFORE DECODER
                # Calculate the number of elements needed for the target shape
                num_elements_needed = d_xyz_before_dynamic.shape[0] * self.W
                # If the flattened tensor has fewer elements than needed, pad with zeros (or replicate)
                if flattened_image_latent_data.numel() < num_elements_needed:
                    repeats = (num_elements_needed // flattened_image_latent_data.numel()) + 1
                    padded_image_latent_data = flattened_image_latent_data.repeat(repeats)[:num_elements_needed]
                else:
                    padded_image_latent_data = flattened_image_latent_data[:num_elements_needed]
                # Reshape to the desired shape [50000, 64]
                reshaped_image_latent_data = padded_image_latent_data.reshape(d_xyz_before_dynamic.shape[0], self.W)
                image_combined_encoded = reshaped_image_latent_data
                '''
                ## CONCATENATE TO THE COMBINED OUTPUT
                # Calculate the number of elements needed for the target shape
                num_elements_needed = d_xyz_before_dynamic.shape[0] * self.Wi
                # If the flattened tensor has fewer elements than needed, pad with zeros (or replicate)
                if flattened_image_latent_data.numel() < num_elements_needed:
                    repeats = (num_elements_needed // flattened_image_latent_data.numel()) + 1
                    padded_image_latent_data = flattened_image_latent_data.repeat(repeats)[:num_elements_needed]
                else:
                    padded_image_latent_data = flattened_image_latent_data[:num_elements_needed]
                # Reshape to the desired shape [50000, 64]
                reshaped_image_latent_data = padded_image_latent_data.reshape(d_xyz_before_dynamic.shape[0], self.Wi)
                reshaped_image_latent_data = reshaped_image_latent_data.to(device)
                #image_combined_wt = reshaped_image_latent_data

                ## Apply Image Embedding Decoder
                image_combined = torch.cat([reshaped_image_latent_data], dim=-1) #, time_current_embed
                ### image convert to output
                image_combined_wt = image_combined
                for i, l in enumerate(self.linear_image):
                    image_combined_wt = self.linear_image[i](image_combined_wt)
                    #image_combined_wt = self.norms_linear_image[i](image_combined_wt)
                    image_combined_wt = F.relu(image_combined_wt)
                    #    #if i == (self.Dd//2 - 1) :
                    #    #    image_combined_wt = self.drop_out(image_combined_wt)
                    if i in self.skips_d:
                        image_combined_wt = torch.cat([image_combined, image_combined_wt], -1) 

                # Decode the image back to the original latent
                image_decoded =image_combined_wt
                for i, l in enumerate(self.linear_image_decode):
                    image_decoded = self.linear_image_decode[i](image_decoded)
                    image_decoded = F.relu(image_decoded)
                    #    #if i == (self.Dd//2 - 1) :
                    #    #    image_combined_wt = self.drop_out(image_combined_wt)
                    if i in self.skips_d:
                        image_decoded = torch.cat([image_decoded, image_combined_wt], -1) 
                image_decoded = self.output_layer(image_decoded)
                loss_image_latent_encoding = l2_loss(image_decoded, reshaped_image_latent_data) + cos_loss(image_decoded, reshaped_image_latent_data)*0.001
                
                #### IMAGE DECODER - Multi-Images ------------------------------------------------------------------------------------------------------------------
                ## Apply latent compilation through 3d self-attention and convo3d layers
                new_feature_latent_data_all = new_feature_latent_data_all.to(device)
                new_feature_latent_data_all = self.decoder_latent_compilation(new_feature_latent_data_all) # without time dimension
                # Flatten the tensor to 1D
                flattened_image_latent_data_all = new_feature_latent_data_all.flatten()
                flattened_image_latent_data_all = flattened_image_latent_data_all.to(device)

                ## CONCATENATE TO THE INPUT BEFORE DECODER
                # Calculate the number of elements needed for the target shape
                num_elements_needed = d_xyz_before_dynamic.shape[0] * self.Wi
                # If the flattened tensor has fewer elements than needed, pad with zeros (or replicate)
                if flattened_image_latent_data_all.numel() < num_elements_needed:
                    repeats = (num_elements_needed // flattened_image_latent_data_all.numel()) + 1
                    padded_image_latent_data = flattened_image_latent_data_all.repeat(repeats)[:num_elements_needed]
                else:
                    padded_image_latent_data = flattened_image_latent_data_all[:num_elements_needed]
                # Reshape to the desired shape [50000, 64]
                reshaped_image_latent_data = padded_image_latent_data.reshape(d_xyz_before_dynamic.shape[0], self.Wi)
                reshaped_image_latent_data = reshaped_image_latent_data.to(device)

                ## Apply Image Embedding Decoder
                image_combined = torch.cat([reshaped_image_latent_data, time_current_embed], dim=-1)
                ### image convert to output
                image_combined_wt_all = image_combined
                for i, l in enumerate(self.linear_image_multi):
                    image_combined_wt_all = self.linear_image_multi[i](image_combined_wt_all)
                    #image_combined_wt_all = self.norms_linear_image_multi[i](image_combined_wt_all)
                    image_combined_wt_all = F.relu(image_combined_wt_all)
                    #    #if i == (self.Dd//2 - 1) :
                    #    #    image_combined_wt = self.drop_out(image_combined_wt)
                    if i in self.skips_d:
                        image_combined_wt_all = torch.cat([image_combined, image_combined_wt_all], -1) 
                image_combined_wt = image_combined_wt + image_combined_wt_all

                #### Encoding data from before-after frames
                ### d_xyz before
                d_xyz_before_wt = d_xyz_before_embed_wt
                for i, l in enumerate(self.linear_dxyz_past):
                #for i, (l, norm_layer) in enumerate(zip(self.linear_dxyz_past, self.norm_layers)):
                    d_xyz_before_wt = self.linear_dxyz_past[i](d_xyz_before_wt)                
                    #d_xyz_before_wt = self.norms_dxyz_past[i](d_xyz_before_wt)
                    d_xyz_before_wt = F.relu(d_xyz_before_wt)
                    #if i == (self.D//2 - 1) :
                    #    d_xyz_before_wt = self.drop_out(d_xyz_before_wt)
                    if i in self.skips:
                        d_xyz_before_wt = torch.cat([d_xyz_before_embed_wt, d_xyz_before_wt], -1)
                #print(d_xyz_before_wt.shape)
                ### d_xyz after
                d_xyz_after_wt = d_xyz_after_embed_wt
                for i, l in enumerate(self.linear_dxyz_future):
                    d_xyz_after_wt = self.linear_dxyz_future[i](d_xyz_after_wt)
                    #d_xyz_after_wt = self.norms_dxyz_future[i](d_xyz_after_wt)
                    d_xyz_after_wt = F.relu(d_xyz_after_wt)
                    #if i == (self.D//2 - 1) :
                    #    d_xyz_after_wt = self.drop_out(d_xyz_after_wt)
                    if i in self.skips:
                        d_xyz_after_wt = torch.cat([d_xyz_after_embed_wt, d_xyz_after_wt], -1)
                        
                d_xyz_r = torch.cat([d_xyz_before_wt, d_xyz_after_wt, time_current_embed], dim=-1)
                ### d_xyz convert to output
                d_xyz_r_wt = d_xyz_r
                for i, l in enumerate(self.linear_dxyz):
                    d_xyz_r_wt = self.linear_dxyz[i](d_xyz_r_wt)
                    #d_xyz_r_wt = self.norms_linear_dxyz[i](d_xyz_r_wt)
                    d_xyz_r_wt = F.relu(d_xyz_r_wt)
                    #if i == (self.Dd//2 - 1) :
                    #    d_xyz_r_wt = self.drop_out(d_xyz_r_wt)
                    if i in self.skips_d:
                        d_xyz_r_wt = torch.cat([d_xyz_r, d_xyz_r_wt], -1) 

                combined_input_wt = d_xyz_r_wt + combined_input_wt


                ## COMBINED BY ADDITION
                #combined_input_wt = combined_input_wt + image_combined_wt
                #combined_input_wt = torch.cat([combined_input_wt, image_combined_wt], -1) 
    

        #### d_xyz
        ## dynamic part
        # Linear
        #d_xyz = self.gaussian_warp(combined_input_wt) #+ self.gaussian_warp_image(image_combined_wt)
        #d_xyz = self.gaussian_warp_final(torch.cat([self.gaussian_warp(combined_input_wt),self.gaussian_warp_image(image_combined_wt)], -1))
        d_xyz = self.gaussian_warp_final(self.gaussian_warp(combined_input_wt)+self.gaussian_warp_image(image_combined_wt))
        #### d_rotation
        ## dynamic part
        # Linear
        #d_rotation = self.gaussian_rotation(combined_input_wt) #+ self.gaussian_rotation_image(image_combined_wt) 
        #d_rotation = self.gaussian_rotation_final(torch.cat([self.gaussian_rotation(combined_input_wt),self.gaussian_rotation_image(image_combined_wt)], -1))
        d_rotation = self.gaussian_rotation_final(self.gaussian_rotation(combined_input_wt)+self.gaussian_rotation_image(image_combined_wt))
        #### d_scaling
        ## dynamic 
        # Linear
        #d_scaling = self.gaussian_scaling(combined_input_wt) #+ self.gaussian_scaling_image(image_combined_wt)  
        #d_scaling = self.gaussian_scaling_final(torch.cat([self.gaussian_scaling(combined_input_wt),self.gaussian_scaling_image(image_combined_wt)], -1))
        d_scaling = self.gaussian_scaling_final(self.gaussian_scaling(combined_input_wt)+self.gaussian_scaling_image(image_combined_wt))

        return  d_xyz, d_rotation, d_scaling, loss_image_latent_encoding


#### Compile multiple latent data
class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention3D, self).__init__()
        self.query_conv = nn.Conv3d(in_channels, max(1, in_channels // 8), kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, max(1, in_channels // 8), kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        #print(batch_size)
        #print(C)
        query = self.query_conv(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)  # B, D*H*W, C//8
        key = self.key_conv(x).view(batch_size, -1, D * H * W)  # B, C//8, D*H*W
        value = self.value_conv(x).view(batch_size, -1, D * H * W)  # B, C, D*H*W

        attention = torch.bmm(query, key)  # B, D*H*W, D*H*W
        attention = F.softmax(attention, dim=-1)  # B, D*H*W, D*H*W

        out = torch.bmm(value, attention.permute(0, 2, 1))  # B, C, D*H*W
        out = out.view(batch_size, C, D, H, W)

        out = self.gamma * out + x
        return out

class ConvDecoderSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(ConvDecoderSelfAttention, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(True)
        )
        self.self_attention = SelfAttention3D(in_channels)
        self.final_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 33, 60))
        #self.temporal_embed = nn.Linear(1, 668, bias=False) ### --------------------------------------------------------------> Manually Input
    def forward(self, x):
    #def forward(self, x, t):
        x = x.unsqueeze(2)  # Add an additional dimension for depth: [B, C, 1, H, W]
        x = self.conv_layers(x)
        # Add time dimension before self-attention
        #t = self.temporal_embed(t).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        #x = x + t # Add time dimension
        # ----------------------------------------
        x = self.self_attention(x)
        x = self.final_conv(x)
        x = self.global_pool(x)  # Aggregate along the batch dimension
        return x.mean(dim=0, keepdim=True).squeeze(2).squeeze(0) # Reduce batch dimension to 1 and remove depth dimension




