import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3
from utils.loss_utils import kl_divergence_loss, kl_divergence, contrastive_loss
import os
# If using ViT Transformer
#import timm
#import torchvision.transforms as transforms
# If using Stable Diffusion
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionXLImg2ImgPipeline



device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    

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
    def __init__(self, D=8, W=128, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False):  #D=8,W=256
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
## Feature Enhancement
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(1, d_model, bias=False)
        #self.embed = nn.Embedding(d_inp,d_model).to(self.args.device)    

    def forward(self, x):
        #x = self.elu(x)
        #x = np.concatenate((x,x_vab),axis=2)
        return self.embed(x)
    
class DiffusionModelLatent(nn.Module):
    def __init__(self, input_size=768, output_size=1280):
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
        self.temporal_embed = nn.Linear(1, output_size, bias=False).to(device)
    def forward(self, features_before, features_after, time_dim):
    #def forward(self, features_before, features_after,time_start,time_end,num_frames):
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
        pooled_time_embedding = self.temporal_embed(time_dim) #for pooled image embedding with 2 dims
        time_embedding = pooled_time_embedding.unsqueeze(0) #for image embedding with 3 dims
        ## Apply Diffusion 
        with torch.no_grad():
            #image_latent_data = self.pipe_dm(prompt_embeds = final_image_embedding, pooled_prompt_embeds = final_pooled_image_embedding,  
            #                             image=features_before, output_type="latent",num_inference_steps=50).images[0] #currently using Preceeding Image
            image_latent_data = self.pipe_dm(prompt_embeds = final_image_embedding + time_embedding, pooled_prompt_embeds = final_pooled_image_embedding + pooled_time_embedding,  
                                         image=features_before, output_type="latent",num_inference_steps=25).images[0] #currently using Preceeding Image
            
        return image_latent_data
            
## Motion Prediction
class Deform_Predict(nn.Module):
    def __init__(self, D=4, W=32, Dd=8, Wd=128, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False
                 ,embed_dim = 128, num_heads = 8 # 3D Self-Attention, embed_dim = latent_dim 
                 ,input_channels = 3, hidden_dim = 256, latent_dim = 128, output_channels = 3 # VAE
                 ,input_image_bedding=768, output_image_embedding=1280, Wi = 64, Ddi = 16 # Diffusion Model
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

        ## Define a linear layer for image projection
        #self.projection_layer_before = nn.Linear(768, self.W) # if using vit_base_patch16_224_dino
        #self.projection_layer_after = nn.Linear(768, self.W) # if using vit_base_patch16_224_dino
        #self.projection_layer = nn.Linear(768, self.W) # if using vit_base_patch16_224_dino
        ## 3D Self-attention
        #self.embed_dim = embed_dim
        #self.num_heads = num_heads 
        #self.self_attention = SelfAttention3D(self.embed_dim, self.num_heads) #input as [1, 197, 768]

        '''
        # pre-determined
        # time embedding
        self.embed_time_fn_before, time_input_ch_before = get_embedder(self.t_multires, 1)
        self.embed_time_fn_after, time_input_ch_after = get_embedder(self.t_multires, 1)
        # x_embedding
        self.embed_fn_before, xyz_input_ch_before = get_embedder(multires, 3)
        self.embed_fn_after, xyz_input_ch_after = get_embedder(multires, 3)
        # Number of Input
        self.input_ch_before = time_input_ch_before + xyz_input_ch_before
        self.input_ch_after = time_input_ch_after + xyz_input_ch_after
        '''
        # Current Input
        self.embed_time_fn_current, time_input_ch_current = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch_current


        ## VAE FOR IMAGE EMBEDDING
        #self.input_channels = input_channels
        #self.hidden_dim = hidden_dim
        #self.latent_dim = latent_dim
        #self.output_channels = output_channels
        #self.num_heads = num_heads 
        #self.VAE = VAE(self.input_channels, self.hidden_dim, self.latent_dim, self.output_channels, self.num_heads)
        #self.projection_layer = nn.Linear(self.latent_dim, self.W)

        '''
        ### ENCODER FOR XYZ
        # Preceeding frame
        self.linear_dxyz_past = nn.ModuleList(
                [nn.Linear(self.input_ch_before, self.W)] + [
                    nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch_before, self.W)
                    for i in range(self.D - 1)]
            )

        # Proceeding frame
        self.linear_dxyz_future = nn.ModuleList(
                [nn.Linear(self.input_ch_after, self.W)] + [
                    nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch_after, self.W)
                    for i in range(self.D - 1)]
            )

        ### DECODER
        ## xyz
        self.linear_dxyz = nn.ModuleList(
                [nn.Linear(time_input_ch_current + self.W*2, self.Wd)] + [ #self.W*2 + 
                    nn.Linear(self.Wd, self.Wd) if i not in self.skips_d else nn.Linear(self.Wd + time_input_ch_current + self.W*2, self.Wd) # self.W*2 + 
                    for i in range(self.Dd - 1)])
                    
        ## Deform Network
        self.linear_deform = nn.ModuleList(
                [nn.Linear(self.input_ch, Wd)] + [
                    nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + self.input_ch, Wd)
                    for i in range(Dd - 1)]
            )
            '''
        ## Image Embedding - MLP
        self.linear_image = nn.ModuleList(
                [nn.Linear(self.Wi + time_input_ch_current, Wd)] + [
                    nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + self.Wi + time_input_ch_current, Wd)
                    for i in range(self.Ddi - 1)])
        ## Combined to get the output - if using MLP
        #self.linear_combined = nn.ModuleList(
        #        [nn.Linear(Wd*2, Wd)] + [
        #            nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + Wd*2, Wd)
        #            for i in range(Dd - 1)])

        ## DECODER BUT COMBINE BEFORE-AFTER (XYZ-IMAGE)
        # before
        #self.linear_combined_before = nn.ModuleList(
        #        [nn.Linear(W*2 + time_input_ch_current, Wd)] + [
        #            nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + W*2 + time_input_ch_current, Wd)
        #            for i in range(Dd - 1)])
        # after
        #self.linear_combined_after = nn.ModuleList(
        #        [nn.Linear(W*2 + time_input_ch_current, Wd)] + [
        #            nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + W*2 + time_input_ch_current, Wd)
        #            for i in range(Dd - 1)])

        
        ### OUTPUT LAYER FOR COMBINATION
        #self.combined_layer = nn.Linear(Wd*2,Wd*2)

        ### OUTPUT LAYER FOR D-XYZ, D-ROTATION, D-SCALING
        ## IF USING LINEAR LAYER
        # d_xyz
        self.gaussian_warp = nn.Linear(Wd, 3)
        #self.gaussian_warp = nn.Linear(Wd*2, 3)
        # d_rotation
        self.gaussian_rotation = nn.Linear(Wd, 4)
        #self.gaussian_rotation = nn.Linear(Wd*2, 4)
        # d_scaling
        self.gaussian_scaling = nn.Linear(Wd, 3)
        #self.gaussian_scaling = nn.Linear(Wd*2, 3)

        ## IF USING MLPs LAYERS
        # d_xyz
        #self.gaussian_warp = nn.ModuleList([
        #                        nn.Linear(2*Wd, Wd),
        #                        nn.Linear(Wd, Wd - Wd // 4),
        #                        nn.Linear(Wd - Wd // 4, Wd // 2),
        #                        nn.Linear(Wd // 2, Wd // 4),
        #                        nn.Linear(Wd // 4, 3)
        #                    ])
        # d_rotation
        #self.gaussian_rotation = nn.ModuleList([
        #                        nn.Linear(2*Wd, Wd),
        #                        nn.Linear(Wd, Wd - Wd // 4),
        #                        nn.Linear(Wd - Wd // 4, Wd // 2),
        #                        nn.Linear(Wd // 2, Wd // 4),
        #                        nn.Linear(Wd // 4, 4)
        #                    ])
        # d_scaling
        #self.gaussian_scaling = nn.ModuleList([
        #                        nn.Linear(2*Wd, Wd),
        #                        nn.Linear(Wd, Wd - Wd // 4),
        #                        nn.Linear(Wd - Wd // 4, Wd // 2),
        #                        nn.Linear(Wd // 2, Wd // 4),
        #                        nn.Linear(Wd // 4, 3)
        #                    ])      

        #self.drop_out = nn.Dropout(0.2)

        ### Diffusion Model
        ## Foundation Model
        #self.model_embedding = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        #self.pipe_dm = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16).to(device)
        #self.pipe_dm.set_progress_bar_config(disable=True)
        ## Input processing for 2 conditioning imges
        #self.linear_layer = torch.nn.Linear(self.input_image_bedding, self.output_image_embedding).to(device)
        #self.linear_pooled_prompt = torch.nn.Linear(self.input_image_bedding*2, self.output_image_embedding).to(device)
        ## Input processing for 1 conditioning imge
        #self.linear_layer = nn.Linear(768, 1280)
        ## final_pooled_image_embedding
        #self.pooled_linear_layer = nn.Linear(768, 1280)

        ### ViT
        # Pretrained Model ViT
        #pretrained_model = timm.create_model('vit_base_patch16_224_dino', pretrained=True).to(device)
        self.decoder_latent_compilation = ConvDecoderSelfAttention(in_channels=4)


    def forward(self, time_input,time_input_before,time_input_after,d_xyz_before, d_xyz_after, new_feature_latent_data):
    #def forward(self, time_input, xyz, new_feature_latent_data):    

        '''
        ### STATIC-DYNAMIC REGION MASKING
        combined_mask_threshold = 0.3
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
        '''

        # No Masking
        #d_xyz_before_dynamic = d_xyz_before.to(device)
        #d_xyz_after_dynamic = d_xyz_after.to(device)

        '''
        ### EMBEDDING XYZ AND TEMPORAL DIMENSION
        ## predetermined
        # before
        time_diff_before_emb = self.embed_time_fn_before(time_input_before)
        d_xyz_before_emb = self.embed_fn_before(d_xyz_before_dynamic)
        d_xyz_before_embed_wt = torch.cat([time_diff_before_emb, d_xyz_before_emb], dim=-1)
        # after
        time_diff_after_emb = self.embed_time_fn_after(time_input_after)
        d_xyz_after_emb = self.embed_fn_after(d_xyz_after_dynamic)
        d_xyz_after_embed_wt = torch.cat([time_diff_after_emb, d_xyz_after_emb], dim=-1)
        '''
        # current
        #time_input = time_input[condition]
        time_current_embed = self.embed_time_fn_current(time_input)     
        

        ### IMAGE Embedding   
        #print('features_before before shape: ',features_before.size())
        #print('features_before after shape: ',features_before.size())
        ### IF USING VIT
        ## Image Embedding by ViT
        # Before
        #image_before = preprocess(image_before).unsqueeze(0)
        #features_before = self.pretrained_model.forward_features(image_before)
        # After
        #image_after = preprocess(image_after).unsqueeze(0)
        #features_after = pretrained_model.forward_features(image_after)
        # Reshape
        #projected_features_before = features_before.view(-1, 768)
        #projected_features_before = self.projection_layer(projected_features_before)
        #projected_features_after = features_after.view(-1, 768)
        #projected_features_after = self.projection_layer(projected_features_after)
        # Calculate the number of repetitions needed
        #num_repeats = d_xyz_before_dynamic.shape[0] // 197 # if using vit_base_patch16_224_dino
        ## Repeat the tensor
        ## Before
        #expanded_features_before = projected_features_before.repeat(num_repeats + 1, 1)[:100000]  # Shape: [174498, 64]
        ## After
        # Repeat the tensor
        #expanded_features_after = projected_features_after.repeat(num_repeats + 1, 1)[:d_xyz_before_dynamic.shape[0]]  # Shape: [174498, 64]
        # KL Divergence Loss for image embedding
        #embeded_image_loss = kl_divergence_loss(expanded_features_before,expanded_tensor_after)
        #print('embeded_image_loss: ',embeded_image_loss)
        ### IF USING STABLE DIFFUSION
        # Process the image for Prompt embedding
        #image_ref_1 = preprocess(features_before).unsqueeze(0)
        #image_ref_2 = preprocess(features_after).unsqueeze(0)
        #with torch.no_grad():
            # Image 1
            #output_1 = self.model_embedding.vision_model(image_ref_1) #(**input_1)
            #image_embedding_1 = output_1.last_hidden_state  # Token-level embeddings
            #pooled_image_embed_1 = output_1.pooler_output  # Pooled embedding
            # Image 2
            #output_2 = self.model_embedding.vision_model(image_ref_2) #(**input_2)
            #image_embedding_2 = output_2.last_hidden_state  # Token-level embeddings
            #pooled_image_embed_2 = output_2.pooler_output  # Pooled embedding
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
        # If conditioning on future image only
            #final_image_embedding = torch.nn.functional.pad(image_embedding_2, (0, 0, 0, 27))  
        # Apply the linear layer to transform the feature dimension
        #final_image_embedding = self.linear_layer(final_image_embedding)
        #final_pooled_image_embedding = self.pooled_linear_layer(pooled_image_embed_2)

        ## Apply Diffusion 
        #with torch.no_grad():
        #    new_feature_latent_data = self.pipe_dm(prompt_embeds = final_image_embedding, pooled_prompt_embeds = final_pooled_image_embedding,  
        #                                 image=features_before, output_type="latent",num_inference_steps=10).images[0] #currently using Preceeding Image

        '''
        #### ENCODER FOR XYZ
        ### d_xyz before
        d_xyz_before_wt = d_xyz_before_embed_wt
        for i, l in enumerate(self.linear_dxyz_past):
        ##for i, (l, norm_layer) in enumerate(zip(self.linear_dxyz_past, self.norm_layers)):
            d_xyz_before_wt = self.linear_dxyz_past[i](d_xyz_before_wt)                
        #    #d_xyz_before_wt = self.norm_layers_past(d_xyz_before_wt)
            d_xyz_before_wt = F.relu(d_xyz_before_wt)
        #    #if i == (self.D//2 - 1) :
        #    #    d_xyz_before_wt = self.drop_out(d_xyz_before_wt)
            if i in self.skips:
                d_xyz_before_wt = torch.cat([d_xyz_before_embed_wt, d_xyz_before_wt], -1)
        ##print(d_xyz_before_wt.shape)
        ### d_xyz after
        d_xyz_after_wt = d_xyz_after_embed_wt
        for i, l in enumerate(self.linear_dxyz_future):
            d_xyz_after_wt = self.linear_dxyz_future[i](d_xyz_after_wt)
        #    #d_xyz_after_wt = self.norm_layers_future(d_xyz_after_wt)
            d_xyz_after_wt = F.relu(d_xyz_after_wt)
        #    #if i == (self.D//2 - 1) :
        #    #    d_xyz_after_wt = self.drop_out(d_xyz_after_wt)
            if i in self.skips:
                d_xyz_after_wt = torch.cat([d_xyz_after_embed_wt, d_xyz_after_wt], -1)
        # KL Divergence Loss for xyz
        #embeded_xyz_loss = kl_divergence_loss(d_xyz_before_wt,d_xyz_after_wt)
        #print('embeded_xyz_loss: ',embeded_xyz_loss)
        # Total KL Divergence Loss
        #kl_loss = embeded_xyz_loss #+ 0.01*embeded_image_loss
        #print('kl_loss: ',kl_loss)
        #contrastive_loss_result = contrastive_loss(d_xyz_before_wt, d_xyz_after_wt, temperature=0.5)
        '''

        #### APPLY VAE
        ### ENCODER BY VAE
        #features_before = features_before.unsqueeze(0)
        #features_after = features_after.unsqueeze(0)
        #image_embedding_new, before_encoded, after_encoded = self.VAE(features_before, features_after)
        #print(image_new.size())
        #print(before_encoded.size())
        #print(after_encoded.size())
        ### Apply diffusion mode
        ## ADD LATER
        ## Projection Image Embedding
        #before_encoded = self.projection_layer(before_encoded)
        #after_encoded = self.projection_layer(after_encoded)
        #num_repeats = d_xyz_before_wt.shape[0]
        #before_encoded_p = before_encoded.repeat(num_repeats + 1, 1)[:d_xyz_before_wt.shape[0]]
        #after_encoded_p = after_encoded.repeat(num_repeats + 1, 1)[:d_xyz_before_wt.shape[0]]
        ## Multiplication to Encoded xyz
        #d_xyz_before_wt = d_xyz_before_wt + before_encoded_p #torch.cat([d_xyz_before_wt,before_encoded_p], dim=-1)
        #d_xyz_after_wt = d_xyz_after_wt + after_encoded_p #torch.cat([d_xyz_after_wt,after_encoded_p], dim=-1)

        if new_feature_latent_data.numel() == 0:
            '''
            #### XYZ DECODER
            ### Combine before-after d_xyz  
            d_xyz_before_wt = d_xyz_before_wt
            d_xyz_after_wt = d_xyz_after_wt                                               
            d_xyz_r = torch.cat([d_xyz_before_wt, d_xyz_after_wt, time_current_embed], dim=-1)
            ### d_xyz convert to output
            d_xyz_r_wt = d_xyz_r
            for i, l in enumerate(self.linear_dxyz):
                d_xyz_r_wt = self.linear_dxyz[i](d_xyz_r_wt)
            #    #d_xyz_r_wt = self.norm_layers_combined(d_xyz_r_wt)
                d_xyz_r_wt = F.relu(d_xyz_r_wt)
            #    #if i == (self.Dd//2 - 1) :
            #    #    d_xyz_r_wt = self.drop_out(d_xyz_r_wt)
                if i in self.skips_d:
                    d_xyz_r_wt = torch.cat([d_xyz_r, d_xyz_r_wt], -1) 
            combined_input_wt = d_xyz_r_wt
            #print('PREDICT WITHOUT LATENT DATA ....................... SUCCESSFULLY')
            #print(d_xyz_r_wt.size())
            '''
            pass
        else: 
            '''
            #### XYZ DECODER
            ### Combine before-after d_xyz  
            d_xyz_before_wt = d_xyz_before_wt
            d_xyz_after_wt = d_xyz_after_wt                                               
            d_xyz_r = torch.cat([d_xyz_before_wt, d_xyz_after_wt, time_current_embed], dim=-1)
            ### d_xyz convert to output
            d_xyz_r_wt = d_xyz_r
            for i, l in enumerate(self.linear_dxyz):
                d_xyz_r_wt = self.linear_dxyz[i](d_xyz_r_wt)
                #    #d_xyz_r_wt = self.norm_layers_combined(d_xyz_r_wt)
                d_xyz_r_wt = F.relu(d_xyz_r_wt)
                #    #if i == (self.Dd//2 - 1) :
                #    #    d_xyz_r_wt = self.drop_out(d_xyz_r_wt)
                if i in self.skips_d:
                    d_xyz_r_wt = torch.cat([d_xyz_r, d_xyz_r_wt], -1) 
            combined_input_wt = d_xyz_r_wt
                #print(d_xyz_r_wt.size())
            '''
            ### UPDATE NEW FEATURE ENHANCEMENT DATA
            if new_feature_latent_data.ndimension() == 3:
                #### IMAGE DECODER
                ## Convert to d-xyz latent
                # Flatten the tensor to 1D
                flattened_image_latent_data = new_feature_latent_data.flatten()
                # Calculate the number of elements needed for the target shape
                num_elements_needed = d_xyz_before.shape[0] * self.Wi
                # If the flattened tensor has fewer elements than needed, pad with zeros (or replicate)
                if flattened_image_latent_data.numel() < num_elements_needed:
                    repeats = (num_elements_needed // flattened_image_latent_data.numel()) + 1
                    padded_image_latent_data = flattened_image_latent_data.repeat(repeats)[:num_elements_needed]
                else:
                    padded_image_latent_data = flattened_image_latent_data[:num_elements_needed]
                # Reshape to the desired shape [50000, 64]
                reshaped_image_latent_data = padded_image_latent_data.reshape(d_xyz_before.shape[0], self.Wi)
                ## Apply Image Embedding Decoder
                image_combined = torch.cat([reshaped_image_latent_data, time_current_embed], dim=-1)
                ### image convert to output
                image_combined_wt = image_combined
                for i, l in enumerate(self.linear_image):
                    image_combined_wt = self.linear_image[i](image_combined_wt)
                    image_combined_wt = F.relu(image_combined_wt)
                    #    #if i == (self.Dd//2 - 1) :
                    #    #    image_combined_wt = self.drop_out(image_combined_wt)
                    if i in self.skips_d:
                        image_combined_wt = torch.cat([image_combined, image_combined_wt], -1) 
                
                ## COMBINED BY ADDITION
                combined_input_wt = image_combined_wt #combined_input_wt + image_combined_wt
                ## COMBINED BY CONCATENATION
                # combined_input_wt = torch.cat([combined_input_wt, image_combined_wt], dim=-1)
                # combined_input_wt = self.combined_layer(combined_input_wt) 
                    #print('TEST ..................... SUCCESFUL')
                    #combined_input_wt = combined_input_wt + reshaped_image_latent_data
            else:
                #### IMAGE DECODER
                # Apply the model to the input tensor
                #print(new_feature_latent_data.size())
                new_feature_latent_data = self.decoder_latent_compilation(new_feature_latent_data) # without time dimension
                #print('NOT ERROR HERE')
                #print(new_feature_latent_data.size())
                #new_feature_latent_data = self.decoder_latent_compilation(new_feature_latent_data, time_input[0,:]) # with addition of time dimension
                #print('EXECUTED.......................... SUCCESSFULLY')
                ## Convert to d-xyz latent
                # Flatten the tensor to 1D
                flattened_image_latent_data = new_feature_latent_data.flatten()
                # Calculate the number of elements needed for the target shape
                num_elements_needed = d_xyz_before.shape[0] * self.Wi
                # If the flattened tensor has fewer elements than needed, pad with zeros (or replicate)
                if flattened_image_latent_data.numel() < num_elements_needed:
                    repeats = (num_elements_needed // flattened_image_latent_data.numel()) + 1
                    padded_image_latent_data = flattened_image_latent_data.repeat(repeats)[:num_elements_needed]
                else:
                    padded_image_latent_data = flattened_image_latent_data[:num_elements_needed]
                # Reshape to the desired shape [50000, 64]
                reshaped_image_latent_data = padded_image_latent_data.reshape(d_xyz_before.shape[0], self.Wi)
                ## Apply Image Embedding Decoder
                image_combined = torch.cat([reshaped_image_latent_data, time_current_embed], dim=-1)
                ### image convert to output
                image_combined_wt = image_combined
                for i, l in enumerate(self.linear_image):
                    image_combined_wt = self.linear_image[i](image_combined_wt)
                    image_combined_wt = F.relu(image_combined_wt)
                    #    #if i == (self.Dd//2 - 1) :
                    #    #    image_combined_wt = self.drop_out(image_combined_wt)
                    if i in self.skips_d:
                        image_combined_wt = torch.cat([image_combined, image_combined_wt], -1) 
                
                ## COMBINED BY ADDITION
                combined_input_wt =  image_combined_wt #combined_input_wt + image_combined_wt
                ## COMBINED BY CONCATENATION
                # combined_input_wt = torch.cat([combined_input_wt, image_combined_wt], dim=-1)
                # combined_input_wt = self.combined_layer(combined_input_wt) 
                    #print('TEST ..................... SUCCESFUL')
                    #combined_input_wt = combined_input_wt + reshaped_image_latent_data

            ##### XYZ AND IMAGE COMBINED FOR DECODER
            ### combine before xyz and image
            #combined_before = torch.cat([d_xyz_before_wt, expanded_features_before, time_current_embed], dim=-1)
            #combined_before_wt = combined_before
            #for i, l in enumerate(self.linear_combined_before):
            #    combined_before_wt = self.linear_combined_before[i](combined_before_wt)
            #    combined_before_wt = F.relu(combined_before_wt)
            #    #if i == (self.Dd//2 - 1) :
            #    #    image_combined_wt = self.drop_out(image_combined_wt)
            #    if i in self.skips_d:
            #        combined_before_wt = torch.cat([combined_before, combined_before_wt], -1)
            ### combine after xyz and image
            #combined_after = torch.cat([d_xyz_after_wt, expanded_features_after, time_current_embed], dim=-1)
            #combined_after_wt = combined_before
            #for i, l in enumerate(self.linear_combined_after):
            #    combined_after_wt = self.linear_combined_after[i](combined_after_wt)
            #    combined_after_wt = F.relu(combined_after_wt)
            #    #if i == (self.Dd//2 - 1) :
            #    #    image_combined_wt = self.drop_out(image_combined_wt)
            #    if i in self.skips_d:
            #        combined_after_wt = torch.cat([combined_after, combined_after_wt], -1) 

        ### APPLY MLPs LAYER FOR OUTPUT
        #combined_input_wt = combined_input
        #for i, l in enumerate(self.linear_combined):
        #    combined_input_wt = self.linear_combined[i](combined_input_wt)
        #    combined_input_wt = F.relu(combined_input_wt)
        #    if i in self.skips_d:
        #        combined_input_wt = torch.cat([combined_input, combined_input_wt], -1) 
        
        #if output by linear
        #combined_input_wt = d_xyz_r_wt + image_combined_wt #+ combined_before_wt + combined_after_wt
        #if output by MLP
        #combined_input_wt = torch.cat([d_xyz_r_wt, image_combined_wt], -1) 
    
        #### d_xyz
        ## dynamic part
        # Linear
        d_xyz = self.gaussian_warp(combined_input_wt)
        # MLP
        #d_xyz = combined_input_wt.clone()
        #for i, l in enumerate(self.gaussian_warp):
        #    d_xyz = self.gaussian_warp[i](d_xyz)
        #    d_xyz = F.relu(d_xyz)
        ## static part
        #dxyz_remaining = torch.zeros_like(d_xyz_before_static).to(device)
        ## combine value
        #d_xyz_combined = torch.cat((d_xyz, dxyz_remaining), dim=0)

        #### d_rotation
        ## dynamic part
        # Linear
        d_rotation = self.gaussian_rotation(combined_input_wt)      
        # MLP
        #d_rotation = combined_input_wt.clone()
        #for i, l in enumerate(self.gaussian_rotation):
        #    d_rotation = self.gaussian_rotation[i](d_rotation)
        #    d_rotation = F.relu(d_rotation)
        ## static part   #
        #drotation_remaining = torch.zeros((d_xyz_before_static.size(0), 4)).to(device)
        ## combine value
        #d_rotation_combined = torch.cat((d_rotation, drotation_remaining), dim=0)

        #### d_scaling
        ## dynamic 
        # Linear
        d_scaling = self.gaussian_scaling(combined_input_wt)
        # MLP
        #d_scaling = combined_input_wt.clone()
        #for i, l in enumerate(self.gaussian_scaling):
        #    d_scaling = self.gaussian_scaling[i](d_scaling)
        #    d_scaling = F.relu(d_scaling)
        ## static part
        #dscaling_remaining = torch.zeros_like(d_xyz_before_static).to(device)
        ## combine value
        #d_scaling_combined = torch.cat((d_scaling, dscaling_remaining), dim=0)

        return  d_xyz, d_rotation, d_scaling #d_xyz_combined, d_rotation_combined, d_scaling_combined #d_xyz, d_rotation, d_scaling, kl_loss, contrastive_loss

## Self-Attention for Image Embedding
#class SelfAttention(nn.Module):
#    def __init__(self, embed_dim, num_heads):
#        super(SelfAttention, self).__init__()
#        self.num_heads = num_heads
#        self.embed_dim = embed_dim
#        self.head_dim = embed_dim // num_heads
        
#        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
#        self.query = nn.Linear(embed_dim, embed_dim)
#        self.key = nn.Linear(embed_dim, embed_dim)
#        self.value = nn.Linear(embed_dim, embed_dim)
#        self.fc_out = nn.Linear(embed_dim, embed_dim)
        
#        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

#    def forward(self, x):
#        N, _, _ = x.shape
        
#        # Ensure input is on the correct device
#        x = x.to(device)
        
#        Q = self.query(x)
#        K = self.key(x)
#        V = self.value(x)
        
#        Q = Q.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#        K = K.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#        V = V.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
#        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
#        attention = torch.softmax(energy, dim=-1)
        
#        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
#        out = out.view(N, -1, self.embed_dim)
        
#        out = self.fc_out(out)
#        return out

#class FeedForward(nn.Module):
#    def __init__(self, input_dim, output_dim):
#        super(FeedForward, self).__init__()
#        self.fc1 = nn.Linear(input_dim, 1024)
#        self.fc2 = nn.Linear(1024, 512)
#        self.fc3 = nn.Linear(512, 256)
#        self.fc4 = nn.Linear(256, output_dim)

#    def forward(self, x):
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = F.relu(self.fc3(x))
#        x = self.fc4(x)
#        return x

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




## Self Attention
#class SelfAttention3D_ver_1(nn.Module):
#    def __init__(self, latent_dim, num_heads):
#        super(SelfAttention3D, self).__init__()
#        self.num_heads = num_heads
#        self.head_dim = latent_dim // num_heads
#        assert self.head_dim * num_heads == latent_dim, "latent_dim must be divisible by num_heads"
        
#        self.query = nn.Linear(latent_dim, latent_dim)
#        self.key = nn.Linear(latent_dim, latent_dim)
#        self.value = nn.Linear(latent_dim, latent_dim)
#        self.fc_out = nn.Linear(latent_dim, latent_dim)
        
#        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

#    def forward(self, x):
#        batch_size = x.shape[0]
        
        # x shape: [batch_size, 2, latent_dim]
        # Split into query, key, value projections
#        Q = self.query(x)  # [batch_size, 2, latent_dim]
#        K = self.key(x)  # [batch_size, 2, latent_dim]
#        V = self.value(x)  # [batch_size, 2, latent_dim]
        
        # Reshape for multi-head attention
#        Q = Q.view(batch_size, 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, 2, head_dim]
#        K = K.view(batch_size, 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, 2, head_dim]
#        V = V.view(batch_size, 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, 2, head_dim]
        
        # Compute attention scores
#        energy = torch.einsum("bnqd,bnkd->bnqk", [Q, K]) / self.scale  # [batch_size, num_heads, 2, 2]
#        attention = torch.softmax(energy, dim=-1)  # [batch_size, num_heads, 2, 2]
        
        # Compute the attended values
#        out = torch.einsum("bnqk,bnvd->bnqd", [attention, V])  # [batch_size, num_heads, 2, head_dim]
        
        # Reshape and combine heads
#        out = out.permute(0, 2, 1, 3).contiguous()  # [batch_size, 2, num_heads, head_dim]
#        out = out.view(batch_size, 2, self.num_heads * self.head_dim)  # [batch_size, 2, latent_dim]
        
        # Final linear layer
#        out = self.fc_out(out)  # [batch_size, 2, latent_dim]
        
#        return out

#class Encoder(nn.Module):
#    def __init__(self, input_channels, hidden_dim, latent_dim):
#        super(Encoder, self).__init__()
#        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1)
#        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1)
#        self.conv3 = nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=4, stride=2, padding=1)
#        self.pool = nn.AdaptiveAvgPool2d((4, 4))
#        self.fc_mu = nn.Linear(hidden_dim*4 * 4 * 4, latent_dim)
#        self.fc_logvar = nn.Linear(hidden_dim*4 * 4 * 4, latent_dim)
    
#    def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = F.relu(self.conv2(x))
#        x = F.relu(self.conv3(x))
#        x = self.pool(x)
#        x = x.view(x.size(0), -1)
#        mu = self.fc_mu(x)
#       logvar = self.fc_logvar(x)
#        return mu, logvar

#class Decoder(nn.Module):
#    def __init__(self, latent_dim, hidden_dim, output_channels):
#        super(Decoder, self).__init__()
#        self.fc = nn.Linear(2 * latent_dim, latent_dim * 16 * 16)  # Flatten the input and expand
#        self.deconv1 = nn.ConvTranspose2d(latent_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1)  # Output: [8, 64, 32, 32]
#        self.deconv2 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=4, stride=2, padding=1)   # Output: [8, 32, 64, 64]
#        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1)   # Output: [8, 16, 128, 128]
#        self.deconv4 = nn.ConvTranspose2d(hidden_dim // 2, output_channels, kernel_size=1)                        # Output: [8, 3, 128, 128]
#    def forward(self, x,z_w,z_h):
#        x = x.view(x.size(0), -1)  # Flatten: [8, 2, 128] -> [8, 2 * 128]
#        x = F.relu(self.fc(x))  # Fully connected: [8, 2 * 128] -> [8, 128 * 16 * 16]
#        x = x.view(x.size(0), 128, 16, 16)  # Reshape to 2D: [8, 128 * 16 * 16] -> [8, 128, 16, 16]
#        x = F.relu(self.deconv1(x))  # Transposed conv: [8, 128, 16, 16] -> [8, 64, 32, 32]
#        x = F.relu(self.deconv2(x))  # Transposed conv: [8, 64, 32, 32] -> [8, 32, 64, 64]
#        x = F.relu(self.deconv3(x))  # Transposed conv: [8, 32, 64, 64] -> [8, 16, 128, 128]
#        x = torch.sigmoid(self.deconv4(x))  # Transposed conv: [8, 16, 128, 128] -> [8, 3, 128, 128]
#        x = F.interpolate(x, size=(z_w, z_h), mode='bilinear', align_corners=False)
#        return x

#class VAE(nn.Module):
#    def __init__(self, input_channels, hidden_dim, latent_dim, output_channels, num_heads):
#        super(VAE, self).__init__()
#        self.encoder_before = Encoder(input_channels, hidden_dim, latent_dim)
#        self.encoder_after = Encoder(input_channels, hidden_dim, latent_dim)
#        self.decoder = Decoder(latent_dim, hidden_dim, output_channels)
#        self.SelfAttention3D = SelfAttention3D(latent_dim, num_heads)   
#    def reparameterize(self, mu, logvar):
#        std = torch.exp(0.5 * logvar)
#        eps = torch.randn_like(std)
#        return mu + eps * std   
#    def forward(self, x_before, x_after):
#        z_w = x_before.size()[2]
#        z_h = x_before.size()[3]
#        mu_before, logvar_before = self.encoder_before(x_before)
#        z_before = self.reparameterize(mu_before, logvar_before) # -----> [batch size, latent dim]
#        mu_after, logvar_after = self.encoder_after(x_after)
#        z_after = self.reparameterize(mu_after, logvar_after) # -----> [batch size, latent dim]
        # add frame dim
#        z_before_t = z_before.unsqueeze(1)
#        z_after_t = z_after.unsqueeze(1)
        #print(z_before.shape)
        # concat
#        z = torch.cat([z_before_t,z_after_t],dim=1)
#        z = self.SelfAttention3D(z)
        #print(z.size())
        #print(z.size())
        #return self.decoder(z,z_w,z_h), z_before, z_after # return image 
#        return z, z_before, z_after # return data only

## --------------------------------------------------------------------------------------------

## Ver 3
#class Deform_Predict_ver_with_Probability_Distribution(nn.Module):
#    def __init__(self, grid_size=16, hidden_dim=2, t_multires=5, num_layers=4, num_components=5, multires=10, is_blender=False, is_6dof=False):
#        super(Deform_Predict, self).__init__()
#        self.grid_size = grid_size
#        self.t_multires = t_multires
#        self.num_layers = num_layers
#        self.hidden_dim = hidden_dim
#        self.num_components = num_components

        # pre-determined
        # time embedding
#        self.embed_time_fn_before, time_input_ch_before = get_embedder(self.t_multires, 1)
#        self.embed_time_fn_after, time_input_ch_after = get_embedder(self.t_multires, 1)
        # x_embedding
#        self.embed_fn_before, xyz_input_ch_before = get_embedder(multires, 3)
#        self.embed_fn_after, xyz_input_ch_after = get_embedder(multires, 3)
        # Current Input
#        self.embed_time_fn_current, time_input_ch_current = get_embedder(self.t_multires, 1)
        # Number of Input
#        self.input_ch_before = time_input_ch_before + xyz_input_ch_before
#        self.input_ch_after = time_input_ch_after + xyz_input_ch_after
        # Model
#        self.CustomModelWithGMM = CustomModelWithGMM(self.input_ch_before, self.input_ch_after, self.grid_size, self.hidden_dim, self.num_layers, self.num_components)
        # d_xyz
#        self.gaussian_warp = nn.Linear(3, 3)
        # d_rotation
#        self.gaussian_rotation = nn.Linear(3, 4)
        # d_scaling
#        self.gaussian_scaling = nn.Linear(3, 3)
#    def forward(self, time_input,time_input_before,time_input_after,d_xyz_before, d_xyz_after):
        ### embedding time
        ## predetermined
        # before
        #time_diff_before_emb = self.embed_time_fn_before(time_input_before)
        #d_xyz_before_emb = self.embed_fn_before(d_xyz_before)
        # after
        #time_diff_after_emb = self.embed_time_fn_after(time_input_after)
        #d_xyz_after_emb = self.embed_fn_after(d_xyz_after)
        # current
        #time_current_embed = self.embed_time_fn_current(time_input)
        # temporal before input
        #time_diff_before_emb =  time_current_embed - time_diff_before_emb
        # temporal after input
        #time_diff_after_emb =  time_diff_after_emb - time_current_embed
        # concat input data
        #d_xyz_before_embed_wt = torch.cat([time_diff_before_emb, d_xyz_before_emb], dim=-1)
        #d_xyz_after_embed_wt = torch.cat([time_diff_after_emb, d_xyz_after_emb], dim=-1)
#        d_xyz_before_embed_wt = d_xyz_before
#        d_xyz_after_embed_wt = d_xyz_after
        # model
#        weights, means, covariances = self.CustomModelWithGMM(d_xyz_before_embed_wt, d_xyz_after_embed_wt)
        # sample
#        sampled_means = self.CustomModelWithGMM.sample_gmm(weights, means, covariances, num_samples=1)
        # convert shape
#        sampled_means = sampled_means.squeeze(1)
        #### d_xyz
#        d_xyz = self.gaussian_warp(sampled_means) 
        #### d_rotation
#        d_rotation = self.gaussian_rotation(sampled_means)           
        #### d_scaling
#        d_scaling = self.gaussian_scaling(sampled_means) 
#        return d_xyz, d_rotation, d_scaling
    
'''
## GMM
def normalize_coordinates(coords):
    """
    Normalize xyz coordinates to the range [0, 1].
    """
    coords = coords - coords.min(0, keepdim=True)[0]  # Shift to start from zero
    coords = coords / coords.max(0, keepdim=True)[0]  # Normalize to [0, 1]
    return coords

def coords_to_grid(coords, grid_size):
    """
    Convert normalized xyz coordinates to a 3D grid.
    """
    batch_size = coords.size(0)
    grid = torch.zeros(batch_size, 1, grid_size, grid_size, grid_size)
    indices = (coords * (grid_size - 1)).long()
    for i in range(batch_size):
        x, y, z = indices[i]
        grid[i, 0, x, y, z] = 1
    return grid


class CustomModelWithGMM(nn.Module):
    def __init__(self, input_ch_before, input_ch_after, grid_size, hidden_dim, num_layers, num_components):
        super(CustomModelWithGMM, self).__init__()
        self.input_ch_before = input_ch_before
        self.input_ch_after = input_ch_after
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_components = num_components  # Number of Gaussian components in the GMM

        # Define 3D convolutional layers to capture spatial features
        self.conv1 = nn.Conv3d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)

        # Define GRU layer to capture temporal dependencies
        self.gru = nn.GRU(256 * grid_size * grid_size * grid_size, hidden_dim, num_layers, batch_first=True)

        # Layers to predict the GMM parameters
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_weights = nn.Linear(hidden_dim, num_components)  # Weights of the GMM
        self.fc_means = nn.Linear(hidden_dim, num_components * 3)  # Means of the GMM (3D)
        self.fc_covariances = nn.Linear(hidden_dim, num_components * 3 * 3)  # Covariances of the GMM (3x3)

    def forward(self, x_before, x_after):
        # Normalize and map to grid
        x_before_normalized = normalize_coordinates(x_before)
        x_after_normalized = normalize_coordinates(x_after)
        x_before_grid = coords_to_grid(x_before_normalized, self.grid_size)
        x_after_grid = coords_to_grid(x_after_normalized, self.grid_size)
        x_before_grid = x_before_grid.to(device)
        x_after_grid = x_after_grid.to(device)

        # Concatenate the key frames along the channel dimension
        x = torch.cat([x_before_grid, x_after_grid], dim=1)  # (batch_size, 2, grid_size, grid_size, grid_size)

        # Apply 3D convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and prepare for GRU
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (batch_size, 256 * grid_size * grid_size * grid_size)

        # Apply GRU
        x_combined = torch.stack((x, x), dim=1)  # (batch_size, 2, flattened_size)
        gru_output, _ = self.gru(x_combined)  # (batch_size, 2, hidden_dim)

        # Flatten the GRU output
        gru_output_flattened = gru_output.reshape(gru_output.size(0), -1)  # (batch_size, 2 * hidden_dim)

        # Predict the GMM parameters
        x = F.relu(self.fc1(gru_output_flattened))
        
        weights = F.softmax(self.fc_weights(x), dim=1)  # Ensure weights sum to 1
        means = self.fc_means(x).view(-1, self.num_components, 3)  # Reshape to (batch_size, num_components, 3)
        covariances = self.fc_covariances(x).view(-1, self.num_components, 3, 3)  # Reshape to (batch_size, num_components, 3, 3)

        # Ensure positive definiteness of covariances (simplest way: diagonal covariances)
        covariances = torch.exp(covariances)  # Exponentiate to ensure positive values

        return weights, means, covariances

    def sample_gmm(self, weights, means, covariances, num_samples=1):
        batch_size, num_components, _ = means.size()
        sampled_points = []

        for b in range(batch_size):
            component = torch.multinomial(weights[b], num_samples, replacement=True)
            chosen_means = means[b, component]
            chosen_covariances = covariances[b, component]
            
            sampled_point = torch.randn(num_samples, 3).to(means.device)
            for i in range(num_samples):
                sampled_point[i] = torch.matmul(chosen_covariances[i], sampled_point[i]) + chosen_means[i]
            
            sampled_points.append(sampled_point)
        
        sampled_points = torch.stack(sampled_points)
        return sampled_points
    
## Ver 2
class Deform_Predict_ver2(nn.Module):
    def __init__(self, D=8, W=64, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False): #D=8, W=256
        super(Deform_Predict_ver2, self).__init__()
        self.D = D
        self.W = W
        self.skips = [D // 2]
        self.t_multires = 6 if is_blender else 5
        #self.input_ch = input_ch

        # time embedding
        #self.embed_time_fn_before, time_input_ch_before = get_embedder(self.t_multires, 1)
        #self.embed_time_fn_after, time_input_ch_after = get_embedder(self.t_multires, 1)
        self.embed_time_fn_current, time_input_ch_current = get_embedder(self.t_multires, 1)
        # x_embedding
        self.embed_fn_before, xyz_input_ch_before = get_embedder(multires, 3)
        self.embed_fn_after, xyz_input_ch_after = get_embedder(multires, 3)

        self.input_ch_before = xyz_input_ch_before #+ time_input_ch_before
        self.input_ch_after = xyz_input_ch_after #+ time_input_ch_after


        # d_xyz
        self.linear_dxyz_past = nn.ModuleList(
                [nn.Linear(self.input_ch_before, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch_before, W)
                    for i in range(D - 1)]
            )
        self.linear_dxyz_future = nn.ModuleList(
                [nn.Linear(self.input_ch_after, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch_after, W)
                    for i in range(D - 1)]
            )
        self.linear_dxyz = nn.ModuleList(
                [nn.Linear(W*2 + time_input_ch_current, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + W*2 + time_input_ch_current, W)
                    for i in range(D - 1)]
            )
        self.gaussian_warp = nn.Linear(W, 3)

        # d_rotation
        self.gaussian_rotation = nn.Linear(W, 4)

        # d_scaling
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, time_input,time_input_before,time_input_after,d_xyz_before, d_xyz_after):
        
        #Mask similar Gaussians between frames
        #temporal_invariant = torch.sigmoid(abs(d_xyz_before-d_xyz_after))

        ### embedding time
        # before
        #time_diff_before_emb = self.embed_time_fn_before(time_input_before)
        d_xyz_before_emb = self.embed_fn_before(d_xyz_before)
        d_xyz_before_embed_wt = d_xyz_before_emb #torch.cat([time_diff_before_emb, d_xyz_before_emb], dim=-1)
        # after
        #time_diff_after_emb = self.embed_time_fn_after(time_input_after)
        d_xyz_after_emb = self.embed_fn_after(d_xyz_after)
        d_xyz_after_embed_wt = d_xyz_after_emb #torch.cat([time_diff_after_emb, d_xyz_after_emb], dim=-1)
        # current
        time_current_embed = self.embed_time_fn_current(time_input)
        
        #### d_xyz
        ### d_xyz before
        d_xyz_before_wt = d_xyz_before_embed_wt
        for i, l in enumerate(self.linear_dxyz_past):
            d_xyz_before_wt = self.linear_dxyz_past[i](d_xyz_before_wt)
            d_xyz_before_wt = F.relu(d_xyz_before_wt)
            if i in self.skips:
                d_xyz_before_wt = torch.cat([d_xyz_before_embed_wt, d_xyz_before_wt], -1)
        ### d_xyz after
        d_xyz_after_wt = d_xyz_after_embed_wt
        for i, l in enumerate(self.linear_dxyz_future):
            d_xyz_after_wt = self.linear_dxyz_future[i](d_xyz_after_wt)
            d_xyz_after_wt = F.relu(d_xyz_after_wt)
            if i in self.skips:
                d_xyz_after_wt = torch.cat([d_xyz_after_embed_wt, d_xyz_after_wt], -1)      
        ### Combine before-after d_xyz  
        d_xyz_r = torch.cat([d_xyz_before_wt, d_xyz_after_wt, time_current_embed], dim=-1)
        ### d_xyz convert to output
        d_xyz_r_wt = d_xyz_r
        for i, l in enumerate(self.linear_dxyz):
            d_xyz_r_wt = self.linear_dxyz[i](d_xyz_r_wt)
            d_xyz_r_wt = F.relu(d_xyz_r_wt)
            if i in self.skips:
                d_xyz_r_wt = torch.cat([d_xyz_r, d_xyz_r_wt], -1) 
        d_xyz = self.gaussian_warp(d_xyz_r_wt)           

        #### d_rotation
        d_rotation = self.gaussian_rotation(d_xyz_r_wt)           

        #### d_scaling
        d_scaling = self.gaussian_scaling(d_xyz_r_wt) 

        return d_xyz, d_rotation, d_scaling



#### ---------------------------------------------------------- Dyanmic Mask ---------------------------------------------------------- ####
class Dynamic_Mask(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False): #D=8
        super(Dynamic_Mask, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 5
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )
        
        self.linear_out = nn.Linear(W,3)

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        #if self.is_blender:
        #    t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        #### ORIGINAL CODE
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)
        return torch.sigmoid(self.linear_out(h))



#### ---------------------------------------------------------- Gaussian Distribution Prediction ---------------------------------------------------------- ####
## Gaussian Distribution Prediction in 2D-planes
class GaussianPredictionModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(GaussianPredictionModel, self).__init__()
        self.dense1 = nn.Linear(input_dim, embed_dim)
        self.self_attention = SelfAttentionLayer(embed_dim, num_heads)
        self.flatten = nn.Flatten()
        self.dense2 = nn.Linear(embed_dim * input_dim, 128)
        self.mean_output_xy = nn.Linear(128, 2)
        self.log_var_output_xy = nn.Linear(128, 2)
        self.mean_output_yz = nn.Linear(128, 2)
        self.log_var_output_yz = nn.Linear(128, 2)
        self.mean_output_xz = nn.Linear(128, 2)
        self.log_var_output_xz = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.self_attention(x)
        x = self.flatten(x)
        x = F.relu(self.dense2(x))
        
        mean_xy = self.mean_output_xy(x)
        log_var_xy = self.log_var_output_xy(x)
        
        mean_yz = self.mean_output_yz(x)
        log_var_yz = self.log_var_output_yz(x)
        
        mean_xz = self.mean_output_xz(x)
        log_var_xz = self.log_var_output_xz(x)
        
        return mean_xy, log_var_xy, mean_yz, log_var_yz, mean_xz, log_var_xz
    

def combine_gaussians(mu_xy, sigma_xy, mu_yz, sigma_yz, mu_xz, sigma_xz):
    # Combine x
    mu_x_combined = (mu_xy[0] / sigma_xy[0]**2 + mu_xz[0] / sigma_xz[0]**2) / (1 / sigma_xy[0]**2 + 1 / sigma_xz[0]**2)
    sigma_x_combined = (1 / (1 / sigma_xy[0]**2 + 1 / sigma_xz[0]**2))**0.5

    # Combine y
    mu_y_combined = (mu_xy[1] / sigma_xy[1]**2 + mu_yz[0] / sigma_yz[0]**2) / (1 / sigma_xy[1]**2 + 1 / sigma_yz[0]**2)
    sigma_y_combined = (1 / (1 / sigma_xy[1]**2 + 1 / sigma_yz[0]**2))**0.5

    # Combine z
    mu_z_combined = (mu_yz[1] / sigma_yz[1]**2 + mu_xz[1] / sigma_xz[1]**2) / (1 / sigma_yz[1]**2 + 1 / sigma_xz[1]**2)
    sigma_z_combined = (1 / (1 / sigma_yz[1]**2 + 1 / sigma_xz[1]**2))**0.5

    return (mu_x_combined, mu_y_combined, mu_z_combined), (sigma_x_combined, sigma_y_combined, sigma_z_combined)

def predict_point(mu_combined, sigma_combined):
    return torch.tensor(mu_combined)


class Gaussians_distribution_predict(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=8):
        super(Gaussians_distribution_predict, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        ### Gaussian Prediction Model
        self.GaussianPredictionModel = GaussianPredictionModel(self.input_dim, self.embed_dim, self.num_heads)
    def forward(self,x):
        ## 2D-Gaussian distributions for each xy-, yz-, xz-planes
        mean_xy, log_var_xy, mean_yz, log_var_yz, mean_xz, log_var_xz = self.GaussianPredictionModel(x)
        ## Combine 2D-plane for x,y,z
        combined_mean, combined_sigma = combine_gaussians(mean_xy, log_var_xy, mean_yz, log_var_yz, mean_xz, log_var_xz)
        ## Predict the final point
        predicted_point = predict_point(combined_mean, combined_sigma)

        return predicted_point
    


class Deform_Predict_with_Gaussians_distribution(nn.Module):
    def __init__(self, multires=10, embed_dim=128, num_heads=8): #D=8, W=256
        super(Deform_Predict_with_Gaussians_distribution, self).__init__()
        #self.t_multires = 5
        # time embedding
        #self.embed_time_fn_before, time_input_ch_before = get_embedder(self.t_multires, 1)
        #self.embed_time_fn_after, time_input_ch_after = get_embedder(self.t_multires, 1)
        #self.embed_time_fn_current, time_input_ch_current = get_embedder(self.t_multires, 1)
        # x_embedding
        self.embed_fn_before, xyz_input_ch_before = get_embedder(multires, 3)
        self.embed_fn_after, xyz_input_ch_after = get_embedder(multires, 3)
        # input dim
        self.input_ch_before = xyz_input_ch_before #+ time_input_ch_before
        self.input_ch_after = xyz_input_ch_after #+ time_input_ch_after

        # Gaussians distribution model
        self.input_dim = self.input_ch_before + self.input_ch_after
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.Gaussians_distribution_predict = Gaussians_distribution_predict(input_dim = self.input_dim, embed_dim= self.embed_dim, num_heads=self.num_heads)

        # xyz
        self.gaussian_warp = nn.Linear(3, 3)
        # d_rotation
        self.gaussian_rotation = nn.Linear(3, 4)
        # d_scaling
        self.gaussian_scaling = nn.Linear(3, 3)

    def forward(self, d_xyz_before, d_xyz_after):
        
        #Mask similar Gaussians between frames
        #temporal_invariant = torch.sigmoid(abs(d_xyz_before-d_xyz_after))

        ### embedding time
        # before
        #time_diff_before_emb = self.embed_time_fn_before(time_input_before)
        d_xyz_before_emb = self.embed_fn_before(d_xyz_before)
        #d_xyz_before_embed_wt = torch.cat([time_diff_before_emb, d_xyz_before_emb], dim=-1)
        # after
        #time_diff_after_emb = self.embed_time_fn_after(time_input_after)
        d_xyz_after_emb = self.embed_fn_after(d_xyz_after)
        #d_xyz_after_embed_wt = torch.cat([time_diff_after_emb, d_xyz_after_emb], dim=-1)
        # current
        #time_current_embed = self.embed_time_fn_current(time_input)
        
        #### combine d_xyz
        d_xyz_combined = torch.cat([d_xyz_before_emb,d_xyz_after_emb], dim=-1)
        d_xyz_output = self.Gaussians_distribution_predict(d_xyz_combined)

        #### d_xyz        
        d_xyz = self.gaussian_warp(d_xyz_output)           
        #### d_rotation
        d_rotation = self.gaussian_rotation(d_xyz_output)           
        #### d_scaling
        d_scaling = self.gaussian_scaling(d_xyz_output) 

        return d_xyz, d_rotation, d_scaling
'''