import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3
from utils.loss_utils import kl_divergence_loss, kl_divergence, contrastive_loss
import os



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
    def __init__(self, D=8, W=64, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False):  #D=8,W=256
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
## Ver 1
class Deform_Predict(nn.Module):
    def __init__(self, D=8, W=64, Dd=8, Wd=64, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False, embed_dim = 768, num_heads = 12): #D=8, W=256
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

        ## Static-Dynamic Masking
        self.linear_filter = nn.Linear(3,1)

        ## Define a linear layer for image projection
        self.projection_layer_before = nn.Linear(768, self.W) # if using vit_base_patch16_224_dino
        self.projection_layer_after = nn.Linear(768, self.W) # if using vit_base_patch16_224_dino
        # with self-attention
        self.embed_dim = embed_dim
        self.num_heads = num_heads 
        self.self_attention = SelfAttention(self.embed_dim, self.num_heads) #input as [1, 197, 768]
        self.feed_forward = FeedForward(embed_dim, self.W)

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
        # Current Input
        self.embed_time_fn_current, time_input_ch_current = get_embedder(self.t_multires, 1)


        # input
        # pre-determined
        # before + h
        self.linear_dxyz_past = nn.ModuleList(
                [nn.Linear(self.input_ch_before, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch_before, W)
                    for i in range(D - 1)]
            )

        # after - h
        self.linear_dxyz_future = nn.ModuleList(
                [nn.Linear(self.input_ch_after, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch_after, W)
                    for i in range(D - 1)]
            )

        # output
        # xyz
        self.linear_dxyz = nn.ModuleList(
                [nn.Linear(W*2 + time_input_ch_current, Wd)] + [
                    nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + W*2 + time_input_ch_current, Wd)
                    for i in range(Dd - 1)])
        # image embedding - MLP
        self.linear_image = nn.ModuleList(
                [nn.Linear(W*2 + time_input_ch_current, Wd)] + [
                    nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + W*2 + time_input_ch_current, Wd)
                    for i in range(Dd - 1)])
        # combined to get the output - if using MLP
        #self.linear_combined = nn.ModuleList(
        #        [nn.Linear(Wd*2, Wd)] + [
        #            nn.Linear(Wd, Wd) if i not in self.skips_d else nn.Linear(Wd + Wd*2, Wd)
        #            for i in range(Dd - 1)])

        # combined xyz-image
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

        
        ## Output by linear layer
        # d_xyz
        self.gaussian_warp = nn.Linear(Wd, 3)
        # d_rotation
        self.gaussian_rotation = nn.Linear(Wd, 4)
        # d_scaling
        self.gaussian_scaling = nn.Linear(Wd, 3)
        ## Output by MLPs
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

    def forward(self, time_input,time_input_before,time_input_after,d_xyz_before, d_xyz_after, features_before, features_after):
        ### Apply Static-Dynamic mask
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


        ### xyz embedding
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

        ### image embedding   
        ## Before
        # Flatten the input tensor to remove the batch dimension
        #flattened_features_before = features_before.view(-1, 768)  # Shape: [197, 768] # if using vit_base_patch16_224_dino
        # Apply the projection layer to the flattened tensor
        #projected_features_before = self.projection_layer_before(flattened_features_before) 
        # Repeat the projected tensor to achieve the shape [174498, 64]
        ## Apply Self-Attention
        # Before
        features_before_embeded = self.self_attention(features_before)
        projected_features_before = self.feed_forward(features_before_embeded)
        # After
        features_after_embeded = self.self_attention(features_after)
        projected_features_after = self.feed_forward(features_after_embeded)
        # Calculate the number of repetitions needed
        num_repeats = d_xyz_after_emb.shape[0] // 197 # if using vit_base_patch16_224_dino
        # Repeat the tensor
        expanded_features_before = projected_features_before.repeat(num_repeats + 1, 1)[:d_xyz_after_emb.shape[0]]  # Shape: [174498, 64]
        ## After
        # Flatten the input tensor to remove the batch dimension
        #flattened_features_after = features_after.view(-1, 768)  # Shape: [197, 768] # if using vit_base_patch16_224_dino
        # Apply the projection layer to the flattened tensor
        #projected_features_after = self.projection_layer_after(flattened_features_after) 
        # Repeat the projected tensor to achieve the shape [174498, 64]
        # Repeat the tensor
        expanded_features_after = projected_features_after.repeat(num_repeats + 1, 1)[:d_xyz_after_emb.shape[0]]  # Shape: [174498, 64]
        # KL Divergence Loss for image embedding
        #embeded_image_loss = kl_divergence_loss(expanded_features_before,expanded_tensor_after)
        #print('embeded_image_loss: ',embeded_image_loss)
        
        #### pre-determined
        ### d_xyz before
        d_xyz_before_wt = d_xyz_before_embed_wt
        for i, l in enumerate(self.linear_dxyz_past):
        #for i, (l, norm_layer) in enumerate(zip(self.linear_dxyz_past, self.norm_layers)):
            d_xyz_before_wt = self.linear_dxyz_past[i](d_xyz_before_wt)                
            #d_xyz_before_wt = self.norm_layers_past(d_xyz_before_wt)
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
            #d_xyz_after_wt = self.norm_layers_future(d_xyz_after_wt)
            d_xyz_after_wt = F.relu(d_xyz_after_wt)
            #if i == (self.D//2 - 1) :
            #    d_xyz_after_wt = self.drop_out(d_xyz_after_wt)
            if i in self.skips:
                d_xyz_after_wt = torch.cat([d_xyz_after_embed_wt, d_xyz_after_wt], -1)
        # KL Divergence Loss for xyz
        #embeded_xyz_loss = kl_divergence_loss(d_xyz_before_wt,d_xyz_after_wt)
        #print('embeded_xyz_loss: ',embeded_xyz_loss)
        # Total KL Divergence Loss
        #kl_loss = embeded_xyz_loss #+ 0.01*embeded_image_loss
        #print('kl_loss: ',kl_loss)
        #contrastive_loss_result = contrastive_loss(d_xyz_before_wt, d_xyz_after_wt, temperature=0.5)

        #### xyz input
        ### Combine before-after d_xyz  
        #d_xyz_before_wt = d_xyz_before_wt + expanded_features_before
        #d_xyz_after_wt = d_xyz_after_wt + expanded_features_after
        d_xyz_r = torch.cat([d_xyz_before_wt, d_xyz_after_wt, time_current_embed], dim=-1)
        ### d_xyz convert to output
        d_xyz_r_wt = d_xyz_r
        for i, l in enumerate(self.linear_dxyz):
            d_xyz_r_wt = self.linear_dxyz[i](d_xyz_r_wt)
            #d_xyz_r_wt = self.norm_layers_combined(d_xyz_r_wt)
            d_xyz_r_wt = F.relu(d_xyz_r_wt)
            #if i == (self.Dd//2 - 1) :
            #    d_xyz_r_wt = self.drop_out(d_xyz_r_wt)
            if i in self.skips_d:
                d_xyz_r_wt = torch.cat([d_xyz_r, d_xyz_r_wt], -1) 
        
        #### image input
        image_combined = torch.cat([expanded_features_before, expanded_features_after, time_current_embed], dim=-1)
        ### image convert to output
        image_combined_wt = image_combined
        for i, l in enumerate(self.linear_image):
            image_combined_wt = self.linear_image[i](image_combined_wt)
            image_combined_wt = F.relu(image_combined_wt)
            #if i == (self.Dd//2 - 1) :
            #    image_combined_wt = self.drop_out(image_combined_wt)
            if i in self.skips_d:
                image_combined_wt = torch.cat([image_combined, image_combined_wt], -1) 

        ##### xyz and image input
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

        #### combined input
        #combined_input = torch.cat([d_xyz_r_wt, image_combined_wt], dim=-1)
        ### combined input convert to output
        #combined_input_wt = combined_input
        #for i, l in enumerate(self.linear_combined):
        #    combined_input_wt = self.linear_combined[i](combined_input_wt)
        #    combined_input_wt = F.relu(combined_input_wt)
        #    if i in self.skips_d:
        #        combined_input_wt = torch.cat([combined_input, combined_input_wt], -1) 
        
        #if output by linear
        combined_input_wt = d_xyz_r_wt + image_combined_wt #+ combined_before_wt + combined_after_wt
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
        dxyz_remaining = torch.zeros_like(d_xyz_before_static).to(device)
        ## combine value
        d_xyz_combined = torch.cat((d_xyz, dxyz_remaining), dim=0)

        #### d_rotation
        ## dynamic part
        # Linear
        d_rotation = self.gaussian_rotation(combined_input_wt)      
        # MLP
        #d_rotation = combined_input_wt.clone()
        #for i, l in enumerate(self.gaussian_rotation):
        #    d_rotation = self.gaussian_rotation[i](d_rotation)
        #    d_rotation = F.relu(d_rotation)
        ## static part   
        drotation_remaining = torch.zeros((d_xyz_before_static.size(0), 4)).to(device)
        ## combine value
        d_rotation_combined = torch.cat((d_rotation, drotation_remaining), dim=0)

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
        dscaling_remaining = torch.zeros_like(d_xyz_before_static).to(device)
        ## combine value
        d_scaling_combined = torch.cat((d_scaling, dscaling_remaining), dim=0)

        return d_xyz_combined, d_rotation_combined, d_scaling_combined #, kl_loss, contrastive_loss
    


## Self-Attention for Image Embedding
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x):
        N = x.shape[0]
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        out = out.view(N, -1, self.embed_dim)
        
        out = self.fc_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


## Ver 3
class Deform_Predict_ver_with_Probability_Distribution(nn.Module):
    def __init__(self, grid_size=16, hidden_dim=2, t_multires=5, num_layers=4, num_components=5, multires=10, is_blender=False, is_6dof=False):
        super(Deform_Predict, self).__init__()
        self.grid_size = grid_size
        self.t_multires = t_multires
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_components = num_components

        # pre-determined
        # time embedding
        self.embed_time_fn_before, time_input_ch_before = get_embedder(self.t_multires, 1)
        self.embed_time_fn_after, time_input_ch_after = get_embedder(self.t_multires, 1)
        # x_embedding
        self.embed_fn_before, xyz_input_ch_before = get_embedder(multires, 3)
        self.embed_fn_after, xyz_input_ch_after = get_embedder(multires, 3)
        # Current Input
        self.embed_time_fn_current, time_input_ch_current = get_embedder(self.t_multires, 1)
        # Number of Input
        self.input_ch_before = time_input_ch_before + xyz_input_ch_before
        self.input_ch_after = time_input_ch_after + xyz_input_ch_after
 
        # Model
        self.CustomModelWithGMM = CustomModelWithGMM(self.input_ch_before, self.input_ch_after, self.grid_size, self.hidden_dim, self.num_layers, self.num_components)

        # d_xyz
        self.gaussian_warp = nn.Linear(3, 3)
        # d_rotation
        self.gaussian_rotation = nn.Linear(3, 4)
        # d_scaling
        self.gaussian_scaling = nn.Linear(3, 3)

    def forward(self, time_input,time_input_before,time_input_after,d_xyz_before, d_xyz_after):
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
        d_xyz_before_embed_wt = d_xyz_before
        d_xyz_after_embed_wt = d_xyz_after
        

        # model
        weights, means, covariances = self.CustomModelWithGMM(d_xyz_before_embed_wt, d_xyz_after_embed_wt)
        # sample
        sampled_means = self.CustomModelWithGMM.sample_gmm(weights, means, covariances, num_samples=1)
        # convert shape
        sampled_means = sampled_means.squeeze(1)

        #### d_xyz
        d_xyz = self.gaussian_warp(sampled_means) 

        #### d_rotation
        d_rotation = self.gaussian_rotation(sampled_means)           

        #### d_scaling
        d_scaling = self.gaussian_scaling(sampled_means) 

        return d_xyz, d_rotation, d_scaling
    


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

#### ---------------------------------------------------------- VAE Prediction ---------------------------------------------------------- ####
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Latent space layers
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
            

#### ---------------------------------------------------------- Gaussian Distribution Prediction ---------------------------------------------------------- ####
## Self Attention
class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        return self.norm2(out1 + ffn_output)
    
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
