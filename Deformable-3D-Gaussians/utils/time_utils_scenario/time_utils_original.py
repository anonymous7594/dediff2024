import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3


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
    def __init__(self, D=4, W=64, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False):
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

    


class Deform_Predict(nn.Module):
    def __init__(self, D=8, W=128, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False): #D=8, W=256
        super(Deform_Predict, self).__init__()
        self.D = D
        self.W = W
        self.skips = [D // 2]
        self.t_multires = 6 if is_blender else 5
        #self.input_ch = input_ch

        # time embedding
        self.embed_time_fn_before, time_input_ch_before = get_embedder(self.t_multires, 1)
        self.embed_time_fn_after, time_input_ch_after = get_embedder(self.t_multires, 1)
        self.embed_time_fn_current, time_input_ch_current = get_embedder(self.t_multires, 1)
        # x_embedding
        self.embed_fn_before, xyz_input_ch_before = get_embedder(multires, 3)
        self.embed_fn_after, xyz_input_ch_after = get_embedder(multires, 3)

        self.input_ch_before = time_input_ch_before + xyz_input_ch_before
        self.input_ch_after = time_input_ch_after + xyz_input_ch_after


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
        temporal_invariant = torch.sigmoid(abs(d_xyz_before-d_xyz_after))

        ### embedding time
        # before
        time_diff_before_emb = self.embed_time_fn_before(time_input_before)
        d_xyz_before_emb = self.embed_fn_before(d_xyz_before*temporal_invariant)
        d_xyz_before_embed_wt = torch.cat([time_diff_before_emb, d_xyz_before_emb], dim=-1)
        # after
        time_diff_after_emb = self.embed_time_fn_after(time_input_after)
        d_xyz_after_emb = self.embed_fn_after(d_xyz_after*temporal_invariant)
        d_xyz_after_embed_wt = torch.cat([time_diff_after_emb, d_xyz_after_emb], dim=-1)
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
            