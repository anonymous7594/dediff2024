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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
#import numpy as np


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


### KL Divergence Loss provided by original code
def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5)))



### KL Divergence Loss Update
def kl_divergence_loss(encoded1, encoded2):
    # Ensure both tensors represent probability distributions using softmax
    probs1 = F.softmax(encoded1, dim=-1)
    probs2 = F.softmax(encoded2, dim=-1)

    # Compute the log probabilities
    log_probs1 = torch.log(probs1 + 1e-10)  # Adding a small epsilon to avoid log(0)
    log_probs2 = torch.log(probs2 + 1e-10)

    #print('log_probs1: ',log_probs1)
    #print('log_probs2: ',log_probs2)

    # Calculate the KL divergence
    kl_div = F.kl_div(log_probs1, probs2, reduction='batchmean')
    #print('kl_div: ',kl_div)

    return kl_div

### Contrastive Loss
def contrastive_loss(z_i, z_j, temperature=0.5):
    # Normalize the embeddings
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Concatenate the embeddings
    z = torch.cat([z_i, z_j], dim=0)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(z, z.T)
    
    # Create labels
    batch_size = z_i.shape[0]
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).cuda()
    
    # Mask to remove self-similarity
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    
    # Apply the temperature scaling
    similarity_matrix = similarity_matrix / temperature
    
    # Apply mask to remove self-similarity
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    
    # Compute the contrastive loss
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss.mean()



def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = img1.to(device)
    img2 = img2.to(device)
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


### Motion Loss
## ver 1
#def motion_loss(p_t_minus_h, p_t, p_t_plus_h, h):
    ## Calculate velocities
    #v_t_minus_h = (p_t - p_t_minus_h) / h
    #v_t = (p_t_plus_h - p_t) / h

    ## Calculate acceleration
    #a_t = (p_t_plus_h - 2 * p_t + p_t_minus_h) / (h ** 2)

    ## Calculate motion loss (squared norm of acceleration)
    #motion_loss = (torch.norm(a_t) ** 2).mean()

    #return motion_loss
## ver 2
def calculate_deformation(positions_t1, positions_t2):
    return positions_t2 - positions_t1

def calculate_motion_loss(positions_t_minus_h, positions_t, positions_t_plus_h):
    # Calculate deformations between consecutive time steps
    deformation_t_minus_h_to_t = calculate_deformation(positions_t_minus_h, positions_t)
    deformation_t_to_t_plus_h = calculate_deformation(positions_t, positions_t_plus_h)
    
    # Motion loss is the mean of the norm of the difference in deformations
    motion_loss = torch.norm(deformation_t_minus_h_to_t - deformation_t_to_t_plus_h, dim=-1).mean()
    
    return motion_loss


def calculate_motion_loss_ver_2(positions_t_minus_h, positions_t, positions_t_plus_h, t_diff_minus_h, t_diff_plus_h):
    # Calculate deformations between consecutive time steps
    deformation_t_minus_h_to_t = abs(calculate_deformation(positions_t_minus_h, positions_t))
    deformation_t_to_t_plus_h = abs(calculate_deformation(positions_t, positions_t_plus_h))

    # Assign weight based on time diff
    total_diff = abs(t_diff_minus_h) + abs(t_diff_plus_h)
    weight_minus = abs(t_diff_plus_h)/total_diff
    weight_plus = abs(t_diff_minus_h)/total_diff
    
    # Motion loss is the mean of the norm of the difference in deformations
    motion_loss = torch.norm(weight_minus*deformation_t_minus_h_to_t - weight_plus*deformation_t_to_t_plus_h, dim=-1).mean()
    
    return motion_loss

'''
def calculate_motion_loss_ver_2(positions_t_minus_h, positions_t, positions_t_plus_h, t_minus, t_plus): #### -- exp with bad results
    # Calculate deformations between consecutive time steps
    deformation_t_minus_h_to_t = calculate_deformation(positions_t_minus_h, positions_t)/t_minus
    deformation_t_to_t_plus_h = calculate_deformation(positions_t, positions_t_plus_h)/t_plus
    
    # Motion loss is the mean of the norm of the difference in deformations
    motion_loss = torch.norm(deformation_t_minus_h_to_t - deformation_t_to_t_plus_h, dim=-1).mean()
    
    return motion_loss

def calculate_motion_loss_ver_3(positions_t_minus_h, positions_t, positions_t_plus_h):
    ## xy dimension
    # Calculate deformations between consecutive time steps
    deformation_t_minus_h_to_t_xy = calculate_deformation(positions_t_minus_h[:,0:2], positions_t[:,0:2])
    deformation_t_to_t_plus_h_xy = calculate_deformation(positions_t[:,0:2], positions_t_plus_h[:,0:2])
    # Motion loss is the mean of the norm of the difference in deformations
    motion_loss_xy = torch.norm(deformation_t_minus_h_to_t_xy - deformation_t_to_t_plus_h_xy, dim=-1).mean()

    ## yz dimension
    # Calculate deformations between consecutive time steps
    deformation_t_minus_h_to_t_yz = calculate_deformation(positions_t_minus_h[:,1:], positions_t[:,1:])
    deformation_t_to_t_plus_h_yz = calculate_deformation(positions_t[:,1:], positions_t_plus_h[:,1:])
    # Motion loss is the mean of the norm of the difference in deformations
    motion_loss_yz = torch.norm(deformation_t_minus_h_to_t_yz - deformation_t_to_t_plus_h_yz, dim=-1).mean()

    ## yz dimension
    # Calculate deformations between consecutive time steps
    deformation_t_minus_h_to_t_xz = calculate_deformation(positions_t_minus_h[:,[0, -1]], positions_t[:,[0, -1]])
    deformation_t_to_t_plus_h_xz = calculate_deformation(positions_t[:,[0, -1]], positions_t_plus_h[:,[0, -1]])
    # Motion loss is the mean of the norm of the difference in deformations
    motion_loss_xz = torch.norm(deformation_t_minus_h_to_t_xz - deformation_t_to_t_plus_h_xz, dim=-1).mean()

    motion_loss = (motion_loss_xy + motion_loss_yz + motion_loss_xz)/3
    
    return motion_loss

### Rotation Loss
def calculate_rotation(positions_t1, positions_t2):
    return positions_t2 - positions_t1

def calculate_rotation_loss(rotation_t_minus_h, rotation_t, rotation_t_plus_h):
    # Calculate deformations between consecutive time steps
    deformation_t_minus_h_to_t = calculate_rotation(rotation_t_minus_h, rotation_t)
    deformation_t_to_t_plus_h = calculate_rotation(rotation_t, rotation_t_plus_h)
    
    # rotation loss is the mean of the norm of the difference in deformations
    rotation_loss = torch.norm(deformation_t_minus_h_to_t - deformation_t_to_t_plus_h, dim=-1).mean()
    
    return rotation_loss


### Rigid Loss
## ver 1
def knn(points, k):
    """
    Find the k-nearest neighbors for each point.
    Args:
        points (torch.Tensor): Tensor of shape [sample, 3] representing the positions.
        k (int): Number of nearest neighbors to find.
    Returns:
        torch.Tensor: Tensor of shape [sample, k] with the indices of the nearest neighbors.
    """
    num_samples = points.size(0)
    distances = torch.cdist(points, points)  # Compute pairwise distances
    knn_indices = distances.topk(k+1, largest=False).indices[:, 1:]  # Get the k nearest neighbors, excluding self
    return knn_indices

def calculate_local_distances(points, knn_indices):
    """
    Calculate the local distances to the k-nearest neighbors.
    Args:
        points (torch.Tensor): Tensor of shape [sample, 3] representing the positions.
        knn_indices (torch.Tensor): Tensor of shape [sample, k] with the indices of the nearest neighbors.
    Returns:
        torch.Tensor: Tensor of shape [sample, k] with the distances to the k-nearest neighbors.
    """
    num_samples, k = knn_indices.size()
    neighbors = points[knn_indices.view(-1)].view(num_samples, k, -1)
    local_distances = torch.norm(points.unsqueeze(1) - neighbors, dim=-1)
    return local_distances

def calculate_rigid_loss_knn(positions_t_minus_h, positions_t, positions_t_plus_h, k):
    """
    Calculate the rigid loss based on the consistency of local distances to k-nearest neighbors.
    Args:
        positions_t_minus_h (torch.Tensor): Tensor of shape [sample, 3] at time t - h.
        positions_t (torch.Tensor): Tensor of shape [sample, 3] at time t.
        positions_t_plus_h (torch.Tensor): Tensor of shape [sample, 3] at time t + h.
        k (int): Number of nearest neighbors to consider.
    Returns:
        torch.Tensor: The rigid loss value.
    """
    # Find k-nearest neighbors
    knn_indices_t_minus_h = knn(positions_t_minus_h, k)
    knn_indices_t = knn(positions_t, k)
    knn_indices_t_plus_h = knn(positions_t_plus_h, k)
    
    # Calculate local distances at each time step
    local_distances_t_minus_h = calculate_local_distances(positions_t_minus_h, knn_indices_t_minus_h)
    local_distances_t = calculate_local_distances(positions_t, knn_indices_t)
    local_distances_t_plus_h = calculate_local_distances(positions_t_plus_h, knn_indices_t_plus_h)
    
    # Calculate the change in local distances between time steps
    delta_d_t_minus_h_to_t = local_distances_t - local_distances_t_minus_h
    delta_d_t_to_t_plus_h = local_distances_t_plus_h - local_distances_t
    
    # Calculate the rigid loss as the sum of squared changes in local distances
    rigid_loss = torch.sum(delta_d_t_minus_h_to_t**2) + torch.sum(delta_d_t_to_t_plus_h**2)
    
    return rigid_loss.mean()
    '''