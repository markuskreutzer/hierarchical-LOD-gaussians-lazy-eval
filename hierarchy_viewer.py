#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from utils.general_utils import get_expon_lr_func
import os
import torch
from torch import nn
import debug_utils
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_post, render, render_coarse, render_on_disk, render_vanilla, render_stp, network_gui
import sys
from scene import Scene, GaussianModel, OurAdam
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math
import torchvision
from fused_ssim import fused_ssim
import random
from torch.utils.tensorboard import SummaryWriter
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights, expand_to_size_dynamic, get_interpolation_weights_dynamic
import time
from torchviz import make_dot
import consistency_graph
from scipy.spatial import KDTree
import numpy as np
from gaussian_hierarchy._C import  get_spt_cut_cuda
from gaussian_renderer import occlusion_cull
import json
import pickle
from debugging import timestamp_wrapper

clock_start = True
clock_time = time.time()
def clock():
    global clock_start
    global clock_time
    if clock_start:
        clock_start = False
        clock_time = time.time()
    else:
        clock_start = True
        return time.time()-clock_time


render_metrics = timestamp_wrapper()

def direct_collate(x):
    return x


WriteTensorBoard = False
#Standard
densify_interval = 200
lr_multiplier = 1
#Unused
Random_Hierarchy_Cut = True
Only_Noise_Visible = True
#MCMC
Max_Cap = 25_000_000
MCMC_Densification = True
MCMC_Noise_LR = 0  #5e5
lambda_scaling = 0
lambda_opacity = 0
#Hierarchical
Gaussian_Interpolation = False
# Upward Propagation 
Gradient_Propagation = False
Propagation_Strength = 1.0
#Culling
Use_Bounding_Spheres = False
Use_Occlusion_Culling = False
Use_Frustum_Culling = True
Use_MIP_respawn = False
# SPTs
Storage_Device = 'cpu'
lambda_hierarchy = 0.00
SPT_Root_Volume = 5 #0.025
Target_Granularity_Pixels = 5
Cache_SPTs = True
Reuse_SPT_Tolerarance = 0.1
#View Selection
Use_Consistency_Graph = False
# Rasterizer
Rasterizer = "Vanilla"


SH_properties = [0, 3, 8, 15]
SH_properties_single = SH_properties[1] 
SH_properties = SH_properties[1] * 3
xyz1 = 0
xyz2 = 3
scales1 = 3
scales2 = 6
rotation1 = 6
rotation2 = 10
features1 = 10
features2 = 13
opacity1 = 13
opacity2 = 14
features_rest1 = 14
features_rest2 = 14 + SH_properties
number_properties = features_rest2

range1 = [xyz1, scales1, rotation1, features1, opacity1, features_rest1]
range2 = [xyz2, scales2, rotation2, features2, opacity2, features_rest2]

non_blocking=False
def render(dataset, opt:OptimizationParams, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from,  hierarchy_path, replay=False, cam_path_id=0):
    camera_path_id = random.randint(0, 100000) if not replay else cam_path_id
    global Reuse_SPT_Tolerarance
    global Use_Occlusion_Culling
    opt.densification_interval = densify_interval
    #torch.cuda.memory._record_memory_history()
    #torch.autograd.set_detect_anomaly(True)
    
    splat_settings = None
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = 1
    gaussians.max_sh_degree = 1
    gaussians.scaffold_points = None
    with torch.no_grad():
        gaussians._features_dc = gaussians._features_dc.abs() 
    dataset.eval = True
    dataset.hierarchy = hierarchy_path
    scene = Scene(dataset, gaussians, resolution_scales = [1], create_from_hier=True)
    gaussians.skybox_points = 100000
    base_focal_length = 1000 #TODO ????
    SPT_Target_Granularity = (1.0/base_focal_length) * Target_Granularity_Pixels
    with torch.no_grad():
        gaussians._opacity.clamp_(0, 0.99999)
        gaussians._opacity = gaussians.inverse_opacity_activation(gaussians._opacity)
        
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
    gaussian_indices = torch.zeros(gaussians._xyz.size(0)).int().cuda()
    
    #scene.dump_gaussians("Dump", only_leaves=True)

    gaussians.move_storage_to(Storage_Device, None, False, False, False)

    gaussians.build_hierarchical_SPT(SPT_Root_Volume, SPT_Target_Granularity, use_bounding_spheres=Use_Bounding_Spheres)
    print("Built SPTs")
    #scene.dump_gaussians("Dump", False, file_name="SPTNodes.ply", indices=spt_nodes.to(Storage_Device))
    #exit()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    #progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    indices = None

    iteration = first_iter
    # Dataloader loads data from disk
    # DONT SHUFFLE IF USING CORRESPONDENCE GRAPH

    
    if Gradient_Propagation:
        gaussians.recompute_weights()
    
    #means3D, opacity, scales, rotations, features_dc, features_rest, gaussian_indices = torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0, 1), device='cuda', dtype=torch.float32), torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0,4), device='cuda', dtype=torch.float32), torch.empty((0, 1, 3), device='cuda', dtype=torch.float32), torch.empty((0,15, 3), device='cuda', dtype=torch.float32), torch.empty(0, device='cuda', dtype=torch.int32)
    if Cache_SPTs:
        gaussian_indices = torch.arange(0, gaussians.skybox_points, device='cuda')
        means3D = gaussians.properties[:gaussians.skybox_points, xyz1:xyz2].cuda().contiguous()
        scales = gaussians.properties[:gaussians.skybox_points, scales1:scales2].cuda().contiguous()
        rotations = gaussians.properties[:gaussians.skybox_points, rotation1:rotation2].cuda().contiguous()
        features_dc = gaussians.properties[:gaussians.skybox_points, features1:features2].cuda().unsqueeze(1).contiguous()
        opacity = gaussians.properties[:gaussians.skybox_points, opacity1].cuda().unsqueeze(1).contiguous()
        features_rest = gaussians.properties[:gaussians.skybox_points, features_rest1: features_rest2].cuda().reshape(gaussians.skybox_points, 3, SH_properties_single).contiguous()
        
    else:
        means3D, opacity, scales, rotations, features_dc, features_rest, gaussian_indices = torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0, 1), device='cuda', dtype=torch.float32), torch.empty((0, 3), device='cuda', dtype=torch.float32), torch.empty((0,4), device='cuda', dtype=torch.float32), torch.empty((0, 1, 3), device='cuda', dtype=torch.float32), torch.empty((0,15, 3), device='cuda', dtype=torch.float32), torch.empty(0, device='cuda', dtype=torch.int32)
    
    
    
    
    prev_SPT_distances = torch.empty(0, dtype = torch.float32, device='cuda')
    prev_SPT_indices = torch.empty(0, dtype = torch.int32, device='cuda')
    prev_SPT_starts = torch.empty(0, dtype = torch.int32, device='cuda')    
    gaussians.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init*gaussians.spatial_lr_scale,
                                                    lr_final=opt.position_lr_final*gaussians.spatial_lr_scale,
                                                    lr_delay_mult=opt.position_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)
    if not replay:
        with open(f"CameraPaths/camera_path_{camera_path_id}.txt", "w") as f:
            f.write("")
    else: 
        cam_path_file = open(f"CameraPaths/camera_path_{camera_path_id}.txt", "rb")
    ######## VIEWER
    if not replay:
        network_gui.init("127.0.0.1", 6009)
    while True:
        if network_gui.conn == None and not replay:
            print("Try Connect")
            network_gui.try_connect()
        while network_gui.conn != None or replay:
            try:
                net_image_bytes = None
                if not replay:
                    custom_cam, do_training, keep_alive_, scaling_modifer, slider = network_gui.receive()
                    if "distance_multiplier" in slider:
                        distance_multiplier = slider["distance_multiplier"]
                    else:
                        distance_multiplier = 1.0
                    if "render_SPTs" in slider:
                        render_SPTs = slider["render_SPTs"] > 0
                        if render_SPTs:
                            Reuse_SPT_Tolerarance = 0.0
                    else:
                        render_SPTs = False
                    if "freeze_view" in slider:    
                        freeze_view = slider["freeze_view"] > 0
                    else:
                        freeze_view = False

                    if "distance_color" in slider:    
                        color_distance = slider["distance_color"] > 0
                    else:
                        color_distance = False
                    if "size_color" in slider and not color_distance:    
                        color_size = slider["size_color"] > 0
                    else:
                        color_size = False
                    if "reuse_SPT_tolerance" in slider and not render_SPTs:    
                        Reuse_SPT_Tolerarance = slider["reuse_SPT_tolerance"]
                    if "separate_SPTs" in slider and not render_SPTs:    
                        separate_SPTs = slider["separate_SPTs"] > 0
                    else:
                        separate_SPTs = False
                    if "highlight_leaves" in slider:    
                        highlight_leaves = slider["highlight_leaves"] > 0
                    else:
                        highlight_leaves = False
                    if "show_occlusion" in slider:    
                        show_occlusion = slider["show_occlusion"] > 0
                        Use_Occlusion_Culling = slider["show_occlusion"] > 0
                    else:
                        show_occlusion = False
                    if "record_traj" in slider:    
                        record_trajectory = slider["record_traj"] > 0
                    else: 
                        record_trajectory = False
                else:
                    distance_multiplier = 1.0
                    freeze_view = False
                    color_distance = False
                    color_size = False
                    separate_SPTs = False
                    highlight_leaves = False
                    show_occlusion = False
                    record_trajectory = False
                    custom_cam =  pickle.load(cam_path_file)
                if custom_cam != None:
                    ####### RENDER
                    viewpoint_cam = custom_cam
                    
                    if record_trajectory:
                        with open(f"CameraPaths/camera_path_{camera_path_id}.txt", "ab") as f:
                            pickle.dump(custom_cam, f)
                    
                    viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                    #viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                    viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                    viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                    if not freeze_view:

                        ############# SPT Cache

                        # FILTER PHASE BEGIN

                        render_metrics.stamp('filter')

                        if Use_Bounding_Spheres:
                            bounds = gaussians.bounding_sphere_radii
                        else: 
                            bounds = (gaussians.scaling_activation(torch.max(gaussians.upper_tree_scaling, dim=-1)[0]) * 3.0)
                        planes = gaussians.extract_frustum_planes(viewpoint_cam.full_proj_transform.cuda())
                        if Use_Frustum_Culling:
                            frustum_cull = lambda indices : gaussians.frustum_cull_spheres(gaussians.upper_tree_xyz[indices], bounds[indices], planes)
                        else:
                            frustum_cull = lambda indices : torch.ones(len(indices), dtype = torch.bool)
                        camera_position = viewpoint_cam.camera_center.cuda()
                        clock()
                        LOD_detail_cut = lambda indices : gaussians.min_distance_squared[indices] > (camera_position - gaussians.upper_tree_xyz[indices]).square().sum(dim=-1) * distance_multiplier
                        # The coarse cut contains intermediate nodes from the upper tree and leaf nodes, with some leaf nodes containing SPTs
                        coarse_cut = gaussians.cut_hierarchy_on_condition(gaussians.upper_tree_nodes, LOD_detail_cut, return_upper_tree=False, root_node=0, leave_out_of_cut_condition=frustum_cull)

                        if Use_Occlusion_Culling:
                            bg_color = [0, 0, 0]
                            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                            temp = len(coarse_cut)
                            occlusion_indices = gaussians.upper_tree_nodes[coarse_cut, 5]
                            occlusion_mask, occlusion_image = occlusion_cull(occlusion_indices.to(Storage_Device), gaussians, viewpoint_cam, pipe, background)
                            occlusion_mask = occlusion_mask.cuda()
                            coarse_cut = coarse_cut[occlusion_mask]
                            print(f"Occlusion Cull {temp - len(coarse_cut)} out of {temp} upper tree gaussians")

                        # leaf nodes have 0 children
                        cut_leaf_nodes = coarse_cut[gaussians.upper_tree_nodes[coarse_cut, 2] == 0]
                        # separate the cut into leafs that contain an SPT and those that don't
                        # The SPT indices are the child indices of those nodes that have a 0 child count
                        SPT_indices = gaussians.upper_tree_nodes[cut_leaf_nodes][gaussians.upper_tree_nodes[cut_leaf_nodes, 3] >= 0, 3]

                        upper_tree_nodes_to_render = gaussians.upper_tree_nodes[coarse_cut][gaussians.upper_tree_nodes[coarse_cut, 3] <= 0, 5]

                        SPT_upper_tree_indices = cut_leaf_nodes[gaussians.upper_tree_nodes[cut_leaf_nodes, 3] >= 0]

                        SPT_distances = (gaussians.upper_tree_xyz[SPT_upper_tree_indices] - camera_position).pow(2).sum(1).sqrt() * distance_multiplier

                        ### Band Aid Fix
                        #if len(SPT_indices) == 0:
                            # Just load whatever is already in memory
                            #SPT_indices = torch.zeros(1, dtype=torch.int32, device='cuda')
                            #if prev_SPT_indices.size(0) > 0:
                            #    SPT_indices[0] = prev_SPT_indices[-1]
                            #else:
                            #    # Or just load the first one, whatever
                            #    SPT_indices[0] = torch.zeros(1, dtype=torch.int32, device='cuda')
                            #SPT_distances = torch.zeros(1, dtype=torch.float32, device='cuda')
                            #SPT_distances[0] = 100000.0
                            #print("No SPT in image")
                            #clock()
                            
                        ### Band Aid Fix
                        

                        prev_to_new_SPT_order = torch.searchsorted(SPT_indices, prev_SPT_indices)


                        # Clamp indices to valid range [0, len(SPT_indices) - 1]
                        clamped_indices = prev_to_new_SPT_order.clamp(0, len(SPT_indices) - 1)

                        # Build mask: only trust indices that are in bounds and match
                        valid_range_mask = (prev_to_new_SPT_order >= 0) & (prev_to_new_SPT_order < len(SPT_indices))
                        if SPT_indices.numel() > 0:
                            # Clamp and check validity
                            clamped_indices = prev_to_new_SPT_order.clamp(0, len(SPT_indices) - 1)
                            valid_range_mask = (prev_to_new_SPT_order >= 0) & (prev_to_new_SPT_order < len(SPT_indices))

                            # Safe comparison
                            equal_SPT_cache_mask = valid_range_mask & (SPT_indices[clamped_indices] == prev_SPT_indices)
                        else:
                            # Nothing to compare against
                            equal_SPT_cache_mask = torch.zeros_like(prev_SPT_indices, dtype=torch.bool)

                        #equal_SPT_cache_mask = valid_range_mask & (SPT_indices[clamped_indices] == prev_SPT_indices)
                        #index -1 access sometimes..
                        #equal_SPT_cache_mask = (prev_to_new_SPT_order < len(SPT_indices)) & (SPT_indices[prev_to_new_SPT_order.clamp_max(len(SPT_indices)-1)] == prev_SPT_indices)
                        prev_equal_SPT_cache_indices = torch.nonzero(equal_SPT_cache_mask, as_tuple=True)[0]
                        equal_SPT_cache_indices = prev_to_new_SPT_order[equal_SPT_cache_mask]

                        prev_distances_compare = prev_SPT_distances[prev_equal_SPT_cache_indices]
                        distances_compare = SPT_distances[equal_SPT_cache_indices]
                        #close_enough = torch.isclose(distances_compare, prev_distances_compare, rtol=Reuse_SPT_Tolerarance, atol=0.05)
                        close_enough = (prev_distances_compare/distances_compare) > 0.3
                        close_enough &= (prev_distances_compare/distances_compare) < 1.5
                        reuse_SPT_indices = SPT_indices[equal_SPT_cache_indices[close_enough]]


                        prev_keep_SPT_cache_indices = prev_equal_SPT_cache_indices[close_enough]

                        # Keep all the gaussians that are contained in an STP that is reused and close enough
                        # Cumulative Sum Trick
                        reuse_gaussians_mask = torch.zeros(len(gaussian_indices)+1, dtype=torch.int32, device='cuda')
                        debug_x = prev_SPT_starts[prev_keep_SPT_cache_indices]
                        if len(debug_x) > 0:
                            if torch.max(debug_x, 0)[0].item() > len(reuse_gaussians_mask):
                                print("Debug X is bigger than reuse_gaussians_mask")
                            if torch.min(debug_x, 0)[0].item() < 0:
                                print("Debug X is less than 0")
                        reuse_gaussians_mask[debug_x] += 1
                        debug_y = prev_SPT_starts[prev_keep_SPT_cache_indices+1]
                        reuse_gaussians_mask[debug_y] -= 1
                        reuse_gaussians_mask = reuse_gaussians_mask.cumsum(0)[:-1].bool()

                        # Keep Skybox
                        #keep_gaussians_mask[:gaussians.skybox_points] = True

                        #mask = torch.isin(SPT_indices, keep_SPT_indices)

                        load_SPT_mask = torch.zeros(len(SPT_indices), device='cuda', dtype=torch.bool)
                        load_SPT_mask.scatter_(0, equal_SPT_cache_indices[close_enough].to(torch.int64), True)      
                        load_SPT_mask = ~load_SPT_mask                  
                        load_SPT_indices = SPT_indices[load_SPT_mask]
                        load_SPT_distances = SPT_distances[load_SPT_mask]

                        if len(load_SPT_indices) > 0:
                            #LOAD SPT CUT
                            #load_SPT_distances = torch.full((len(load_SPT_indices),), 10000.0).cuda()
                            load_SPT_gaussian_indices, load_SPT_starts = get_spt_cut_cuda(len(load_SPT_indices), gaussians.SPT_gaussian_indices, gaussians.SPT_starts, gaussians.SPT_max, gaussians.SPT_min, load_SPT_indices, load_SPT_distances)
                        else:
                            print("No SPTs loaded")
                            load_SPT_gaussian_indices, load_SPT_starts = torch.empty(0, dtype=torch.int32, device='cuda'), torch.empty(0, dtype=torch.int32, device='cuda')
                        
                        # FLAG
                        
                        print(f'timestamp:{clock()} | len(load_SPT_indices)={len(load_SPT_indices)} | len(SPT_indices)={len(SPT_indices)} | ')
                        #SPT_counts += gaussians.skybox_points

                        ### BAND AID FIX
                        #difference = load_SPT_starts[1:] - load_SPT_starts[:-1]
                        #empty_SPTs = torch.where(difference == 0)[0]
                        #if len(empty_SPTs) > 0:
                        #    print(f"Empty SPTs {empty_SPTs} encountered")
                        #    #mask = torch.ones(len(load_SPT_starts), dtype=torch.bool, device='cuda')
                        #    #mask.scatter_(0, empty_SPTs, False)
                        #    #load_SPT_starts = load_SPT_starts[mask]
                        #    #load_SPT_distances = load_SPT_distances[mask]
                        #    #load_SPT_indices = load_SPT_indices[mask]
                        if len(load_SPT_starts) > 0:    
                            if len(load_SPT_gaussian_indices) == load_SPT_starts[-1]:
                                print("Last SPT empty")
                                load_SPT_starts = load_SPT_starts[:-1]
                                load_SPT_distances = load_SPT_distances[:-1]
                                load_SPT_indices = load_SPT_indices[:-1]
                        #    
                        ### BAND AID FIX

                        assert(len(load_SPT_starts.unique()) == len(load_SPT_starts))
                        #cache_SPT_cache_indices = torch.where(~equal_SPT_cache_mask)[0]    
                        #cache_SPT_indices = prev_SPT_indices[cache_SPT_cache_indices]
                        SPT_indices = torch.cat((load_SPT_indices, reuse_SPT_indices))


                        SPT_starts_new = torch.zeros(len(load_SPT_indices) + len(reuse_SPT_indices) + 1,dtype=torch.int32, device='cuda')
                        # compact the prefix sum of SPT_counts

                        SPT_starts_new[:len(load_SPT_starts)] = load_SPT_starts + gaussians.skybox_points
                        SPT_starts_new[len(load_SPT_starts)] = len(load_SPT_gaussian_indices) + gaussians.skybox_points
                        #prefix = len(cut_SPTs) + gaussians.skybox_points
                        #for index, i in enumerate(SPT_keep_counts_indices):
                        #    SPT_starts_new[index + len(SPT_counts)] = prefix
                        #    prefix += (prev_SPT_counts[i+1] - prev_SPT_counts[i]).item()

                        sizes = prev_SPT_starts[prev_keep_SPT_cache_indices + 1] - prev_SPT_starts[prev_keep_SPT_cache_indices]
                        SPT_starts_new[len(load_SPT_starts) + 1:len(load_SPT_starts) + 1 + len(sizes)] = torch.cumsum(sizes, dim=0) +  len(load_SPT_gaussian_indices) + gaussians.skybox_points

                        number_of_gaussians_to_render = SPT_starts_new[len(load_SPT_starts) + len(sizes)]


                        SPT_distances = torch.cat((load_SPT_distances, prev_SPT_distances[prev_keep_SPT_cache_indices]))

                        load_from_disk_indices = torch.cat((upper_tree_nodes_to_render, load_SPT_gaussian_indices))



                        gaussian_indices = torch.cat((gaussian_indices[:gaussians.skybox_points], load_from_disk_indices, gaussian_indices[reuse_gaussians_mask]))
                        print(f"Load Percent: {len(load_from_disk_indices) * 100/ number_of_gaussians_to_render}")
                        load_from_disk_indices = load_from_disk_indices.to(Storage_Device)
                        SPT_starts_new += len(upper_tree_nodes_to_render)

                        assert(SPT_starts_new[-1] == len(gaussian_indices))
                        number_to_render = len(gaussian_indices)
                        distance_multiplier = distance_multiplier
                    
                    # LOAD PHASE BEGIN

                        render_metrics.stamp('load')

                        load_tensor = gaussians.properties[load_from_disk_indices, :].cuda(non_blocking=non_blocking)

                        means3D = nn.Parameter(torch.cat((means3D[:gaussians.skybox_points], load_tensor[:, xyz1:xyz2].cuda(non_blocking=non_blocking), means3D[reuse_gaussians_mask])).contiguous())
                        opacity = nn.Parameter(torch.cat((opacity[:gaussians.skybox_points], load_tensor[:, opacity1:opacity2].cuda(non_blocking=non_blocking), opacity[reuse_gaussians_mask])).contiguous())
                        scales = nn.Parameter(torch.cat((scales[:gaussians.skybox_points], load_tensor[:, scales1:scales2].cuda(non_blocking=non_blocking), scales[reuse_gaussians_mask])).contiguous())
                        rotations = nn.Parameter(torch.cat((rotations[:gaussians.skybox_points], load_tensor[:, rotation1:rotation2].cuda(non_blocking=non_blocking), rotations[reuse_gaussians_mask])).contiguous())
                        # TODO: ABS?
                        features_dc = nn.Parameter(torch.cat((features_dc[:gaussians.skybox_points], load_tensor[:, features1:features2].cuda(non_blocking=non_blocking).unsqueeze(1), features_dc[reuse_gaussians_mask])).contiguous())
                        features_rest = nn.Parameter(torch.cat((features_rest[:gaussians.skybox_points], load_tensor[:, features_rest1:features_rest2].cuda(non_blocking=non_blocking).reshape(len(load_tensor), 3, SH_properties_single), features_rest[reuse_gaussians_mask])).contiguous())


                        prev_SPT_indices = SPT_indices
                        prev_SPT_distances = SPT_distances
                        prev_SPT_starts = SPT_starts_new
                        torch.cuda.empty_cache()
                        
                        
                    
                    # RENDER PHASE BEGIN

                    render_metrics.stamp('render')
                    
                     # Render
                    if Rasterizer == "Hierarchical":
                        render_pkg = render_on_disk(
                        viewpoint_cam, 
                        means3D,
                        gaussians.opacity_activation(opacity),
                        gaussians.scaling_activation(scales), 
                        gaussians.rotation_activation(rotations),
                        shs,
                        pipe, 
                        background,
                        sh_degree = gaussians.active_sh_degree)
                    elif Rasterizer == "Vanilla":
                        if color_distance:
                            colors_precomp = torch.zeros_like(scales, device='cuda')
                            SPT_distances_remapped = (SPT_distances - SPT_distances.min()) / (SPT_distances.max() - SPT_distances.min())
                            # Skybox is blue
                            colors_precomp[:gaussians.skybox_points, 0] = 0
                            colors_precomp[:gaussians.skybox_points, 1] = 0
                            colors_precomp[:gaussians.skybox_points, 2] = 1
                            for i in range(len(SPT_indices)):
                                min_range = gaussians.skybox_points + SPT_starts_new[i]
                                max_range = len(gaussian_indices) if i+1 == len(SPT_indices) else gaussians.skybox_points + SPT_starts_new[i+1]
                                colors_precomp[min_range:max_range, 0] = SPT_distances_remapped[i]
                                colors_precomp[min_range:max_range, 1] = SPT_distances_remapped[i]
                                colors_precomp[min_range:max_range, 2] = SPT_distances_remapped[i]
                        elif color_size:
                            colors_precomp = torch.zeros_like(scales, device='cuda')
                            SPT_sizes = SPT_starts_new
                            SPT_sizes_remapped = (SPT_sizes - SPT_sizes.min()) / (SPT_sizes.max() - SPT_sizes.min())
                            # Skybox is blue
                            colors_precomp[:gaussians.skybox_points, 0] = 0
                            colors_precomp[:gaussians.skybox_points, 1] = 0
                            colors_precomp[:gaussians.skybox_points, 2] = 1
                            for i in range(len(SPT_indices)):
                                min_range = gaussians.skybox_points + SPT_starts_new[i]
                                max_range = len(gaussian_indices) if i+1 == len(SPT_indices) else gaussians.skybox_points + SPT_starts_new[i+1]
                                colors_precomp[min_range:max_range, 0] = SPT_sizes_remapped[i]
                                colors_precomp[min_range:max_range, 1] = SPT_sizes_remapped[i]
                                colors_precomp[min_range:max_range, 2] = SPT_sizes_remapped[i]
                        elif separate_SPTs:
                            colors_precomp = torch.zeros_like(scales, device='cuda')
                            # Skybox is blue
                            colors_precomp[:gaussians.skybox_points, 0] = 0
                            colors_precomp[:gaussians.skybox_points, 1] = 0
                            colors_precomp[:gaussians.skybox_points, 2] = 1
                            random_colors = torch.rand((len(SPT_indices), 3))
                            for i in range(len(SPT_indices)):
                                min_range = gaussians.skybox_points + SPT_starts_new[i]
                                max_range = len(gaussian_indices) if i+1 == len(SPT_indices) else gaussians.skybox_points + SPT_starts_new[i+1]
                                colors_precomp[min_range:max_range, :] = random_colors[i]
                            colors_precomp[-len(upper_tree_nodes_to_render):, :] = 1
                        elif highlight_leaves:
                            colors_precomp = torch.zeros_like(scales, device='cuda')
                            # Skybox is blue
                            colors_precomp[:, 0] = 1
                            leaf_mask = gaussians.nodes[gaussian_indices.cpu(), 2] == 0
                            colors_precomp[leaf_mask.cuda(), :] = 1
                        else:
                            colors_precomp = None
                        render_pkg = render_vanilla(
                            viewpoint_cam, 
                            means3D,
                            gaussians.opacity_activation(opacity),
                            gaussians.scaling_activation(scales), 
                            gaussians.rotation_activation(rotations),
                            features_dc,
                            features_rest,
                            pipe, 
                            background,
                            #splat_args=splat_settings,
                            override_color = colors_precomp, 
                            sh_degree = gaussians.active_sh_degree,
                            )
                    else:
                        render_pkg = render_stp(
                            viewpoint_cam, 
                            means3D,
                            gaussians.opacity_activation(opacity),
                            gaussians.scaling_activation(scales), 
                            gaussians.rotation_activation(rotations),
                            shs,
                            pipe, 
                            background,
                            splat_args=splat_settings,
                            sh_degree = gaussians.active_sh_degree,
                            )


                    image = render_pkg["render"]
                    ####### RENDER
                    if show_occlusion:
                        image=occlusion_image
                    
                    
                    if replay:
                        torchvision.utils.save_image(image, f"CameraPaths/{camera_path_id}/frame_{iteration}.png")
                        iteration += 1
                    else:
                        net_image = image.cpu()
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().to('cpu').numpy())
                if not replay:
                    train_params = {"Num_Rendered" : len(gaussian_indices),
                                    "Number_of_SPTs" : len(SPT_indices), 
                                    "Percentage_Rendered" : len(gaussian_indices)/gaussians.size,
                                    "Percentage_SPTs" : len(SPT_indices)/len(gaussians.SPT_starts),
                                    "TEST": 420.69}
                    # ADD STATS HERE
                    render_metrics.stamp('do you know the tragedy of darth plaguise the wise?')

                    # print(render_metrics.get_timings())
                    # print mean timings over last 10 iterations
                    print(render_metrics.get_means())

                    network_gui.send(net_image_bytes, json.dumps({"iteration" : 99, "num_gaussians" : gaussians.size, "loss" : 0, "sh_degree":1, "error" : 0, "paused" : False, "train_params" : train_params})) #dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive_):
                        break
                
                
            except ValueError as e:
                print(e)
                raise e
                network_gui.conn = None
######## VIEWER

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--hierarchy_path", type=str, default = None)
    parser.add_argument('--replay', type=bool, default=False)
    parser.add_argument('--ID', type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    print(args)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    render(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.hierarchy_path, args.replay, args.ID)

    print("\nTraining complete.")
