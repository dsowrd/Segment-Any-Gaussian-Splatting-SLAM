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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            sh_objs = pc.get_objects
    else:
        colors_precomp = override_color
    
    # print(means3D)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    depth_image, rendered_image, radii, rendered_objects, is_used = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        sh_objs = sh_objs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # torch.cuda.empty_cache()
    # print(colors_precomp)
    # print(means3D)
    # print(radii)
    # print(rendered_image)
    # print(depth_image)
    # print(radii)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # print(depth_image.shape, rendered_image.shape)
    return {"render": rendered_image,
            "render_depth": depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "render_object": rendered_objects,
            "is_used": is_used,
            }

def render_2(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           training_stage=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if training_stage==0:
        resolution_width = int(viewpoint_camera.image_width)
        resolution_height = int(viewpoint_camera.image_height)
    else:
        resolution_width = int(viewpoint_camera.image_width/(training_stage*2))
        resolution_height = int(viewpoint_camera.image_height/(training_stage*2))
    
    raster_settings = GaussianRasterizationSettings(
        image_height=resolution_height,
        image_width=resolution_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # print(means3D)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    depth_image, rendered_image, radii, is_used = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # torch.cuda.empty_cache()
    # print(colors_precomp)
    # print(means3D)
    # print(radii)
    # print(rendered_image)
    # print(depth_image)
    # print(radii)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # print(depth_image.shape, rendered_image.shape)
    return {"render": rendered_image,
            "render_depth": depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "is_used": is_used,
            }

def render_3(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           training_stage=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Set up rasterization configuration
    tanfovx = math.tan(float(viewpoint_camera.FoVx[0]) * 0.5)
    tanfovy = math.tan(float(viewpoint_camera.FoVy[0]) * 0.5)

    if training_stage==0:
        resolution_width = int(viewpoint_camera.image_width[0])
        resolution_height = int(viewpoint_camera.image_height[0])
    else:
        resolution_width = int(viewpoint_camera.image_width[0]/(training_stage*2))
        resolution_height = int(viewpoint_camera.image_height[0]/(training_stage*2))
    
    raster_settings = GaussianRasterizationSettings(
        image_height=resolution_height,
        image_width=resolution_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # print(means3D)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    depth_image, rendered_image, radii, is_used = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # torch.cuda.empty_cache()
    # print(colors_precomp)
    # print(means3D)
    # print(radii)
    # print(rendered_image)
    # print(depth_image)
    # print(radii)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # print(depth_image.shape, rendered_image.shape)
    return {"render": rendered_image,
            "render_depth": depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "is_used": is_used,
            }

def render_4(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           training_stage=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # print("=== Entering render_4 ===")
    # print("Function arguments:")
    # print(f"  viewpoint_camera: {viewpoint_camera}")
    # if hasattr(viewpoint_camera, "image_width"):
    #     print(f"    image_width: {viewpoint_camera.image_width}")
    # if hasattr(viewpoint_camera, "image_height"):
    #     print(f"    image_height: {viewpoint_camera.image_height}")
    # if hasattr(viewpoint_camera, "FoVx"):
    #     print(f"    FoVx: {viewpoint_camera.FoVx}")
    # if hasattr(viewpoint_camera, "FoVy"):
    #     print(f"    FoVy: {viewpoint_camera.FoVy}")
    # if hasattr(viewpoint_camera, "camera_center"):
    #     print(f"    camera_center shape: {viewpoint_camera.camera_center.shape}")
    #     print(f"    camera_center: {viewpoint_camera.camera_center}")

    # print(f"  pc: {pc}")  # Typically prints the object class; if you want details, print pc.__dict__ or check shapes below
    # print(f"  pipe: {pipe}")
    # print(f"  bg_color shape: {bg_color.shape}, device={bg_color.device}, dtype={bg_color.dtype}")
    # print(f"  scaling_modifier: {scaling_modifier}")
    # print(f"  override_color: {override_color if override_color is not None else 'None'}")
    # print(f"  training_stage: {training_stage}")
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # # Print shapes for debugging
    # print("=== render_4 Debug Info ===")
    # print(f"pc.get_xyz shape: {pc.get_xyz.shape}")
    # print(f"screenspace_points shape: {screenspace_points.shape}")
    
    # Set up rasterization configuration
    tanfovx = math.tan(float(viewpoint_camera.FoVx[0]) * 0.5)
    tanfovy = math.tan(float(viewpoint_camera.FoVy[0]) * 0.5)

    if training_stage==0:
        resolution_width = int(viewpoint_camera.image_width[0])
        resolution_height = int(viewpoint_camera.image_height[0])
    else:
        resolution_width = int(viewpoint_camera.image_width[0]/(training_stage*2))
        resolution_height = int(viewpoint_camera.image_height[0]/(training_stage*2))

    # print(f"Resolution: {resolution_width} x {resolution_height}")
    
    raster_settings = GaussianRasterizationSettings(
        image_height=resolution_height,
        image_width=resolution_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # print(f"means3D shape: {means3D.shape}")
    # print(f"means2D shape: {means2D.shape}")
    # print(f"opacity shape: {opacity.shape}")

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        # print(f"cov3D_precomp shape: {cov3D_precomp.shape if cov3D_precomp is not None else None}")
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        # if scales is not None:
        #     print(f"scales shape: {scales.shape}")
        # if rotations is not None:
        #     print(f"rotations shape: {rotations.shape}")

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # print(f"shs_view shape: {shs_view.shape}")
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            # print(f"colors_precomp shape: {colors_precomp.shape}")
        else:
            shs = pc.get_features
            sh_objs = pc.get_objects
            # print(f"shs shape: {shs.shape}")
            # print(f"sh_objs shape: {sh_objs.shape}")
    else:
        colors_precomp = override_color
        # print(f"override_color shape (if tensor): "
        #       f"{colors_precomp.shape if isinstance(colors_precomp, torch.Tensor) else 'Not a tensor'}")

    
    # print(means3D)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    _, rendered_image, radii, rendered_objects, is_used = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        sh_objs = sh_objs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # torch.cuda.empty_cache()
    # print(colors_precomp)
    # print(means3D)
    # print(radii)
    # print(rendered_image)
    # print(depth_image)
    # print(radii)
    # print(f"depth_image shape: {depth_image.shape}")
    # print(f"rendered_image shape: {rendered_image.shape}")
    # print(f"radii shape: {radii.shape}")
    # print(f"rendered_objects shape: {rendered_objects.shape}")
    # print(f"is_used shape: {is_used.shape}")

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # print(depth_image.shape, rendered_image.shape)
    return {"render": rendered_image,
            # "render_depth": depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "render_object": rendered_objects,
            "is_used": is_used
            }