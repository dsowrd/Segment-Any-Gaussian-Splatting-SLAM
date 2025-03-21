#!/usr/bin/env python3
import sys
import os
import rospy
import time
import random
import numpy as np
import struct
import cv2
import torch
import math
import threading

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ROSImage, PointCloud2
from nav_msgs.msg import Odometry
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import rerun as rr
import message_filters
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import colorsys

sys.path.append('/home/irl/Workspace_Hyundo/catkin_ws/src/gaussian_splatting/gaussian_splatting')
sys.path.append('/home/irl/Workspace_Hyundo/catkin_ws/src/gaussian_splatting/MobileSAM/MobileSAMv2')
from arguments import SLAMParameters
from scene import GaussianModel
from scene.shared_objs import SharedCam
from utils.graphics_utils import focal2fov
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
from gaussian_renderer import render_2

from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor
from mobilesamv2.utils.transforms import ResizeLongestSide

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image

def read_points_direct(cloud_msg):
    point_dtype = np.dtype([('x', np.float32),
                            ('y', np.float32),
                            ('z', np.float32),
                            ('rgb', np.float32)])
    num_points = cloud_msg.width * cloud_msg.height
    if cloud_msg.point_step != point_dtype.itemsize:
        new_dtype = np.dtype({
            'names': ['x', 'y', 'z', 'rgb'],
            'formats': [np.float32, np.float32, np.float32, np.float32],
            'offsets': [0, 4, 8, 16],
            'itemsize': cloud_msg.point_step
        })
    else:
        new_dtype = point_dtype
    return np.frombuffer(cloud_msg.data, dtype=new_dtype, count=num_points).copy()

def read_xyz_rgb_from_raw(cloud_msg, num_points=None):
    pts_struct = read_points_direct(cloud_msg)
    points = np.vstack((pts_struct['x'], pts_struct['y'], pts_struct['z'])).T
    rgb_int = pts_struct['rgb'].view(np.uint32)
    r = ((rgb_int >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((rgb_int >> 8) & 0xFF).astype(np.float32) / 255.0
    b = (rgb_int & 0xFF).astype(np.float32) / 255.0
    colors = np.vstack((r, g, b)).T

    if num_points is not None:
        points = points[:num_points]
        colors = colors[:num_points]
    return points, colors

# def filter_new_points(new_points, new_colors, existing_points, distance_threshold):
#     if existing_points.size == 0:
#         return new_points, new_colors
#     tree = cKDTree(existing_points)
#     distances, _ = tree.query(new_points, k=1)
#     mask = distances > distance_threshold
#     return new_points[mask], new_colors[mask]

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug

class GaussianSplatting(SLAMParameters):
    def __init__(self):
        super().__init__()
        rospy.init_node('gaussian_splatting', anonymous=True)
        self.bridge = CvBridge()
        self.saved_eval_scenes = []
        self.synced_data = None
        self.last_pose = None
        self.frame_count = 0
        self.iteration_images = 0
        self.translation_path = []
        self.total_start_time = time.time()
        self.scene_extent = 2.5
        self.prune_th = 2.5
        self.gaussians = GaussianModel(self.sh_degree)
        self.pipe = Pipe(self.convert_SHs_python, self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.train_iter = 0

        self.scale = 0.5
        # Retail_Street, CBD_Building_01
        self.fx = 1293.56944 * self.scale
        self.fy = 1293.3155 * self.scale
        self.cx = 626.91359 * self.scale
        self.cy = 522.799224 * self.scale
        # # HKU_Campus, CBD_Building_02
        # self.fx = 1176.2874292149932 * self.scale
        # self.fy = 1176.21585445307 * self.scale
        # self.cx = 592.1187382755453 * self.scale
        # self.cy = 509.0864309628322 * self.scale
        # # SYSU_01, HIT_Graffiti_Wall_01
        # self.fx = 1311.89517127580 * self.scale
        # self.fy = 1311.36488586115 * self.scale
        # self.cx = 656.523841857393 * self.scale
        # self.cy = 504.136322840350 * self.scale
        # # Red_Sculpture
        # self.fx = 1294.7265716372897 * self.scale
        # self.fy = 1294.8678078910468 * self.scale
        # self.cx = 626.663267153558 * self.scale
        # self.cy = 531.0334324363173 * self.scale
        self.cam_intrinsic = np.array([[self.fx, 0, self.cx],
                                       [0, self.fy, self.cy],
                                       [0, 0, 1]])
        self.W = int(1280 * self.scale)
        self.H = int(1024 * self.scale)

        self.mapping_new_cams = []
        self.mapping_cams = []

        # Setup message_filters for synchronization.
        img_sub = message_filters.Subscriber("/rgb_img", ROSImage)
        cloud_sub = message_filters.Subscriber("/cloud_registered", PointCloud2)
        odom_sub = message_filters.Subscriber("/aft_mapped_to_init", Odometry)
        self.slop = 0.1  # adjust as needed
        self.ts = message_filters.ApproximateTimeSynchronizer([img_sub, cloud_sub, odom_sub],
                                                               queue_size=10,
                                                               slop=self.slop)
        self.ts.registerCallback(self.synced_callback)

        # Track the time of the last received topic and whether we've received the first topic.
        self.last_topic_time = time.time()
        self.first_topic_received = False

        self.scene_threshold = 0.7
        self.post_train_iter = 0
        self.add_gaussians_only = False

        self.output_dir = "/home/irl/Workspace_Hyundo/catkin_ws/saved_scenes_images"

        # Initialize rerun visualization.
        rr.init("3dgsviewer")
        rr.spawn(connect=False)
        rr.connect()

    def synced_callback(self, image_msg, cloud_msg, odom_msg):
        # Update the time and flag for the first topic.
        self.last_topic_time = time.time()
        if not self.first_topic_received:
            self.first_topic_received = True

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # Use full resolution for visualization and downstream processing.
            image_resized = cv2.resize(cv_image_rgb, (self.W, self.H))

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
            return

        pts, cols = read_xyz_rgb_from_raw(cloud_msg)
        synced = {
            "image": image_resized,  # ground truth image
            "points": np.array(pts, dtype=np.float32),
            "colors": np.array(cols, dtype=np.float32),
            "pose": odom_msg.pose.pose,
            "trained": False,  # mark as not yet processed
        }
        
        self.synced_data = synced

        # # Save scene if movement exceeds thresholds.
        # current_pose = odom_msg.pose.pose
        # save_scene = False
        # if not self.synced_data:
        #     self.last_pose = current_pose
        #     save_scene = True
        # else:
        #     last_pose = self.last_pose
        #     current_translation = np.array([current_pose.position.x,
        #                                     current_pose.position.y,
        #                                     current_pose.position.z])
        #     last_translation = np.array([last_pose.position.x,
        #                                  last_pose.position.y,
        #                                  last_pose.position.z])
        #     diff_translation = np.linalg.norm(current_translation - last_translation)
        #     current_quat = np.array([current_pose.orientation.x,
        #                              current_pose.orientation.y,
        #                              current_pose.orientation.z,
        #                              current_pose.orientation.w])
        #     last_quat = np.array([last_pose.orientation.x,
        #                           last_pose.orientation.y,
        #                           last_pose.orientation.z,
        #                           last_pose.orientation.w])
        #     rel_rot = Rotation.from_quat(last_quat).inv() * Rotation.from_quat(current_quat)
        #     diff_angle = np.linalg.norm(rel_rot.as_rotvec())
        #     if diff_translation > self.scene_threshold or diff_angle > self.scene_threshold:
        #         save_scene = True
        #     self.last_pose = current_pose

        self.frame_count += 1
        if self.frame_count % 10 == 0:
            save_scene = True
            self.frame_count = 0
        else:
            save_scene = False

        if save_scene:
        # Create a SharedCam for rendering.
            viewpoint_cam = SharedCam(
                FoVx=focal2fov(self.fx, self.W),
                FoVy=focal2fov(self.fy, self.H),
                image=synced["image"],
                cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy
            )

            pose = synced["pose"]
            translation = np.array([pose.position.x,
                                    pose.position.y,
                                    pose.position.z])
            rotation = Rotation.from_quat([pose.orientation.x,
                                            pose.orientation.y,
                                            pose.orientation.z,
                                            pose.orientation.w])
            R_cam = rotation.as_matrix()
            t_cam = translation
            Rz_90 = np.array([[0,  1, 0],
                                [-1, 0, 0],
                                [0,  0, 1]], dtype=float)
            Rx_minus_90 = np.array([[1, 0, 0],
                                    [0, 0, 1],
                                    [0, -1, 0]], dtype=float)
            new_rotation = R_cam @ (Rz_90 @ Rx_minus_90)
            new_translation = t_cam
            
            viewpoint_cam.on_cuda()
            viewpoint_cam.setup_cam(new_rotation, new_translation, synced["image"])

            self.mapping_new_cams.append(viewpoint_cam)
            self.saved_eval_scenes.append(synced)
            self.add_gaussians_only = False
            # print("New scene saved.")
        else:
            self.saved_eval_scenes.append(synced)
            self.add_gaussians_only = True
            # print("New scene added to evaluation.")


    def render_saved_scene(self, scene):
        """
        Re-render a saved scene using its stored parameters.
        Returns the rendered image as a numpy array.
        """
        image_np = scene["image"]
        pose = scene["pose"]

        translation = np.array([pose.position.x,
                                pose.position.y,
                                pose.position.z])
        rotation = Rotation.from_quat([pose.orientation.x,
                                       pose.orientation.y,
                                       pose.orientation.z,
                                       pose.orientation.w])
        R_cam = rotation.as_matrix()
        t_cam = translation
        Rz_90 = np.array([[0,  1, 0],
                          [-1, 0, 0],
                          [0,  0, 1]], dtype=float)
        Rx_minus_90 = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]], dtype=float)
        new_rotation = R_cam @ (Rz_90 @ Rx_minus_90)
        new_translation = t_cam

        # Create a SharedCam for rendering.
        viewpoint_cam = SharedCam(
            FoVx=focal2fov(self.fx, self.W),
            FoVy=focal2fov(self.fy, self.H),
            image=image_np,
            cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy
        )
        viewpoint_cam.on_cuda()
        viewpoint_cam.setup_cam(new_rotation, new_translation, image_np)

        # Render the scene.
        render_pkg = render_2(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
        image_rendered = render_pkg["render"]

        # Convert the rendered image to a numpy uint8 image.
        rendered_np = image_rendered.detach().cpu().numpy().transpose(1, 2, 0)
        rendered_np = np.clip(rendered_np, 0, 1) * 255
        rendered_np = rendered_np.astype(np.uint8)
        return rendered_np, image_rendered

    def save_scene_images(self):
        """After termination, re-render each saved scene and display both ground truth and rendered images in one window."""
        for idx, scene in enumerate(self.mapping_cams):
            # Get the ground truth image.
            if "image" in scene:
                gt_image = scene.original_image.cuda()
                # Convert from RGB to BGR for OpenCV display.
                gt_bgr = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
            else:
                rospy.logwarn("No ground truth image in scene %d", idx)
                continue

            # Render the scene after termination.
            rendered_np, _ = self.render_saved_scene(scene)
            
            # Concatenate the ground truth and rendered images horizontally.
            combined = np.hstack([gt_bgr, rendered_np])
            
            # Show the combined image in one window.
            window_name = f"Scene {idx:03d} - GT (left) | Rendered (right)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, combined)
            rospy.loginfo("Displaying scene %d. Press any key to continue.", idx)
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)

    def calc_2d_metric(self):
        psnrs = []
        ssims = []
        lpips = []

        cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to("cuda")

        """After termination, re-render each saved scene and display both ground truth and rendered images in one window."""
        for idx, scene in enumerate(tqdm(self.saved_eval_scenes)):
            if idx % 100 == 0:
                # Get the ground truth image.
                if "image" in scene:
                    gt_rgb = scene["image"]
                    gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
                    gt_rgb = gt_rgb/255
                    gt_rgb_ = torch.from_numpy(gt_rgb).float().cuda().permute(2,0,1)
                else:
                    rospy.logwarn("No ground truth image in scene %d", idx)
                    continue

                # Render the scene after termination.
                _, ours_rgb_ = self.render_saved_scene(scene)
                ours_rgb_ = torch.clamp(ours_rgb_, 0., 1.).cuda()
                
                square_error = (gt_rgb_-ours_rgb_)**2
                mse_error = torch.mean(torch.mean(square_error, axis=2))
                psnr = mse2psnr(mse_error)

                psnrs += [psnr.detach().cpu()]
                _, ssim_error = ssim(ours_rgb_, gt_rgb_)
                ssims += [ssim_error.detach().cpu()]
                lpips_value = cal_lpips(gt_rgb_.unsqueeze(0), ours_rgb_.unsqueeze(0))
                lpips += [lpips_value.detach().cpu()]
            
        psnrs = np.array(psnrs)
        ssims = np.array(ssims)
        lpips = np.array(lpips)
            
        print(f"PSNR: {psnrs.mean():.2f}\nSSIM: {ssims.mean():.3f}\nLPIPS: {lpips.mean():.3f}")

    def feature_to_rgb(self, features):
        # features: (C, H, W) -> reshape to (H*W, C)
        C, H, W = features.shape
        features_reshaped = features.view(C, -1).T  # shape: (H*W, C)
        
        # torch.pca_lowrank returns (U, S, V) such that X ~ U * diag(S) * V^T
        # 여기서 V의 첫 3열을 사용하여 3차원으로 투영
        U, S, V = torch.pca_lowrank(features_reshaped, q=3)
        proj = features_reshaped @ V[:, :3]  # shape: (H*W, 3)
        
        # reshape to (H, W, 3)
        proj = proj.reshape(H, W, 3)
        
        # normalize to [0, 255]
        proj_min = proj.min()
        proj_max = proj.max()
        proj_norm = 255 * (proj - proj_min) / (proj_max - proj_min + 1e-6)
        rgb_array = proj_norm.to(torch.uint8).cpu().numpy()
        
        return rgb_array
    
    def visualize_obj(self, objects):
        # objects: 2D numpy array (uint8) with object IDs
        # 미리 0~max_num_obj까지의 색상 LUT를 생성 (예, 256개)
        lut = np.array([self.id2rgb(i) for i in range(256)], dtype=np.uint8)
        rgb_mask = lut[objects]  # objects가 각 픽셀에 해당하는 색상 인덱스로 사용됨
        return rgb_mask
    
    def id2rgb(self, id, max_num_obj=256):
        if not 0 <= id <= max_num_obj:
            raise ValueError("ID should be in range(0, max_num_obj)")

        # Convert the ID into a hue value
        golden_ratio = 1.6180339887
        h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
        s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
        l = 0.5

        
        # Use colorsys to convert HSL to RGB
        rgb = np.zeros((3, ), dtype=np.uint8)
        if id==0:   #invalid region
            return rgb
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

        return rgb

    def run(self):
        rate = rospy.Rate(30)
        self.gaussians.training_setup(self)
        self.gaussians.spatial_lr_scale = self.scene_extent
        self.gaussians.update_learning_rate(1)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree

        # Define a timeout threshold (in seconds)
        TIMEOUT_THRESHOLD = 10.0

        while True:
            # Only check for timeout if the first topic has been received.
            if self.first_topic_received and (time.time() - self.last_topic_time > TIMEOUT_THRESHOLD):
                rospy.loginfo("No new topics received for %.2f seconds. Shutting down.", TIMEOUT_THRESHOLD)
                # rospy.signal_shutdown("No topics received for a while")
                break
            
            if not self.synced_data:
                # print("No synced data yet.")
                continue

            image_np = self.synced_data["image"]
            pts = self.synced_data["points"]
            cols = self.synced_data["colors"]
            pose = self.synced_data["pose"]

            # Add new points.
            num_points = pts.shape[0]
            _points = torch.tensor(pts, dtype=torch.float32).cuda()
            _colors = torch.tensor(cols, dtype=torch.float32).cuda()
            _identity_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float32).cuda()
            _rots = _identity_quat.unsqueeze(0).repeat(num_points, 1)
            _initial_scale = 0.01
            _scales = torch.full((num_points, 3), _initial_scale, dtype=torch.float32).cuda()
            _z_values = torch.tensor(np.linalg.norm(pts, axis=1) / 5000., dtype=torch.float32).cuda()

            self.gaussians.add_from_pcd2_tensor(
                _points, _colors, _rots, _scales, _z_values, []
            )

            # Process new scenes with visualization.
            if self.mapping_new_cams:
                viewpoint_cam = self.mapping_new_cams.pop(0)
                print("New scene processing...")

                gt_image = viewpoint_cam.original_image.cuda()

                self.training = True
                render_pkg = render_2(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                image_rendered = render_pkg["render"]

                Ll1_map, Ll1 = l1_loss(image_rendered, gt_image)
                L_ssim_map, L_ssim = ssim(image_rendered, gt_image)
                loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - L_ssim)
                loss = loss_rgb
                loss.backward()

                with torch.no_grad():
                    if self.train_iter % 200 == 0:
                        self.gaussians.prune_large_and_transparent(0.005, self.prune_th)
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.training = False
                self.train_iter += 1
                self.mapping_cams.append(viewpoint_cam)

                # Log visualization outputs.
                rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                rr.log("cam/current", rr.Image(image_np))
                rendered_np = image_rendered.detach().cpu().numpy().transpose(1, 2, 0)
                rendered_np = np.clip(rendered_np, 0, 1) * 255
                rendered_np = rendered_np.astype(np.uint8)
                rr.log("rendered_image", rr.Image(rendered_np))
                new_rotation = viewpoint_cam.R.cpu().numpy()
                new_translation = viewpoint_cam.t.cpu().numpy()
                new_rot_obj = Rotation.from_matrix(new_rotation)
                rr.log("cam/current", rr.Transform3D(
                    translation=new_translation.tolist(),
                    rotation=rr.Quaternion(xyzw=new_rot_obj.as_quat().tolist())
                ))
                rr.log("cam/current", rr.Pinhole(
                    resolution=[self.W, self.H],
                    image_from_camera=self.cam_intrinsic,
                    camera_xyz=rr.ViewCoordinates.RDF,
                ))
                rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(pts, colors=cols, radii=0.1))
                self.iteration_images += 1
                self.translation_path.append(new_translation.tolist())
                rr.log("path/translation", rr.LineStrips3D(
                    [self.translation_path],
                    radii=0.05,
                    colors=[0, 255, 0]
                ))
                rospy.loginfo("New scene trained & visualized, iter: %d, Loss: %.4f", self.train_iter, loss.item())

            
            elif self.mapping_cams:
                scene = random.choice(self.mapping_cams)
                # print("Random scene processing...")
                gt_image = scene.original_image.cuda()
                self.training = True
                render_pkg = render_2(scene, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                image_rendered = render_pkg["render"]

                Ll1_map, Ll1 = l1_loss(image_rendered, gt_image)
                L_ssim_map, L_ssim = ssim(image_rendered, gt_image)
                loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - L_ssim)
                loss = loss_rgb
                loss.backward()

                with torch.no_grad():
                    if self.train_iter % 200 == 0:
                        self.gaussians.prune_large_and_transparent(0.005, self.prune_th)
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.training = False
                self.train_iter += 1
                rospy.loginfo("Random scene trained, iter: %d, Loss: %.4f", self.train_iter, loss.item())

            rate.sleep()

        rospy.loginfo("Saving scene images...")
        for i in range(self.post_train_iter):
            # Once out of the loop, save all scene images.
            # Process a random previously trained scene (without visualization).
            trained_scenes = [s for s in self.saved_scenes if s.get("trained", False)]
            if trained_scenes:
                scene = random.choice(trained_scenes)
                if all(key in scene for key in ("image", "points", "colors", "pose")):
                    image_np = scene["image"]
                    pts = scene["points"]
                    cols = scene["colors"]
                    pose = scene["pose"]

                    translation = np.array([pose.position.x,
                                            pose.position.y,
                                            pose.position.z])
                    rotation = Rotation.from_quat([pose.orientation.x,
                                                    pose.orientation.y,
                                                    pose.orientation.z,
                                                    pose.orientation.w])
                    R_cam = rotation.as_matrix()
                    t_cam = translation
                    Rz_90 = np.array([[0,  1, 0],
                                        [-1, 0, 0],
                                        [0,  0, 1]], dtype=float)
                    Rx_minus_90 = np.array([[1, 0, 0],
                                            [0, 0, 1],
                                            [0, -1, 0]], dtype=float)
                    new_rotation = R_cam @ (Rz_90 @ Rx_minus_90)
                    new_translation = t_cam

                    viewpoint_cam = SharedCam(
                        FoVx=focal2fov(self.fx, self.W),
                        FoVy=focal2fov(self.fy, self.H),
                        image=image_np,
                        cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy
                    )
                    viewpoint_cam.on_cuda()
                    viewpoint_cam.setup_cam(new_rotation, new_translation, image_np)

                    gt_image = viewpoint_cam.original_image.cuda()
                    self.training = True
                    render_pkg = render_2(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                    image_rendered = render_pkg["render"]

                    # gt_obj = gt_objects
                    # logits = self.classifier(objects)
                    # pred_obj = torch.argmax(logits,dim=0)
                    # pred_obj_mask = self.visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
                    # rgb_mask = self.feature_to_rgb(objects)
                    # loss_obj = self.cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
                    # loss_obj = loss_obj / torch.log(torch.tensor(self.num_classes))

                    Ll1_map, Ll1 = l1_loss(image_rendered, gt_image)
                    L_ssim_map, L_ssim = ssim(image_rendered, gt_image)
                    loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - L_ssim)

                    # loss_obj_3d = None
                    # if self.train_iter % 5 == 0:
                    #     # regularize at certain intervals
                    #     logits3d = self.classifier(self.gaussians._objects_dc.permute(2,0,1))
                    #     prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
                    #     loss_obj_3d = loss_cls_3d(self.gaussians._xyz.squeeze().detach(), prob_obj3d, 5, 2, 300000, 1000)
                    #     loss = (self.loss_rgb_weight * loss_rgb
                    #             + self.loss_obj_weight * loss_obj
                    #             + self.loss_obj_3d_weight * loss_obj_3d)
                    #     # print(self.loss_rgb_weight * loss_rgb,
                    #     #         self.loss_obj_weight * loss_obj,
                    #     #         self.loss_obj_3d_weight * loss_obj_3d)
                    # else:
                    #     loss = (self.loss_rgb_weight * loss_rgb
                    #             + self.loss_obj_weight * loss_obj)

                    loss = loss_rgb
                    
                    loss.backward()

                    with torch.no_grad():
                        if self.train_iter % 200 == 0:
                            self.gaussians.prune_large_and_transparent(0.005, self.prune_th)
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.training = False
                    self.train_iter += 1
                    rospy.loginfo("Random scene post trained, iter: %d, Loss: %.4f", i, loss.item())

                    rr.log("cam/current", rr.Image(image_np))
                    rendered_np = image_rendered.detach().cpu().numpy().transpose(1, 2, 0)
                    rendered_np = np.clip(rendered_np, 0, 1) * 255
                    rendered_np = rendered_np.astype(np.uint8)
                    rr.log("rendered_image", rr.Image(rendered_np))

        # # """Sequentially display and save the combined ground truth and rendered images from saved scenes."""
        # if not os.path.exists(self.output_dir):
        #     os.makedirs(self.output_dir)

        # # Prepare common mask for rendering.
        # object_mask = np.ones((self.H, self.W), dtype=np.float32)

        # # Loop through each saved scene sequentially.
        # for idx, scene in enumerate(self.saved_scenes):
        #     # Check for the ground truth image.
        #     if "image" not in scene:
        #         rospy.logwarn("No ground truth image in scene %d", idx)
        #         continue

        #     # Convert ground truth image from RGB to BGR.
        #     gt_bgr = cv2.cvtColor(scene["image"], cv2.COLOR_RGB2BGR)

        #     # Re-render the scene using the stored scene parameters.
        #     rendered_np, rendered = self.render_saved_scene(scene, object_mask)
        #     # Convert rendered image from RGB to BGR for OpenCV.
        #     rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)

        #     # Concatenate the ground truth and rendered images horizontally.
        #     combined = np.hstack([gt_bgr, rendered_bgr])
            
        #     # Save the combined image.
        #     combined_path = os.path.join(self.output_dir, f"scene_{idx:03d}_combined.png")
        #     cv2.imwrite(combined_path, combined)
        #     rospy.loginfo("Saved combined image for scene %d at %s", idx, combined_path)
            
        #     # Display the combined image in an OpenCV window.
        #     window_name = "Scene - GT (left) | Rendered (right)"
        #     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        #     cv2.imshow(window_name, combined)
        #     rospy.loginfo("Displaying scene %d", idx)
        #     # Wait indefinitely for a key press before moving on.
        #     cv2.waitKey(1)

        self.calc_2d_metric()

def mse2psnr(x):
    return -10.*torch.log(x)/torch.log(torch.tensor(10.))

if __name__ == '__main__':
    gs = GaussianSplatting()
    # Run the training loop in a separate thread.
    training_thread = threading.Thread(target=gs.run)
    training_thread.daemon = True
    training_thread.start()
    # Use a multi-threaded spinner to allow callbacks to run concurrently.
    spinner = rospy.spin()
