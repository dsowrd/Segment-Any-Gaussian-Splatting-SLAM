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
import collections
import math

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import rerun as rr

sys.path.append('/home/hd/catkin_ws/src/gaussian_splatting/gaussian_splatting')
from arguments import SLAMParameters
from scene import GaussianModel
from scene.shared_objs import SharedCam
from utils.graphics_utils import focal2fov
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
from gaussian_renderer import render_4

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
    # Make a copy so the array is writable.
    return np.frombuffer(cloud_msg.data, dtype=new_dtype, count=num_points).copy()

def read_xyz_rgb_from_raw(cloud_msg, num_points=None):
    offsets = {}
    for field in cloud_msg.fields:
        offsets[field.name] = field.offset
    for key in ['x', 'y', 'z', 'rgb']:
        if key not in offsets:
            rospy.logerr(f"Error: Missing field {key}!")
            return [], []
    point_step = cloud_msg.point_step
    total_points = cloud_msg.width * cloud_msg.height
    if num_points is None:
        num_points = total_points
    else:
        num_points = min(num_points, total_points)
    points = []
    colors = []
    for i in range(num_points):
        off_x = i * point_step + offsets['x']
        off_y = i * point_step + offsets['y']
        off_z = i * point_step + offsets['z']
        off_rgb = i * point_step + offsets['rgb']
        x = struct.unpack('f', cloud_msg.data[off_x:off_x+4])[0]
        y = struct.unpack('f', cloud_msg.data[off_y:off_y+4])[0]
        z = struct.unpack('f', cloud_msg.data[off_z:off_z+4])[0]
        points.append([x, y, z])
        rgb_bytes = cloud_msg.data[off_rgb:off_rgb+4]
        int_rgb = struct.unpack('I', rgb_bytes)[0]
        r = (int_rgb >> 16) & 0xFF
        g = (int_rgb >> 8) & 0xFF
        b = int_rgb & 0xFF
        colors.append([r/255.0, g/255.0, b/255.0])
    return points, colors

def filter_new_points(new_points, new_colors, existing_points, distance_threshold):
    """
    Remove new points that are within distance_threshold of any point in existing_points.
    Uses a cKDTree for fast nearest neighbor search.
    """
    if existing_points.size == 0:
        return new_points, new_colors
    tree = cKDTree(existing_points)
    distances, _ = tree.query(new_points, k=1)
    mask = distances > distance_threshold
    return new_points[mask], new_colors[mask]

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
        self.img_queue = collections.deque()
        self.cloud_queue = collections.deque()
        self.odom_queue = collections.deque()
        self.synced_data = None
        self.saved_scenes = []
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

        # Camera intrinsics.
        self.scale = 0.5
        self.fx = 1293.56944 * self.scale
        self.fy = 1293.3155 * self.scale
        self.cx = 626.91359 * self.scale
        self.cy = 522.799224 * self.scale
        self.cam_intrinsic = np.array([[self.fx, 0, self.cx],
                                       [0, self.fy, self.cy],
                                       [0, 0, 1]])
        self.W = int(1280 * self.scale)
        self.H = int(1024 * self.scale)
        # Set voxel size for duplicate filtering.
        self.voxel_size = 0.1

        # Extrinsic calibration.
        extrinsic_T = np.array([0.04165, 0.02326, -0.0284])
        extrinsic_R = np.eye(3)
        Rcl = np.array([[ 0.00610193, -0.999863, -0.0154172],
                        [-0.00615449,  0.0153796, -0.999863],
                        [ 0.999962,    0.00619598, -0.0060598]])
        Pcl = np.array([0.0194384, 0.104689, -0.0251952])
        self.Pli = -extrinsic_R.T.dot(extrinsic_T)
        self.Rli = extrinsic_R.T
        self.Rcl = Rcl
        self.Pcl = Pcl
        self.Rci = self.Rcl.dot(self.Rli)
        self.Pci = self.Rcl.dot(self.Pli) + self.Pcl

        rospy.Subscriber("/rgb_img", Image, self.image_callback)
        rospy.Subscriber("/cloud_registered", PointCloud2, self.cloud_callback)
        rospy.Subscriber("/aft_mapped_to_init", Odometry, self.odom_callback)
        self.slop = 0.1

        rr.init("3dgsviewer")
        rr.spawn(connect=False)
        rr.connect()

    def image_callback(self, msg):
        ts = msg.header.stamp.to_sec()
        # print("Image callback header timestamp:", ts)
        self.img_queue.append((ts, msg))

    def cloud_callback(self, msg):
        ts = msg.header.stamp.to_sec()
        # print("Cloud callback header timestamp:", ts)
        self.cloud_queue.append((ts, msg))

    def odom_callback(self, msg):
        ts = msg.header.stamp.to_sec()
        # print("Odometry callback header timestamp:", ts)
        self.odom_queue.append((ts, msg))

    # def image_callback(self, msg):
    #     self.img_queue.append((rospy.get_time(), msg))

    # def cloud_callback(self, msg):
    #     self.cloud_queue.append((rospy.get_time(), msg))

    # def odom_callback(self, msg):
    #     self.odom_queue.append((rospy.get_time(), msg))

    def try_match(self):
        # Continue as long as we have at least one image message.
        while self.img_queue:
            # Use the timestamp from the earliest image as the reference.
            t_img, img_msg = self.img_queue[0]
            
            # Try to find a cloud message within self.slop of the image timestamp.
            matched_cloud = None
            cloud_index = None
            for idx, (t_cloud, cloud_msg) in enumerate(self.cloud_queue):
                if abs(t_cloud - t_img) < self.slop:
                    matched_cloud = cloud_msg
                    cloud_index = idx
                    break
            
            # Try to find an odometry message within self.slop of the image timestamp.
            matched_odom = None
            odom_index = None
            for idx, (t_odom, odom_msg) in enumerate(self.odom_queue):
                if abs(t_odom - t_img) < self.slop:
                    matched_odom = odom_msg
                    odom_index = idx
                    break
            
            # If both matching messages are found, process them.
            if matched_cloud is not None and matched_odom is not None:
                # Remove the matched image message.
                self.img_queue.popleft()
                
                # Remove the matched cloud message from its queue.
                # We remove messages up to and including the matched one.
                for _ in range(cloud_index + 1):
                    self.cloud_queue.popleft()
                
                # Remove the matched odom message.
                for _ in range(odom_index + 1):
                    self.odom_queue.popleft()
                
                self.process_synced(img_msg, matched_cloud, matched_odom)
            else:
                # If no match is found for the reference image,
                # you can either break out to wait for more messages
                # or discard the image if it's too old.
                # For example, if the image timestamp is significantly behind the current time:
                current_time = rospy.get_time()
                if current_time - t_img > self.slop:
                    # Discard the image message if too old.
                    self.img_queue.popleft()
                else:
                    # Otherwise, break and wait for new messages.
                    break

    def process_synced(self, image_msg, cloud_msg, odom_msg):
        synced = {}
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            synced["image"] = cv2.resize(cv_image_rgb, (self.W, self.H))
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        pts, cols = read_xyz_rgb_from_raw(cloud_msg)
        all_points = np.array(pts, dtype=np.float32)
        all_colors = np.array(cols, dtype=np.float32)

        # Filter new points using a KD-tree so that we only add new points (voxels not present in the current model).
        existing_xyz_np = (self.gaussians.get_xyz.cpu().detach().numpy()
                           if self.gaussians.get_xyz.numel() > 0 else np.empty((0, 3)))
        # filtered_points, filtered_colors = filter_new_points(all_points, all_colors, existing_xyz_np, distance_threshold=0.05)
        filtered_points, filtered_colors = all_points, all_colors
        synced["points"] = filtered_points
        synced["colors"] = filtered_colors

        synced["pose"] = odom_msg.pose.pose
        self.synced_data = synced

        # Save scene if movement exceeds thresholds.
        current_pose = odom_msg.pose.pose
        save_scene = False
        diff_translation = 0.0
        diff_angle = 0.0
        if not self.saved_scenes:
            save_scene = True
        else:
            last_pose = self.saved_scenes[-1]["pose"]
            current_translation = np.array([current_pose.position.x,
                                            current_pose.position.y,
                                            current_pose.position.z])
            last_translation = np.array([last_pose.position.x,
                                         last_pose.position.y,
                                         last_pose.position.z])
            diff_translation = np.linalg.norm(current_translation - last_translation)
            current_quat = np.array([current_pose.orientation.x,
                                     current_pose.orientation.y,
                                     current_pose.orientation.z,
                                     current_pose.orientation.w])
            last_quat = np.array([last_pose.orientation.x,
                                  last_pose.orientation.y,
                                  last_pose.orientation.z,
                                  last_pose.orientation.w])
            rel_rot = Rotation.from_quat(last_quat).inv() * Rotation.from_quat(current_quat)
            diff_angle = np.linalg.norm(rel_rot.as_rotvec())
            if diff_translation > 0.3 or diff_angle > 0.3:
                save_scene = True

        if save_scene:
            self.saved_scenes.append(synced)

    def run(self):
        rate = rospy.Rate(30)
        self.gaussians.training_setup(self)
        self.gaussians.spatial_lr_scale = self.scene_extent
        self.gaussians.update_learning_rate(1)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree

        depth_image = np.ones((self.H, self.W), dtype=np.float32)
        object_mask = np.ones((self.H, self.W), dtype=np.float32)

        last_scene_count = 0

        while not rospy.is_shutdown():
            self.try_match()

            if self.saved_scenes:
                scene = self.saved_scenes[-1]
                if all(key in scene for key in ("image", "points", "colors", "pose")):
                    image_np = scene["image"]
                    pts = scene["points"]
                    cols = scene["colors"]
                    pose = scene["pose"]

                    # If a new scene has been saved, add new points to the Gaussian model.
                    new_scene_generated = (len(self.saved_scenes) > last_scene_count)
                    if new_scene_generated:
                        last_scene_count = len(self.saved_scenes)
                        num_points = pts.shape[0]
                        _points = torch.tensor(pts, dtype=torch.float32).cuda()
                        _colors = torch.tensor(cols, dtype=torch.float32).cuda()
                        # Initialize rotations with the identity quaternion.
                        _identity_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float32).cuda()
                        _rots = _identity_quat.unsqueeze(0).repeat(num_points, 1)
                        # Initialize scales to a constant value.
                        _initial_scale = 0.01
                        _scales = torch.full((num_points, 3), _initial_scale, dtype=torch.float32).cuda()
                        _z_values = torch.tensor(np.linalg.norm(pts, axis=1) / 5000., dtype=torch.float32).cuda()

                        self.gaussians.add_from_pcd2_tensor(
                            _points, _colors, _rots, _scales, _z_values, []
                        )

                    # Compute camera pose from the scene state.
                    translation = np.array([pose.position.x,
                                            pose.position.y,
                                            pose.position.z])
                    rotation = Rotation.from_quat([pose.orientation.x,
                                                   pose.orientation.y,
                                                   pose.orientation.z,
                                                   pose.orientation.w])
                    R_cam = rotation.as_matrix()
                    t_cam = translation
                    # Apply fixed rotation correction: +90° about Z and -90° about X.
                    Rz_90 = np.array([[0,  1, 0],
                                      [-1, 0, 0],
                                      [0,  0, 1]], dtype=float)
                    Rx_minus_90 = np.array([[1, 0, 0],
                                            [0, 0, 1],
                                            [0, -1, 0]], dtype=float)
                    new_rotation = R_cam @ (Rz_90 @ Rx_minus_90)
                    new_translation = t_cam

                    # Set up the viewpoint camera.
                    viewpoint_cam = SharedCam(
                        FoVx=focal2fov(self.fx, self.W),
                        FoVy=focal2fov(self.fy, self.H),
                        image=image_np,
                        depth_image=depth_image,
                        object_mask=object_mask,
                        cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy
                    )
                    viewpoint_cam.on_cuda()
                    viewpoint_cam.setup_cam(new_rotation, new_translation, image_np, depth_image, object_mask)

                    gt_image = viewpoint_cam.original_image.cuda()

                    self.training = True
                    render_pkg = render_4(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
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

                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log("cam/current", rr.Image(image_np))
                    rendered_np = image_rendered.detach().cpu().numpy().transpose(1, 2, 0)
                    rendered_np = np.clip(rendered_np, 0, 1) * 255
                    rendered_np = rendered_np.astype(np.uint8)
                    rr.log("rendered_image", rr.Image(rendered_np))
                    rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(pts, colors=cols, radii=0.1))
                    self.iteration_images += 1

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
                    self.translation_path.append(new_translation.tolist())
                    rr.log("path/translation", rr.LineStrips3D(
                        [self.translation_path],
                        radii=0.05,
                        colors=[0, 255, 0]
                    ))
            rate.sleep()

if __name__ == '__main__':
    gs = GaussianSplatting()
    gs.run()
