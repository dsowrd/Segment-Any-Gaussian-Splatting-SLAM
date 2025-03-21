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

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
from nav_msgs.msg import Path, Odometry
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation
import rerun as rr
import math

# Pull in your local modules
sys.path.append('/home/hd/catkin_ws/src/gaussian_splatting/gaussian_splatting')
from arguments import SLAMParameters
from scene import GaussianModel
from scene.shared_objs import SharedCam
from utils.graphics_utils import focal2fov
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
from gaussian_renderer import render_2, render_4

def read_points_direct(cloud_msg):
    """
    Directly convert the raw data in a sensor_msgs/PointCloud2 message
    to a structured numpy array. Assumes fields "x", "y", "z", and "rgb"
    with offsets 0, 4, 8, 16 respectively.
    """
    # Define a dtype for the expected fields.
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

    points_array = np.frombuffer(cloud_msg.data, dtype=new_dtype, count=num_points)
    return points_array

def correct_cloud_msg(cloud_msg, default_color_rgb=0):
    """
    Check the "rgb" field in the cloud_msg and replace any NaN values
    with a default packed rgb value (default is 0, which corresponds to black).
    
    The rgb field is stored as a float; to pack a color (RGB8) into a float:
      - Create a 32-bit integer with 8 bits per channel (e.g. 0x00000000 for black)
      - Reinterpret that integer as a float.
    
    Parameters:
      cloud_msg: sensor_msgs/PointCloud2 message.
      default_color_rgb: a 32-bit integer representing the default color.
                         For black, use 0; for white, use 0xFFFFFFFF.
    
    Returns:
      The cloud_msg with its data modified such that any NaN in the rgb field is replaced.
    """
    # Convert raw data to a structured NumPy array.
    points_array = read_points_direct(cloud_msg)
    # Find indices where the 'rgb' field is NaN.
    nan_indices = np.isnan(points_array['rgb'])
    if np.any(nan_indices):
        print(f"Found {np.sum(nan_indices)} NaN values in the rgb field. Replacing with default.")
        # Pack the default color (for black: 0; for white: 0xFFFFFFFF)
        default_rgbf = struct.unpack('f', struct.pack('I', default_color_rgb))[0]
        points_array['rgb'][nan_indices] = default_rgbf
        # Update the cloud_msg.data with the modified array.
        cloud_msg.data = points_array.tobytes()
    else:
        print("No NaN values in the rgb field to correct.")
    return cloud_msg

def print_rgb_raw(cloud_msg, num_points=5):
    # Find the offset for the "rgb" field from the message fields.
    rgb_offset = None
    for field in cloud_msg.fields:
        if field.name == "rgb":
            rgb_offset = field.offset
            break
    if rgb_offset is None:
        print("No 'rgb' field found in the message!")
        return

    # Get point_step (number of bytes per point)
    point_step = cloud_msg.point_step
    total_points = cloud_msg.width * cloud_msg.height

    print(f"Printing raw 'rgb' data for the first {num_points} points (total points: {total_points}):")
    for i in range(min(num_points, total_points)):
        # Compute the offset in the raw data for this point's rgb field.
        offset = i * point_step + rgb_offset
        # Extract 4 bytes (since rgb is a float, 4 bytes)
        rgb_bytes = cloud_msg.data[offset:offset+4]
        print(f"Point {i}: rgb raw bytes: {rgb_bytes.hex()}")

def print_rgb_only(msg, num_points=5):
    # Find the offset for the "rgb" field
    rgb_offset = None
    for field in msg.fields:
        if field.name == "rgb":
            rgb_offset = field.offset
            break
    if rgb_offset is None:
        print("No 'rgb' field found in the message!")
        return

    point_step = msg.point_step
    total_points = msg.width * msg.height

    print(f"Printing raw 'rgb' data for the first {num_points} points (total points: {total_points}):")
    for i in range(min(num_points, total_points)):
        # Calculate the offset in the raw data for this point's rgb field.
        offset = i * point_step + rgb_offset
        # Extract 4 bytes (since rgb is stored as a float, 4 bytes)
        rgb_bytes = msg.data[offset:offset+4]
        # Unpack these 4 bytes into an unsigned 32-bit integer.
        int_rgb = struct.unpack('I', rgb_bytes)[0]
        # Extract each color channel.
        r = (int_rgb >> 16) & 0xFF
        g = (int_rgb >> 8) & 0xFF
        b = int_rgb & 0xFF
        print(f"Point {i}: Raw rgb bytes: {rgb_bytes.hex()}  -> int: {int_rgb}, channels: R={r}, G={g}, B={b}")

def print_xyz_rgb(msg, num_points=5):
    # Find offsets for x, y, z, and rgb from the message fields.
    x_offset = y_offset = z_offset = rgb_offset = None
    for field in msg.fields:
        if field.name == "x":
            x_offset = field.offset
        elif field.name == "y":
            y_offset = field.offset
        elif field.name == "z":
            z_offset = field.offset
        elif field.name == "rgb":
            rgb_offset = field.offset
    if x_offset is None or y_offset is None or z_offset is None or rgb_offset is None:
        print("Missing one or more required fields (x, y, z, rgb)!")
        return

    point_step = msg.point_step
    total_points = msg.width * msg.height

    print(f"Printing raw 'xyz' and 'rgb' data for the first {num_points} points (total points: {total_points}):")
    for i in range(min(num_points, total_points)):
        # Compute byte offsets for x, y, z, and rgb for point i.
        offset_x = i * point_step + x_offset
        offset_y = i * point_step + y_offset
        offset_z = i * point_step + z_offset
        offset_rgb = i * point_step + rgb_offset

        # Extract 4 bytes for x, y, z and unpack them as float.
        x_bytes = msg.data[offset_x:offset_x+4]
        y_bytes = msg.data[offset_y:offset_y+4]
        z_bytes = msg.data[offset_z:offset_z+4]
        x_val = struct.unpack('f', x_bytes)[0]
        y_val = struct.unpack('f', y_bytes)[0]
        z_val = struct.unpack('f', z_bytes)[0]

        # Extract 4 bytes for rgb and convert to an unsigned integer.
        rgb_bytes = msg.data[offset_rgb:offset_rgb+4]
        int_rgb = struct.unpack('I', rgb_bytes)[0]
        # Extract individual channels.
        r = (int_rgb >> 16) & 0xFF
        g = (int_rgb >> 8) & 0xFF
        b = int_rgb & 0xFF

        print(f"Point {i}: XYZ = ({x_val}, {y_val}, {z_val}), Raw rgb bytes: {rgb_bytes.hex()} -> int: {int_rgb}, channels: R={r}, G={g}, B={b}")

def read_xyz_rgb_from_raw(cloud_msg, num_points=None):
    """
    Read the xyz and rgb values from a sensor_msgs/PointCloud2 message by directly
    accessing the raw binary data. Returns two lists:
      - points: a list of [x, y, z] coordinates (floats)
      - colors: a list of [r, g, b] values normalized to [0, 1]
      
    This function assumes that the fields are tightly packed with the following offsets:
      - x: offset 0, float32
      - y: offset 4, float32
      - z: offset 8, float32
      - rgb: offset 16, float32
      
    If num_points is not provided, all points in the message are processed.
    """
    # Retrieve offsets for the desired fields.
    x_offset = y_offset = z_offset = rgb_offset = None
    for field in cloud_msg.fields:
        if field.name == "x":
            x_offset = field.offset
        elif field.name == "y":
            y_offset = field.offset
        elif field.name == "z":
            z_offset = field.offset
        elif field.name == "rgb":
            rgb_offset = field.offset
    if x_offset is None or y_offset is None or z_offset is None or rgb_offset is None:
        print("Error: Missing one or more required fields (x, y, z, rgb)!")
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
        offset_x = i * point_step + x_offset
        offset_y = i * point_step + y_offset
        offset_z = i * point_step + z_offset
        offset_rgb = i * point_step + rgb_offset

        # Unpack x, y, and z (each 4 bytes, float)
        x = struct.unpack('f', cloud_msg.data[offset_x:offset_x+4])[0]
        y = struct.unpack('f', cloud_msg.data[offset_y:offset_y+4])[0]
        z = struct.unpack('f', cloud_msg.data[offset_z:offset_z+4])[0]
        points.append([x, y, z])

        # Unpack rgb field (4 bytes stored as a float but containing packed RGB8 data)
        rgb_bytes = cloud_msg.data[offset_rgb:offset_rgb+4]
        int_rgb = struct.unpack('I', rgb_bytes)[0]
        r = (int_rgb >> 16) & 0xFF
        g = (int_rgb >> 8) & 0xFF
        b = int_rgb & 0xFF
        colors.append([r / 255.0, g / 255.0, b / 255.0])
    
    return points, colors


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
        # Queues for incoming messages
        self.img_queue = collections.deque()
        self.cloud_queue = collections.deque()
        self.path_queue = collections.deque()
        self.odom_queue = collections.deque()
        # Group synchronized messages in one dict
        self.synced_data = None
        # Buffer to save scenes when the camera is moving.
        self.saved_scenes = []

        # Visualization counters
        self.iteration_images = 0
        self.translation_path = []
        self.total_start_time = time.time()

        # Gaussian model parameters
        self.scene_extent = 2.5
        self.prune_th = 2.5
        self.gaussians = GaussianModel(self.sh_degree)
        self.pipe = Pipe(self.convert_SHs_python, self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.train_iter = 0

        # Camera intrinsics
        self.scale = 0.5
        self.fx = 1293.56944 * self.scale
        self.fy = 1293.3155 * self.scale
        self.cx = 626.91359 * self.scale
        self.cy = 522.799224 * self.scale
        self.cam_intrinsic = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1]
        ])
        self.W = int(1280 * self.scale)
        self.H = int(1024 * self.scale)
        # Set grid downsampling scale (e.g., sample every 10 pixels)
        self.downsample_scale = 10

        # ---- Extrinsic Calibration Parameters ----
        # These are provided (typically via ROS parameters) as:
        # extrinsic_T: [0.04165, 0.02326, -0.0284]
        # extrinsic_R: identity (i.e. [1,0,0, 0,1,0, 0,0,1])
        # Rcl: [0.00610193, -0.999863, -0.0154172,
        #       -0.00615449, 0.0153796, -0.999863,
        #       0.999962, 0.00619598, -0.0060598]
        # Pcl: [0.0194384, 0.104689, -0.0251952]
        extrinsic_T = np.array([0.04165, 0.02326, -0.0284])
        extrinsic_R = np.eye(3)  # identity
        Rcl = np.array([[ 0.00610193, -0.999863, -0.0154172],
                        [-0.00615449,  0.0153796, -0.999863],
                        [ 0.999962,    0.00619598, -0.0060598]])
        Pcl = np.array([0.0194384, 0.104689, -0.0251952])

        # Set IMU-to-LiDAR extrinsics.
        # These functions invert the given transformation:
        #   Pli = -extrinsic_R^T * extrinsic_T,
        #   Rli = extrinsic_R^T.
        self.Pli = -extrinsic_R.T.dot(extrinsic_T)   # Pli becomes [-0.04165, -0.02326, 0.0284]
        self.Rli = extrinsic_R.T                      # Rli is identity.

        # Set LiDAR-to-Camera extrinsics.
        # Rcl and Pcl are given directly.
        self.Rcl = Rcl
        self.Pcl = Pcl

        # Now combine these to get camera-to-IMU extrinsics:
        # Rci = Rcl * Rli, and Pci = Rcl * Pli + Pcl.
        self.Rci = self.Rcl.dot(self.Rli)             # Since Rli is I, Rci = Rcl.
        self.Pci = self.Rcl.dot(self.Pli) + self.Pcl    # Compute this vector.

        # For example, compute Pci:
        # Rcl * Pli = Rcl * [-0.04165, -0.02326, 0.0284]
        # (This product gives a vector; then adding Pcl gives the final offset.)
        # In our previous calculation, this came out approximately to:
        #   Pci ≈ [0.04201, 0.07619, -0.06716].
        # These (Rci, Pci) now represent the extrinsic calibration from the camera to the IMU.

        # Subscribe to topics
        rospy.Subscriber("/rgb_img", Image, self.image_callback)
        rospy.Subscriber("/cloud_registered", PointCloud2, self.cloud_callback)
        rospy.Subscriber("/path", Path, self.path_callback)
        rospy.Subscriber("/aft_mapped_to_init", Odometry, self.odom_callback)

        # self.cloud_repub = rospy.Publisher("/cloud_registered_repub", PointCloud2, queue_size=1)


        # Allowable time difference between messages (seconds)
        self.slop = 0.1

        # Initialize rerun viewer
        rr.init("3dgsviewer")
        rr.spawn(connect=False)
        rr.connect()

    #------------------------------------------------------
    # Grid-based downsampling filter for an organized point cloud.
    # Returns a tuple of indices (as a numpy array) and precomputed
    # normalized x and y offsets.
    #------------------------------------------------------
    def set_downsample_filter(self, downsample_scale):
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0, int(self.H / sample_interval) + 1)
        h_val = h_val - 1
        h_val[0] = 0
        h_val = h_val * self.W
        a, b = torch.meshgrid(h_val, torch.arange(0, self.W, sample_interval), indexing='ij')
        pick_idxs = ((a + b).flatten(), )
        v, u = torch.meshgrid(torch.arange(0, self.H), torch.arange(0, self.W), indexing='ij')
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
        x_pre = (u - self.cx) / self.fx
        y_pre = (v - self.cy) / self.fy
        return pick_idxs[0].numpy(), x_pre.numpy(), y_pre.numpy()

    #------------------------------------------------------
    # Callback functions: store messages with arrival time.
    #------------------------------------------------------
    def image_callback(self, msg):
        self.img_queue.append((rospy.get_time(), msg))

    def cloud_callback(self, msg):
        # print("Cloud header:", msg.header)
        # print("Cloud height:", msg.height)
        # print("Cloud width:", msg.width)
        # for field in msg.fields:
        #     print(f"field", field)

        # Print the first 100 bytes of the raw data in hex.
        # (The raw data is a byte-string; printing in hex can help you inspect the bit patterns.)
        # raw_sample = msg.data[:100]
        # print("Raw data (first 100 bytes):", raw_sample.hex())

        # # Print a few sample points (e.g., first 5)
        # print("Sample points:")
        # count = 0
        # for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
        #     # print(f"Point {count}: {point}")
        #     print(f"rgb: {point[3]}")
        #     count += 1
        #     if count >= 5:
        #         break
        # print(msg)
        # print_rgb_raw(msg, num_points=5)
        # print("Cloud header:", msg.header)
        # print_xyz_rgb(msg, num_points=5)
        self.cloud_queue.append((rospy.get_time(), msg))
        # self.cloud_repub.publish(msg)

    def path_callback(self, msg):
        self.path_queue.append((rospy.get_time(), msg))

    def odom_callback(self, msg):
        self.odom_queue.append((rospy.get_time(), msg))

    #------------------------------------------------------
    # Attempt to match messages from all topics based on arrival time.
    #------------------------------------------------------
    def try_match(self):
        while (self.img_queue and self.cloud_queue and self.path_queue and self.odom_queue):
            t_img, img_msg = self.img_queue[0]
            t_cloud, cloud_msg = self.cloud_queue[0]
            t_path, path_msg = self.path_queue[0]
            t_odom, odom_msg = self.odom_queue[0]

            if max(t_img, t_cloud, t_path, t_odom) - min(t_img, t_cloud, t_path, t_odom) < self.slop:
                self.img_queue.popleft()
                self.cloud_queue.popleft()
                self.path_queue.popleft()
                self.odom_queue.popleft()
                self.process_synced(img_msg, cloud_msg, path_msg, odom_msg)
            else:
                if min(t_img, t_cloud, t_path, t_odom) == t_img:
                    self.img_queue.popleft()
                elif min(t_img, t_cloud, t_path, t_odom) == t_cloud:
                    self.cloud_queue.popleft()
                elif min(t_img, t_cloud, t_path, t_odom) == t_path:
                    self.path_queue.popleft()
                else:
                    self.odom_queue.popleft()

    #------------------------------------------------------
    # Process messages and store them as a grouped data structure.
    # Save the scene only if the camera is moving compared to the last saved scene.
    #------------------------------------------------------
    def process_synced(self, image_msg, cloud_msg, path_msg, odom_msg):
        synced = {}
        # Process Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            synced["image"] = cv2.resize(cv_image_rgb, (self.W, self.H))
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Process PointCloud2
        points_array = read_points_direct(cloud_msg)
        points_list = []
        colors_list = []
        
        # for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z", "rgb"), skip_nans=False):
        #     x, y, z, rgbf = point
        #     points_list.append([x, y, z])

        #     # if math.isnan(rgbf):
        #     #     print("Found NaN in rgb field for point:", point)
        #     #     # Print the first 100 bytes of the raw data as a hex string for inspection.
        #     #     print("Raw cloud_msg data (first 100 bytes):", cloud_msg.data[:100].hex())
        #     # print(f"XYZ: {x}, {y}, {z}, RGBF: {rgbf}")
        #     int_rgb = struct.unpack('I', struct.pack('f', rgbf))[0]
        #     # int_rgb = struct.unpack('I', rgbf[0])
        #     r = (int_rgb >> 16) & 0x0000ff
        #     g = (int_rgb >> 8) & 0x0000ff
        #     b = int_rgb & 0x0000ff
        #     # print(f"XYZ: {x}, {y}, {z}")
        #     # print(f"RGB: {r}, {g}, {b}")
        #     colors_list.append([r / 255.0, g / 255.0, b / 255.0])
        pts, cols = read_xyz_rgb_from_raw(cloud_msg)
        # all_points = np.array(points_list, dtype=np.float32)
        # all_colors = np.array(colors_list, dtype=np.float32)
        all_points = np.array(pts, dtype=np.float32)
        all_colors = np.array(cols, dtype=np.float32)

        # header = rospy.Header()
        # header.stamp = rospy.Time.now()
        # header.frame_id = "camera_init"  # Or the appropriate frame
        # fields = [
        #     PointField('x', 0, PointField.FLOAT32, 1),
        #     PointField('y', 4, PointField.FLOAT32, 1),
        #     PointField('z', 8, PointField.FLOAT32, 1),
        #     PointField('rgb', 16, PointField.FLOAT32, 1)
        # ]
        # data = []
        # for i in range(len(all_points)):
        #     x, y, z = all_points[i]
        #     # Pack the color back into a float.
        #     r, g, b = [int(c * 255) for c in all_colors[i]]
        #     rgb_int = (r << 16) | (g << 8) | b
        #     rgb_float = struct.unpack('f', struct.pack('I', rgb_int))[0]
        #     data.append([x, y, z, rgb_float])
        # new_cloud = point_cloud2.create_cloud(header, fields, data)
        # self.cloud_repub.publish(new_cloud)

        if len(all_points) == self.H * self.W:
            organized_points = all_points.reshape(self.H, self.W, 3)
            organized_colors = all_colors.reshape(self.H, self.W, 3)
            pick_idxs, x_pre, y_pre = self.set_downsample_filter(self.downsample_scale)
            flat_points = organized_points.reshape(-1, 3)
            flat_colors = organized_colors.reshape(-1, 3)
            sampled_points = flat_points[pick_idxs]
            sampled_colors = flat_colors[pick_idxs]
            synced["points"] = sampled_points
            synced["colors"] = sampled_colors
        else:
            if len(all_points) > 0:
                # Use all points (or adjust the downsample factor as needed)
                downsample_factor = 1.0
                num_sample = int(len(all_points) * downsample_factor)
                num_sample = max(num_sample, 1)
                indices = np.random.choice(len(all_points), num_sample, replace=False)
                synced["points"] = all_points[indices]
                synced["colors"] = all_colors[indices]
            else:
                synced["points"] = None
                synced["colors"] = None

        synced["path"] = path_msg
        synced["pose"] = odom_msg.pose.pose
        self.synced_data = synced

        # --- Save the scene only if the camera is moving ---
        current_pose = odom_msg.pose.pose
        save_scene = False
        diff_translation = 0.0
        diff_angle = 0.0
        if not self.saved_scenes:
            save_scene = True
        else:
            last_scene = self.saved_scenes[-1]
            last_pose = last_scene["pose"]
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
            diff_angle = np.linalg.norm(rel_rot.as_rotvec())  # in radians
            threshold_translation = 0.3  # 30 cm
            threshold_angle = 0.3        # ~17.2 degrees (0.3 rad)
            if diff_translation > threshold_translation or diff_angle > threshold_angle:
                save_scene = True

        if save_scene:
            self.saved_scenes.append(synced)
            # rospy.loginfo("Scene saved: Translation change = %.3f m, Rotation change = %.3f rad",
            #               diff_translation, diff_angle)
        else:
            # rospy.loginfo("Scene NOT saved (camera static): Translation change = %.3f m, Rotation change = %.3f rad",
            #               diff_translation, diff_angle)
            pass

    #------------------------------------------------------
    # Main processing loop.
    # Instead of always using the most recent scene, randomly select a saved scene.
    #------------------------------------------------------
    def run(self):
        rate = rospy.Rate(30)  # 10 Hz
        t = torch.zeros((1, 1), dtype=torch.float32).cuda()
        self.gaussians.training_setup(self)
        self.gaussians.spatial_lr_scale = self.scene_extent
        self.gaussians.update_learning_rate(1)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree

        # Create dummy depth_image and object_mask
        depth_image = np.ones((self.H, self.W), dtype=np.float32)
        object_mask = np.ones((self.H, self.W), dtype=np.float32)

        while not rospy.is_shutdown():
            # First, accumulate new scenes.
            self.try_match()

            # If we have any saved scenes, randomly select one for training.
            if self.saved_scenes:
                # scene = random.choice(self.saved_scenes)
                scene = self.saved_scenes[-1]
                # self.saved_scenes = []
                if (scene.get("image") is not None and
                    scene.get("points") is not None and
                    scene.get("colors") is not None and
                    scene.get("pose") is not None):

                    image_np = scene["image"]
                    points = scene["points"]
                    colors = scene["colors"]
                    pose = scene["pose"]

                    image_tensor = (torch.tensor(image_np, dtype=torch.float32)
                                    .permute(2, 0, 1).unsqueeze(0).cuda())

                    num_points = points.shape[0]
                    _points = torch.tensor(points, dtype=torch.float32).cuda()
                    _colors = torch.tensor(colors, dtype=torch.float32).cuda()
                    _rots = torch.zeros((num_points, 4), dtype=torch.float32).cuda()
                    _scales = torch.zeros((num_points, 3), dtype=torch.float32).cuda()
                    _z_values = torch.tensor(np.linalg.norm(points, axis=1) / 5000.,
                                             dtype=torch.float32).cuda()

                    self.gaussians.add_from_pcd2_tensor(
                        _points, _colors, _rots, _scales, _z_values, []
                    )

                    # # Compute new camera pose from the saved scene's pose (state)
                    # # The state (from the topic) gives the IMU pose: Rwi and Pwi.
                    # # We now compute the camera pose (T_cw) as:
                    # #    Rcw = Rci * Rwiᵀ
                    # #    Pcw = -Rci * Rwiᵀ * Pwi + Pci
                    # # Where:
                    # #    Rci = Rcl * Rli   (and here Rli is identity and Pli = -extrinsic_T)
                    # #    Pci = Rcl * Pli + Pcl
                    # # In our initialization above, we have computed:
                    # #    self.Rci and self.Pci
                    # # Therefore:
                    # Rwi = Rotation.from_quat([pose.orientation.x,
                    #                           pose.orientation.y,
                    #                           pose.orientation.z,
                    #                           pose.orientation.w]).as_matrix()
                    # Pwi = np.array([pose.position.x,
                    #                 pose.position.y,
                    #                 pose.position.z])
                    # # Compute camera pose:
                    # R_cam_new = self.Rci.dot(Rwi.T)
                    # P_cam_new = -self.Rci.dot(Pwi) + self.Pci

                    # new_translation = P_cam_new  # This is the camera position in the world.
                    # # new_translation = Pwi
                    # new_rotation = R_cam_new     # This is the camera rotation in the world.

                    translation = np.array([
                        pose.position.x,
                        pose.position.y,
                        pose.position.z
                    ])
                    rotation = Rotation.from_quat([
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w
                    ])

                    R_cam = rotation.as_matrix()  # (3,3)
                    t_cam = translation           # (3,)

                    # 3) Define +90 deg about Z and -90 deg about X
                    Rz_90 = np.array([
                        [0,  1,  0],
                        [-1, 0,  0],
                        [0,  0,  1]
                    ], dtype=float)

                    Rx_minus_90 = np.array([
                        [1,  0,  0],
                        [0,  0,  1],
                        [0, -1,  0]
                    ], dtype=float)

                    # 4) Combine them, post-multiplying by Rx_90
                    Rz_90_then_Rx_minus_90 = Rz_90 @ Rx_minus_90
                    new_rotation = R_cam @ Rz_90_then_Rx_minus_90
                    # new_rotation = Rotation.from_matrix(R_cam_new)

                    # 5) Translation can remain the same if pivoting in place
                    # new_translation = np.array([t_cam[1], t_cam[2], -t_cam[0]])
                    new_translation = t_cam

                    # Setup the viewpoint camera with the computed transformation.
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
                    # print(f"Using transformation: R: {new_rotation}, t: {new_translation}")

                    gt_image = viewpoint_cam.original_image.cuda()

                    # Training and rendering.
                    self.training = True
                    render_pkg = render_4(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                    # render_pkg = render_2(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
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

                    # Visualization / Logging with rerun.
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log("cam/current", rr.Image(image_np))
                    rendered_np = image_rendered.detach().cpu().numpy().transpose(1, 2, 0)
                    rendered_np = np.clip(rendered_np, 0, 1) * 255
                    rendered_np = rendered_np.astype(np.uint8)
                    # rr.log("cam/current", rr.Image(rendered_np))
                    rr.log("rendered_image", rr.Image(rendered_np))
                    rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.1))
                    self.iteration_images += 1

                    new_rot_obj = Rotation.from_matrix(new_rotation)
                    rr.log("cam/current", rr.Transform3D(
                        translation=new_translation.tolist(),
                        rotation=rr.Quaternion(xyzw=new_rot_obj.as_quat().tolist())
                    ))
                    # print(f"translation: {new_translation}")
                    # print(f"rotation: {new_rot_obj.as_quat()}")
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
