#!/usr/bin/env python3
import sys
import os
import rospy
import time
import numpy as np
import struct
import cv2
import torch
import collections

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Path, Odometry
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation
import rerun as rr

# Pull in your local modules
sys.path.append('/home/hd/catkin_ws/src/gaussian_splatting/gaussian_splatting')
from arguments import SLAMParameters
from scene import GaussianModel
from scene.shared_objs import SharedCam
from utils.graphics_utils import focal2fov
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
from gaussian_renderer import render_4

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

        # Queues to store (arrival_time, message)
        self.img_queue = collections.deque()
        self.cloud_queue = collections.deque()
        self.path_queue = collections.deque()
        self.odom_queue = collections.deque()

        # Buffer for "current" data after matching
        self.current_image = np.zeros((512, 640, 3), dtype=np.uint8)
        self.points = None
        self.colors = None
        self.latest_pose = None

        # Visualization counters
        self.iteration_images = 0
        self.total_start_time = time.time()

        # Gaussian model
        self.scene_extent = 2.5
        self.prune_th = 2.5

        self.gaussians = GaussianModel(self.sh_degree)
        self.pipe = Pipe(self.convert_SHs_python, self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.train_iter = 0
        self.mapping_cams = []
        self.mapping_losses = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []
        self.mapping_cams = []

        # Intrinsics
        self.fx = 431.795259219
        self.fy = 431.550090267
        self.cx = 310.833037316
        self.cy = 266.985989326
        self.cam_intrinsic = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1]
        ])
        self.W = 640
        self.H = 512

        # Extrinsic parameters (not strictly used in this example)
        self.extrinsic_T = np.array([0.04165, 0.02326, -0.0284])
        self.extrinsic_R = np.eye(3)

        self.translation_path = []

        # Subscribe independently (no message_filters)
        rospy.Subscriber("/left_camera/image", Image, self.image_callback)
        rospy.Subscriber("/cloud_registered", PointCloud2, self.cloud_callback)
        rospy.Subscriber("/path", Path, self.path_callback)
        rospy.Subscriber("/aft_mapped_to_init", Odometry, self.odom_callback)

        # For local time-based matching
        self.slop = 0.1  # seconds of allowable difference in arrival times

        # Initialize rerun
        rospy.loginfo("Launching Gaussian Splatting process (local arrival-time sync)...")
        rr.init("3dgsviewer")
        rr.spawn(connect=False)
        rr.connect()

    #-------------------------------
    #   Callbacks
    #-------------------------------
    def image_callback(self, msg):
        """Stores (arrival_time, Image) in img_queue."""
        arrival_time = rospy.get_time()
        self.img_queue.append((arrival_time, msg))

    def cloud_callback(self, msg):
        """Stores (arrival_time, PointCloud2) in cloud_queue."""
        arrival_time = rospy.get_time()
        self.cloud_queue.append((arrival_time, msg))

    def path_callback(self, msg):
        """Stores (arrival_time, Path) in path_queue."""
        arrival_time = rospy.get_time()
        self.path_queue.append((arrival_time, msg))

    def odom_callback(self, msg):
        """Stores (arrival_time, Odometry) in odom_queue."""
        arrival_time = rospy.get_time()
        self.odom_queue.append((arrival_time, msg))

    #-------------------------------
    #  Matching based on local arrival time
    #-------------------------------
    def try_match(self):
        """
        Attempt to match the earliest messages from each of the four queues
        by comparing their arrival times. If they fall within `self.slop`,
        we process them together and remove them from the queues.
        Otherwise, discard whichever arrived first and try again.
        """
        while (self.img_queue and 
               self.cloud_queue and 
               self.path_queue and 
               self.odom_queue):
            t_img, img_msg = self.img_queue[0]
            t_cloud, cloud_msg = self.cloud_queue[0]
            t_path, path_msg = self.path_queue[0]
            t_odom, odom_msg = self.odom_queue[0]

            # Find min and max arrival time
            min_t = min(t_img, t_cloud, t_path, t_odom)
            max_t = max(t_img, t_cloud, t_path, t_odom)
            time_diff = max_t - min_t

            if time_diff < self.slop:
                # Good match: pop them all
                self.img_queue.popleft()
                self.cloud_queue.popleft()
                self.path_queue.popleft()
                self.odom_queue.popleft()

                # Now process them as if they're "synced"
                self.process_synced(img_msg, cloud_msg, path_msg, odom_msg)
            else:
                # Not matched, discard the earliest one
                # so it doesn't block matching with newer messages
                if min_t == t_img:
                    self.img_queue.popleft()
                elif min_t == t_cloud:
                    self.cloud_queue.popleft()
                elif min_t == t_path:
                    self.path_queue.popleft()
                else:  # min_t == t_odom
                    self.odom_queue.popleft()

    def process_synced(self, image_msg, cloud_msg, path_msg, odom_msg):
        """
        Equivalent to 'synced_callback' but triggered by local arrival-time matching.
        Parses the messages, stores them in self.current_image, self.points, etc.
        """
        # 1) Convert Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            cv_image_resized = cv2.resize(cv_image_rgb, (self.W, self.H))
            self.current_image = cv_image_resized
            # rospy.loginfo("Local-time sync: Received /left_camera/image")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # 2) Parse PointCloud2
        points_list = []
        colors_list = []
        for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z, rgbf = point
            points_list.append([x, y, z])
            # Extract RGB from float
            int_rgb = struct.unpack('I', struct.pack('f', rgbf))[0]
            r = (int_rgb >> 16) & 0x0000ff
            g = (int_rgb >> 8) & 0x0000ff
            b = (int_rgb) & 0x0000ff
            colors_list.append([r / 255.0, g / 255.0, b / 255.0])

        all_points = np.array(points_list, dtype=np.float32)
        all_colors = np.array(colors_list, dtype=np.float32)

        # Downsample
        downsample_factor = 1.0
        if len(all_points) > 0:
            indices = np.random.choice(len(all_points), 
                                       int(len(all_points)*downsample_factor), 
                                       replace=False)
            self.points = all_points[indices]
            self.colors = all_colors[indices]
        else:
            self.points = None
            self.colors = None

        # rospy.loginfo("Local-time sync: Received /cloud_registered")

        # 3) Path
        # if path_msg and len(path_msg.poses) > 0:
        #     rospy.loginfo("Local-time sync: Received /path with {} poses".format(len(path_msg.poses)))

        # 4) Odometry => store pose
        pose_stamped = odom_msg.pose
        self.latest_pose = pose_stamped.pose
        # rospy.loginfo("Local-time sync: Received /aft_mapped_to_init")

    #-------------------------------
    #  Main loop
    #-------------------------------
    def run(self):
        rate = rospy.Rate(10)  # 10 Hz

        t = torch.zeros((1,1)).float().cuda()

        self.gaussians.training_setup(self)
        self.gaussians.spatial_lr_scale = self.scene_extent
        self.gaussians.update_learning_rate(1)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree

        # Create depth_image and object_mask with the same resolution as self.current_image
        depth_image = np.ones((self.H, self.W), dtype=np.float32)
        object_mask = np.ones((self.H, self.W), dtype=np.float32)

        viewpoint_cam = SharedCam(FoVx=focal2fov(self.fx, self.W), FoVy=focal2fov(self.fy, self.H),
                        image=self.current_image, depth_image=depth_image, object_mask=object_mask,
                        cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy)

        while not rospy.is_shutdown():
            # 1) Try to match queued messages by arrival time
            self.try_match()

            # 2) If we have data, do computations and visualize
            if (self.current_image is not None and
                self.points is not None and
                self.colors is not None):

                if self.points.shape[0] != self.colors.shape[0]:
                    rospy.logerr("Points and colors do not have the same size. Skipping.")
                else:
                    # Convert to GPU tensors
                    num_points = self.points.shape[0]
                    _points = torch.tensor(self.points, dtype=torch.float32).cuda()
                    _colors = torch.tensor(self.colors, dtype=torch.float32).cuda()
                    _rots = torch.zeros((num_points, 4), dtype=torch.float32).cuda()
                    _scales = torch.zeros((num_points, 3), dtype=torch.float32).cuda()
                    _z_values = torch.tensor(
                        np.linalg.norm(self.points, axis=1), 
                        dtype=torch.float32
                    ).cuda()

                    # rospy.loginfo(
                    #     f"Points: {num_points}, "
                    #     f"_points shape: {_points.shape}, "
                    #     f"_colors shape: {_colors.shape}"
                    # )
                    self.gaussians.add_from_pcd2_tensor(
                        _points, _colors, _rots, _scales, _z_values, []
                    )
                    # rospy.loginfo(f"Added {num_points} points to Gaussian model")
                    # # Assuming get_xyz() returns a tensor of Gaussian positions
                    # total_gaussians = self.gaussians.get_xyz.shape[0]
                    # rospy.loginfo(f"Total Gaussians: {total_gaussians}")

                    # Extract translation and rotation from the latest pose
                    translation = np.array([
                        self.latest_pose.position.x,
                        self.latest_pose.position.y,
                        self.latest_pose.position.z
                    ])
                    rotation = Rotation.from_quat([
                        self.latest_pose.orientation.x,
                        self.latest_pose.orientation.y,
                        self.latest_pose.orientation.z,
                        self.latest_pose.orientation.w
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
                    R_cam_new = R_cam @ Rz_90_then_Rx_minus_90
                    new_rotation = Rotation.from_matrix(R_cam_new)

                    # 5) Translation can remain the same if pivoting in place
                    new_translation = np.array([t_cam[1], t_cam[2], -t_cam[0]])

                    # print(self.current_image.shape)
                    viewpoint_cam.on_cuda()
                    viewpoint_cam.setup_cam(new_rotation.as_matrix(), new_translation, self.current_image, depth_image, object_mask)

                    gt_image = viewpoint_cam.original_image.cuda()

                    self.training=True
                    render_pkg = render_4(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                    image = render_pkg["render"]

                    # Loss
                    Ll1_map, Ll1 = l1_loss(image, gt_image)
                    L_ssim_map, L_ssim = ssim(image, gt_image)
                    loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - L_ssim)
                    loss = loss_rgb
                    loss.backward()

                    with torch.no_grad():
                        if self.train_iter % 200 == 0:  # 200
                            self.gaussians.prune_large_and_transparent(0.005, self.prune_th)
                        
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none = True)
                        
                        rgb_np = image.detach().cpu().numpy().transpose(1,2,0)
                        rgb_np = np.clip(rgb_np, 0., 1.0) * 255
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                        rr.log("gt_image", rr.Image(gt_image.detach().cpu().numpy().transpose(1,2,0)))
                        rr.log("rendered_rgb", rr.Image(rgb_np))
                    
                    self.training = False
                    self.train_iter += 1

            # 3) Visualization / Logging with rerun
            rr.set_time_seconds("log_time", time.time() - self.total_start_time)

            # # Log image if we have one
            # if self.current_image is not None:
            #     rr.log("cam/current", rr.Image(self.current_image))

            # # Log point cloud if we have it
            # if self.points is not None and self.colors is not None:
            #     # scaled_points = self.points * scale_factor
            #     scaled_points = self.points
            #     rr.log(
            #         f"pt/trackable/{self.iteration_images}",
            #         rr.Points3D(scaled_points, colors=self.colors, radii=0.1)
            #     )
            #     self.iteration_images += 1

            # # Log camera extrinsics/pose
            # if self.latest_pose is not None:
            #     # 1) Get translation & rotation from the latest pose
            #     translation = np.array([
            #         self.latest_pose.position.x,
            #         self.latest_pose.position.y,
            #         self.latest_pose.position.z
            #     ])
            #     rotation = Rotation.from_quat([
            #         self.latest_pose.orientation.x,
            #         self.latest_pose.orientation.y,
            #         self.latest_pose.orientation.z,
            #         self.latest_pose.orientation.w
            #     ])

            #     # 2) Construct rotation & translation
            #     R_cam = rotation.as_matrix()  # (3,3)
            #     t_cam = translation           # (3,)

            #     # 3) Define +90 deg about Z and +90 deg about X
            #     Rz_90 = np.array([
            #         [0,  1,  0],
            #         [-1, 0,  0],
            #         [0,  0,  1]
            #     ], dtype=float)

            #     Rx_90 = np.array([
            #         [1,  0,  0],
            #         [0,  0,  1],
            #         [0, -1,  0]
            #     ], dtype=float)

            #     # 4) Combine them, post-multiplying by Rx_90
            #     Rz_90_then_Rx_90 = Rz_90 @ Rx_90
            #     R_cam_new = R_cam @ Rz_90_then_Rx_90
            #     new_rotation = Rotation.from_matrix(R_cam_new)

            #     # 5) Translation can remain the same if pivoting in place
            #     new_translation = t_cam
            #     self.translation_path.append(new_translation.tolist())

            #     # 6) Log your camera pose
            #     rr.log(
            #         "cam/current",
            #         rr.Transform3D(
            #             translation=new_translation.tolist(),
            #             rotation=rr.Quaternion(xyzw=new_rotation.as_quat().tolist())
            #         )
            #     )
            #     rr.log(
            #         "cam/current",
            #         rr.Pinhole(
            #             resolution=[self.W, self.H],
            #             image_from_camera=self.cam_intrinsic,
            #             camera_xyz=rr.ViewCoordinates.RDF,
            #         )
            #     )

            #     # 7) Log the translation path as a line strip
            #     rr.log(
            #         "path/translation",
            #         rr.LineStrips3D(
            #             [self.translation_path],
            #             radii=0.05,
            #             colors=[0, 255, 0]
            #         )
            #     )

            rate.sleep()


if __name__ == '__main__':
    gs = GaussianSplatting()
    gs.run()
