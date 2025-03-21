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
import argparse
from typing import Any, Dict, Generator,List

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
sys.path.append('/home/irl/Workspace_Hyundo/catkin_ws/src/gaussian_splatting/ByteTrack')
from arguments import SLAMParameters
from scene import GaussianModel
from scene.shared_objs import SharedCam
from utils.graphics_utils import focal2fov
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
from gaussian_renderer import render_2, render_4

from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor
from mobilesamv2.utils.transforms import ResizeLongestSide

# from yolox.data.data_augment import preproc
# from yolox.exp import get_exp
# from yolox.utils import fuse_model, get_model_info, postprocess
# from yolox.utils.visualize import plot_tracking
# from yolox.tracker.byte_tracker import BYTETracker
# from yolox.tracking_utils.timer import Timer

# import kornia
# import kornia.feature as KF
# from kornia.feature.loftr import LoFTR, default_cfg

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image

THRESHOLD_PERCENT = 0.5

def project_points(points, pose, viewpoint_cam):
    """
    주어진 pose와 내부 파라미터를 사용해 points (N,3)를 2D 이미지 좌표 (u,v)로 투영합니다.
    좌표 변환:
      - world-to-camera: p_cam = R_cw @ p_world + t_cw, (R_cw = R.T, t_cw = -R.T @ translation)
      - 새 좌표계 (x: 정면, y: 왼쪽, z: 위쪽) → conventional: X = -y, Y = -z, Z = x
    """
    # pose 정보 (numpy)
    translation = np.array([pose.position.x, pose.position.y, pose.position.z])
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_quat([pose.orientation.x,
                                    pose.orientation.y,
                                    pose.orientation.z,
                                    pose.orientation.w])
    R = rotation.as_matrix()  # numpy array (3,3)
    R_cw = R.T
    t_cw = -R_cw @ translation

    # points는 GPU 텐서; numpy로 변환
    points_np = points.cpu().numpy()  # (N,3)
    points_cam = (R_cw @ points_np.T).T + t_cw  # (N,3)

    # 새 좌표계에서 conventional 카메라 좌표계로 변환: X = -y, Y = -z, Z = x
    X = -points_cam[:, 1]
    Y = -points_cam[:, 2]
    Z = points_cam[:, 0] + 1e-6

    # 내부 파라미터는 torch 텐서일 경우, numpy로 변환
    fx = viewpoint_cam.fx.cpu().item() if hasattr(viewpoint_cam.fx, "cpu") else viewpoint_cam.fx
    fy = viewpoint_cam.fy.cpu().item() if hasattr(viewpoint_cam.fy, "cpu") else viewpoint_cam.fy
    cx = viewpoint_cam.cx.cpu().item() if hasattr(viewpoint_cam.cx, "cpu") else viewpoint_cam.cx
    cy = viewpoint_cam.cy.cpu().item() if hasattr(viewpoint_cam.cy, "cpu") else viewpoint_cam.cy

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    image_width = int(viewpoint_cam.image_width.cpu().item() if hasattr(viewpoint_cam.image_width, "cpu") else viewpoint_cam.image_width)
    image_height = int(viewpoint_cam.image_height.cpu().item() if hasattr(viewpoint_cam.image_height, "cpu") else viewpoint_cam.image_height)

    u_int = np.clip(np.round(u), 0, image_width - 1).astype(np.int32)
    v_int = np.clip(np.round(v), 0, image_height - 1).astype(np.int32)
    return u_int, v_int


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

def create_model():
    Prompt_guided_path='/home/irl/Workspace_Hyundo/catkin_ws/src/gaussian_splatting/MobileSAM/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt'
    obj_model_path='/home/irl/Workspace_Hyundo/catkin_ws/src/gaussian_splatting/MobileSAM/MobileSAMv2/weight/ObjectAwareModel.pt'
    ObjAwareModel = ObjectAwareModel(obj_model_path)
    PromptGuidedDecoder=sam_model_registry['PromptGuidedDecoder'](Prompt_guided_path)
    mobilesamv2 = sam_model_registry['vit_h']()
    mobilesamv2.prompt_encoder=PromptGuidedDecoder['PromtEncoder']
    mobilesamv2.mask_decoder=PromptGuidedDecoder['MaskDecoder']
    return mobilesamv2,ObjAwareModel

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument(
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result", action="store_true", help="whether to save the result"
    )
    parser.add_argument(
        "-f", "--exp_file",
        default="/home/irl/Workspace_Hyundo/catkin_ws/src/gaussian_splatting/ByteTrack/exps/example/mot/yolox_x_ablation.py",
        help="experiment description file",
    )
    parser.add_argument(
        "-c", "--ckpt",
        default="/home/irl/Workspace_Hyundo/catkin_ws/src/gaussian_splatting/ByteTrack/weights/yolox.pth",
        help="model checkpoint file"
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--device", default="cuda:0", help="device to run model")
    parser.add_argument("--conf", default=None, type=float, help="confidence threshold")
    parser.add_argument("--nms", default=None, type=float, help="nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frames per second")
    parser.add_argument("--track_thresh", default=0.3, type=float, help="tracking threshold")
    parser.add_argument("--track_buffer", default=60, type=int, help="frames for keeping track")
    parser.add_argument("--match_thresh", default=0.8, type=float, help="matching threshold for tracking")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="aspect ratio filtering threshold"
    )
    parser.add_argument("--min_box_area", type=float, default=10, help="minimum box area")
    
    return parser

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

        self.encoder_path={'efficientvit_l2':'/home/irl/Workspace_Hyundo/catkin_ws/src/gaussian_splatting/MobileSAM/MobileSAMv2/weight/l2.pt',
                'tiny_vit':'/home/irl/Workspace_Hyundo/catkin_ws/src/gaussian_splatting/MobileSAM/MobileSAMv2/weight/mobile_sam.pt',
                'sam_vit_h':'/home/irl/Workspace_Hyundo/catkin_ws/src/gaussian_splatting/MobileSAM/MobileSAMv2/weight/sam_vit_h.pt',}
        self.mobilesamv2, self.ObjAwareModel = create_model()
        self.image_encoder = sam_model_registry['efficientvit_l2'](self.encoder_path['efficientvit_l2'])
        self.mobilesamv2.image_encoder = self.image_encoder
        self.mobilesamv2.to(device='cuda')
        self.mobilesamv2.eval()
        self.predictor = SamPredictor(self.mobilesamv2)
        self.imgsz = 256
        self.resize_transform = ResizeLongestSide(self.imgsz)
        self.num_classes = 100

        # self.tracker = BYTETracker(args, frame_rate=args.fps)
        # self.timer = Timer()

        self.classifier = torch.nn.Conv2d(self.gaussians.num_objects, self.num_classes, kernel_size=1)
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.cls_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=5e-4)
        self.classifier.cuda()

        self.loss_rgb_weight = 1.0
        self.loss_obj_weight = 1.0
        self.loss_obj_3d_weight = 1.0

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

            # new_objects = self.generate_objects(image_resized)
            # # object_mask = cv2.resize(new_objects.astype(np.uint8),
            # #                         (self.W, self.H),
            # #                         interpolation=cv2.INTER_NEAREST).astype(np.float32)
            # object_mask = new_objects

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
            return

        pts, cols = read_xyz_rgb_from_raw(cloud_msg)
        synced = {
            "image": image_resized,  # ground truth image
            # "object_mask": object_mask,
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
            object_mask = np.ones((self.H, self.W), dtype=np.float32)
        # Create a SharedCam for rendering.
            viewpoint_cam = SharedCam(
                FoVx=focal2fov(self.fx, self.W),
                FoVy=focal2fov(self.fy, self.H),
                image=synced["image"],
                # object_mask=synced["object_mask"],
                object_mask=object_mask,
                original_pose=odom_msg.pose.pose,
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
            viewpoint_cam.setup_cam(new_rotation, new_translation, synced["image"], object_mask)

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
        object_mask = scene["object_mask"]
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
            object_mask=object_mask,
            cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy
        )
        viewpoint_cam.on_cuda()
        viewpoint_cam.setup_cam(new_rotation, new_translation, image_np, object_mask)

        # Render the scene.
        render_pkg = render_4(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
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
    
    def batch_iterator(self, batch_size: int, *args) -> Generator[List[Any], None, None]:
        assert len(args) > 0 and all(
            len(a) == len(args[0]) for a in args
        ), "Batched iteration must have inputs of all the same size."
        n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
        for b in range(n_batches):
            yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

    def generate_grayscale_mask_torch(self, anns, device='cuda'):
        """
        anns: 객체별 마스크가 담긴 텐서. shape: (N, H, W), 여기서 N은 객체 수.
        반환값: 각 객체별로 고유한 랜덤 그레이스케일 값을 가지는 마스크. shape: (H, W)
        """
        if len(anns) == 0:
            return None

        # 텐서의 해상도에 맞는 빈 그레이스케일 마스크 초기화 (배경은 0)
        height, width = anns.shape[1], anns.shape[2]
        grayscale_mask = torch.zeros((height, width), dtype=torch.long, device=device)
        
        self.used_labels = set()  # 중복 방지를 위한 집합
        # 고유한 그레이스케일 값을 객체마다 랜덤으로 부여
        for ann_idx in range(min(anns.shape[0], self.num_classes)):
            m = anns[ann_idx].bool()  # 현재 객체 마스크를 불리언으로 변환
            # 1부터 self.num_classes 범위에서 랜덤 값 선택
            random_label = random.randint(1, self.num_classes - 1)

            # 중복되지 않은 값이 나올 때까지 반복
            while random_label in self.used_labels:
                random_label = random.randint(1, self.num_classes - 1)

            self.used_labels.add(random_label)  # 사용된 라벨 추가
            grayscale_mask[m] = random_label  # 객체별로 랜덤 라벨 할당

        # print(self.used_labels)
        return grayscale_mask
    
    def generate_objects(self, gt_image):
        # Convert the gt_image tensor to a NumPy array
        gt_image_np = gt_image.permute(1, 2, 0).cpu().numpy()
        gt_image_np = np.clip(gt_image_np * 255, 0, 255).astype(np.uint8)
        # gt_image_np = gt_image

        obj_results = self.ObjAwareModel(gt_image_np, device='cuda', retina_masks=True, imgsz=self.imgsz, conf=0.4, iou=0.9)
        if obj_results is None or len(obj_results) == 0:
            image_shape = self.gt_image_np.shape
            return torch.zeros((image_shape[1], image_shape[2]), dtype=torch.long)
        # print(obj_results)

        self.predictor.set_image(gt_image_np)
        # print(obj_results[0].boxes)
        input_boxes1 = obj_results[0].boxes.xyxy
        input_boxes = input_boxes1.cpu().numpy()
        input_boxes = self.predictor.transform.apply_boxes(input_boxes, self.predictor.original_size)
        input_boxes = torch.from_numpy(input_boxes).cuda()
        sam_mask=[]
        image_embedding = self.predictor.features
        image_embedding = torch.repeat_interleave(image_embedding, 32, dim=0)
        prompt_embedding = self.mobilesamv2.prompt_encoder.get_dense_pe()
        prompt_embedding = torch.repeat_interleave(prompt_embedding, 32, dim=0)
        # print(input_boxes)
        for (boxes,) in self.batch_iterator(32, input_boxes):
            with torch.no_grad():
                image_embedding=image_embedding[0:boxes.shape[0],:,:,:]
                prompt_embedding=prompt_embedding[0:boxes.shape[0],:,:,:]
                sparse_embeddings, dense_embeddings = self.mobilesamv2.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,)
                low_res_masks, _ = self.mobilesamv2.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=prompt_embedding,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )
                low_res_masks = self.predictor.model.postprocess_masks(low_res_masks, self.predictor.input_size, self.predictor.original_size)
                sam_mask_pre = (low_res_masks > self.mobilesamv2.mask_threshold)*1.0
                sam_mask.append(sam_mask_pre.squeeze(1))
        sam_mask=torch.cat(sam_mask)
        annotation = sam_mask
        areas = torch.sum(annotation, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=True)
        gt_show_img = annotation[sorted_indices]

        grayscale_mask = self.generate_grayscale_mask_torch(gt_show_img)
        return grayscale_mask

    # 포인트 클라우드를 투영하고 시각화하는 함수 예시
    def project_and_visualize(self, viewpoint_cam):
        # 1. gt_image와 gt_objects 처리
        gt_image = viewpoint_cam.original_image.cuda()  # (C, H, W) 텐서, GPU상에 있음
        gt_objects = self.generate_objects(gt_image)
        viewpoint_cam.setup_obj(gt_objects.long().cuda())
        
        # 2. 포인트 클라우드 얻기 (예: (N, 3) tensor)
        points = self.gaussians.get_xyz.detach()  # GPU상의 tensor

        # 3. 카메라 내·외부 파라미터 가져오기
        world_view_transform = viewpoint_cam.world_view_transform
        if not isinstance(world_view_transform, torch.Tensor):
            world_view_transform = torch.from_numpy(world_view_transform).to(points.device)
        
        # image_width, image_height가 Tensor인 경우 Python int로 변환
        image_width = int(viewpoint_cam.image_width.item() if isinstance(viewpoint_cam.image_width, torch.Tensor) 
                        else viewpoint_cam.image_width)
        image_height = int(viewpoint_cam.image_height.item() if isinstance(viewpoint_cam.image_height, torch.Tensor) 
                        else viewpoint_cam.image_height)

        device = points.device
        fx = torch.tensor(viewpoint_cam.fx, device=device)
        cx = torch.tensor(viewpoint_cam.cx, device=device)
        fy = torch.tensor(viewpoint_cam.fy, device=device)
        cy = torch.tensor(viewpoint_cam.cy, device=device)

        # 4. 포인트들을 동차 좌표계로 변환: (N, 3) -> (N, 4)
        ones = torch.ones(points.shape[0], 1, device=points.device)
        points_hom = torch.cat([points, ones], dim=1)  # (N, 4)

        # 5. world_view_transform을 이용하여 월드 좌표 -> 카메라 좌표로 변환
        points_cam_hom = (world_view_transform @ points_hom.T).T  # (N, 4)
        points_cam = points_cam_hom[:, :3]  # (N, 3)

        # 6. 카메라 내부 파라미터를 사용해 포인트 투영 (핀홀 카메라 모델)
        X = points_cam[:, 0]
        Y = points_cam[:, 1]
        Z = points_cam[:, 2] + 1e-6  # 0 division 방지

        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

        # 7. 정수형 픽셀 좌표로 변환 및 이미지 범위 내 클리핑
        u_int = torch.clamp(u.round().long(), 0, image_width - 1)
        v_int = torch.clamp(v.round().long(), 0, image_height - 1)

        # 8. 시각화를 위해 gt_image를 numpy array로 변환 (C, H, W) -> (H, W, C)
        if gt_image.dim() == 3 and gt_image.shape[0] in [1, 3]:
            gt_image_np = gt_image.cpu().permute(1, 2, 0).numpy().copy()
        else:
            gt_image_np = gt_image.cpu().numpy().copy()

        # 9. 투영된 각 점에 대해 원(circle)을 그려 표시 (여기서는 파란색: (255, 0, 0))
        for x, y in zip(u_int.cpu().numpy(), v_int.cpu().numpy()):
            cv2.circle(gt_image_np, (int(x), int(y)), radius=2, color=(255, 0, 0), thickness=-1)

        # 10. 결과 시각화: OpenCV 창으로 출력 (BGR 순서이므로 색상 변환)
        gt_image_bgr = cv2.cvtColor(gt_image_np, cv2.COLOR_RGB2BGR)
        cv2.imshow("Projected Points", gt_image_bgr)
        cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(30)
        self.gaussians.training_setup(self)
        self.gaussians.spatial_lr_scale = self.scene_extent
        self.gaussians.update_learning_rate(1)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree

        # Define a timeout threshold (in seconds)
        TIMEOUT_THRESHOLD = 10.0

        # # optical flow를 위한 파라미터 설정
        # feature_params = dict(maxCorners=100,
        #                     qualityLevel=0.3,
        #                     minDistance=7,
        #                     blockSize=7)

        # lk_params = dict(winSize=(15, 15),
        #                 maxLevel=2,
        #                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # # 이전 프레임과 추적할 점을 저장할 변수 초기화
        # prev_gray = None
        # prev_pts = None

        # # ORB detector와 BFMatcher 초기화 (ORB는 1000개의 특징점을 검출)
        # orb = cv2.ORB_create(1000)
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # # 이전 프레임 정보를 저장할 변수들
        # prev_gray = None
        # prev_kps = None
        # prev_des = None

        # # LoFTR 모델 초기화 (outdoor configuration 예시)
        # loftr = LoFTR(pretrained=True, config=default_cfg).cuda().eval()

        # # 이전 프레임을 저장할 변수 초기화 (None이면 첫 프레임으로 간주)
        # prev_image_np = None

        # 이전 프레임 정보를 저장할 변수들 (초기에는 None)
        prev_pose = None
        prev_mask_colors = None  # 1D array: 이전 프레임에서 투영된 마스크 intensity (예: shape (N,))
        prev_u_int = None  # 이전 프레임 투영 좌표 (참고용)
        prev_v_int = None
        prev_gt_objects_np = None  # 이전 프레임의 gt_objects 마스크 (numpy array)

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

            # # 현재 프레임을 그레이스케일로 변환 (LoFTR는 그레이스케일 이미지를 입력으로 사용)
            # curr_gray_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

            # # 첫 프레임인 경우, 이전 프레임 변수에 저장하고 다음 루프로 넘어감
            # if prev_image_np is None:
            #     prev_image_np = curr_gray_np.copy()
            #     continue

            # # 이미지 전처리: float tensor로 변환, [1,1,H,W] shape, 0~1 범위 정규화
            # img0 = torch.from_numpy(prev_image_np.astype(np.float32) / 255.0)[None, None].cuda()
            # img1 = torch.from_numpy(curr_gray_np.astype(np.float32) / 255.0)[None, None].cuda()

            # input_dict = {"image0": img0, "image1": img1}

            # # LoFTR를 이용해 dense 매칭 수행 (추론 시 no_grad 사용)
            # with torch.no_grad():
            #     loftr(input_dict)
            # # LoFTR는 매칭 결과를 "mkpts0"와 "mkpts1"에 저장함.
            # mkpts0 = input_dict["mkpts0"].cpu().numpy()  # 이전 이미지의 매칭 점들, shape (N, 2)
            # mkpts1 = input_dict["mkpts1"].cpu().numpy()  # 현재 이미지의 매칭 점들, shape (N, 2)

            # # 시각화를 위해 두 이미지를 side-by-side로 결합 (원본 컬러 이미지 사용)
            # h0, w0 = prev_image_np.shape
            # h1, w1 = image_np.shape[:2]
            # height = max(h0, h1)
            # combined_img = np.zeros((height, w0 + w1, 3), dtype=np.uint8)
            # # 이전 프레임은 그레이스케일이라 컬러 변환
            # prev_img_color = cv2.cvtColor(prev_image_np, cv2.COLOR_GRAY2BGR)
            # combined_img[:h0, :w0, :] = prev_img_color
            # combined_img[:h1, w0:w0+w1, :] = image_np

            # # 매칭 결과 시각화: 일부 매칭 점들만 샘플링해서 선과 원으로 표시
            # num_matches = mkpts0.shape[0]
            # sample_idx = np.arange(num_matches)
            # if num_matches > 100:
            #     sample_idx = np.random.choice(num_matches, 100, replace=False)

            # for i in sample_idx:
            #     pt0 = mkpts0[i]
            #     pt1 = mkpts1[i]
            #     pt0_int = (int(pt0[0]), int(pt0[1]))
            #     pt1_int = (int(pt1[0] + w0), int(pt1[1]))  # 두 번째 이미지의 x좌표는 w0 만큼 offset됨
            #     cv2.circle(combined_img, pt0_int, 3, (0, 255, 0), -1)
            #     cv2.circle(combined_img, pt1_int, 3, (0, 255, 0), -1)
            #     cv2.line(combined_img, pt0_int, pt1_int, (255, 0, 0), 1)

            # cv2.imshow("Kornia LoFTR Matches", combined_img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # # 현재 프레임을 이전 프레임으로 업데이트
            # prev_image_np = curr_gray_np.copy()

            # # 현재 프레임 이미지를 그레이스케일로 변환 (특징점 검출을 위해)
            # curr_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            # curr_kps, curr_des = orb.detectAndCompute(curr_gray, None)

            # # 이전 프레임의 특징점과 현재 프레임의 특징점을 매칭
            # if prev_des is not None and curr_des is not None:
            #     matches = bf.match(prev_des, curr_des)
            #     matches = sorted(matches, key=lambda x: x.distance)
            #     # 매칭 결과 시각화: 이전 프레임과 현재 프레임의 매칭된 특징점을 이어서 보여줌
            #     img_matches = cv2.drawMatches(prev_gray, prev_kps, curr_gray, curr_kps, matches, None, flags=2)
            #     cv2.imshow("Feature Matches", img_matches)
            # else:
            #     # 첫 프레임이거나 매칭할 특징점이 없으면 현재 프레임의 특징점을 표시
            #     img_kp = cv2.drawKeypoints(curr_gray, curr_kps, None, color=(0, 255, 0), flags=0)
            #     cv2.imshow("Feature Matches", img_kp)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # # 현재 프레임 정보를 이전 프레임 변수에 업데이트
            # prev_gray = curr_gray.copy()
            # prev_kps = curr_kps
            # prev_des = curr_des

            # # optical flow 추적을 위한 현재 프레임 그레이스케일 이미지 변환
            # # image_np가 BGR 형식이라면 아래처럼; 만약 RGB라면 COLOR_RGB2GRAY로 변경
            # curr_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

            # if prev_gray is None:
            #     # 첫 프레임인 경우, 추적할 특징점(코너)을 추출
            #     prev_gray = curr_gray.copy()
            #     prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            #     continue

            # # Lucas-Kanade optical flow를 이용해 이전 프레임(prev_gray)의 점(prev_pts)이 현재 프레임(curr_gray)에서 어떻게 이동했는지 계산
            # curr_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

            # # 추적이 성공한 점들만 선택
            # if curr_pts is None or prev_pts is None:
            #     # 만약 점이 없으면 다시 특징점을 추출
            #     prev_gray = curr_gray.copy()
            #     prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            #     continue

            # good_new = curr_pts[st == 1]
            # good_old = prev_pts[st == 1]

            # # optical flow 결과를 시각화: 이전 위치에서 현재 위치로 화살표 그리기
            # for (old, new) in zip(good_old, good_new):
            #     x_old, y_old = old.ravel()
            #     x_new, y_new = new.ravel()
            #     cv2.arrowedLine(image_np, (int(x_old), int(y_old)), (int(x_new), int(y_new)),
            #                     color=(0, 255, 0), thickness=2, tipLength=0.3)

            # # 결과 시각화: OpenCV 창에 표시
            # cv2.imshow("Optical Flow Tracking", image_np)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # # 현재 프레임을 이전 프레임으로 업데이트하고, 추적 점들도 업데이트
            # prev_gray = curr_gray.copy()
            # prev_pts = good_new.reshape(-1, 1, 2)

            # # 누적된 포인트를 numpy 배열로 저장 (self.accumulated_points)
            # if not hasattr(self, "accumulated_points"):
            #     # 처음에는 pts로 초기화
            #     self.accumulated_points = pts.copy()
            # else:
            #     # 이후엔 기존 accumulated_points에 새로운 pts를 수직으로 연결
            #     self.accumulated_points = np.vstack([self.accumulated_points, pts])

            # # Convert the gt_image tensor to a NumPy array
            # # gt_image_np = gt_image.permute(1, 2, 0).cpu().numpy()
            # # gt_image_np = np.clip(gt_image_np * 255, 0, 255).astype(np.uint8)
            # gt_image_np = image_np

            # obj_results = self.ObjAwareModel(gt_image_np, device='cuda', retina_masks=True, imgsz=self.imgsz, conf=0.4, iou=0.9)
            # if obj_results is None or len(obj_results) == 0:
            #     image_shape = self.gt_image_np.shape
            #     return torch.zeros((image_shape[1], image_shape[2]), dtype=torch.long)
            # # print(obj_results)

            # self.predictor.set_image(gt_image_np)
            # # print(obj_results[0].boxes.data)
            # input_boxes1 = obj_results[0].boxes.xyxy
            # input_boxes2 = obj_results[0].boxes.data
            # input_boxes = input_boxes1.cpu().numpy()
            # input_boxes = self.predictor.transform.apply_boxes(input_boxes, self.predictor.original_size)
            # input_boxes = torch.from_numpy(input_boxes).cuda()

            # confidence = input_boxes2[:, 4:5]  # confidence만 따로 추출 (N,1)
            # class_id = input_boxes2[:, -1].unsqueeze(1)  # 클래스 id (N, 1)

            # # 최종 tensor 조합
            # input_boxes2 = torch.cat([
            #     input_boxes2[:, :4],       # x1, y1, x2, y2
            #     input_boxes2[:, 4:5],      # bbox confidence
            #     input_boxes2[:, 4:5],      # class confidence를 bbox confidence로 복사
            #     class_id                   # class id
            # ], dim=1)

            # # gt_image_np는 원본 이미지 넘파이 배열 (H, W, 3), RGB 형태라고 가정
            # box_image = gt_image_np.copy()

            # # bounding box 정보를 tensor에서 numpy로 변환
            # boxes_np = input_boxes2.detach().cpu().numpy()

            # # 각 박스를 이미지에 그리기
            # print(len(input_boxes2))
            # for box in input_boxes2:
            #     x1, y1, x2, y2, score, cls_id = box.cpu().numpy()
            #     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
            #     # # 점수가 너무 낮은 경우는 생략 가능 (예시: score < 0.5 제외)
            #     # if score < 0.5:
            #     #     continue
                
            #     # 이미지에 사각형 그리기 (빨간색 박스)
            #     box_image = cv2.rectangle(
            #         box_image.copy(),
            #         (x1, y1),
            #         (x2, y2),
            #         color=(0, 255, 0),
            #         thickness=2
            #     )

            # rr.log("box_image", rr.Image(box_image))

            # if input_boxes2 is not None:
            #     # print(input_boxes2)
            #     img_height, img_width = gt_image_np.shape[:2]
            #     img_size = (img_width, img_height)  # 원본 이미지의 width, height를 명시적으로 사용

            #     # tracker.update() 호출 시 exp.test_size 대신 원본 이미지 크기 사용
            #     online_targets = self.tracker.update(input_boxes2, [self.H, self.W], img_size=img_size)
            #     # print(online_targets)
            #     online_tlwhs = []
            #     online_ids = []
            #     online_scores = []
            #     for t in online_targets:
            #         tlwh = t.tlwh
            #         tid = t.track_id
            #         vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            #         if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
            #             online_tlwhs.append(tlwh)
            #             online_ids.append(tid)
            #             online_scores.append(t.score)
            #             # print(f"{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n")

            #     self.timer.toc()

            #     if len(online_targets) > 0:
            #         # ìë³¸ ì´ë¯¸ì§ ë³µì¬
            #         box_image = gt_image_np.copy()

            #         # online_targetsìì ë°ë¡ bounding box ê·¸ë¦¬ê¸°
            #         for t in online_targets:
            #             x, y, w, h = t.tlwh
            #             x1, y1, x2, y2 = map(int, [x, y, x + w, y + h])

            #             cv2.rectangle(
            #                 box_image,
            #                 (x1, y1),
            #                 (x2, y2),
            #                 (0, 255, 0),
            #                 thickness=2
            #             )

            #             # IDì confidence score íì (ì íì¬í­)
            #             cv2.putText(
            #                 box_image,
            #                 f'ID:{t.track_id} {t.score:.2f}',
            #                 (x1, max(y1 - 10, 0)),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.5,
            #                 (255, 0, 0),
            #                 2
            #             )

            #         # ìê°íë ì´ë¯¸ì§ë¥¼ rrì ì ë¬
            #         rr.log("box_image", rr.Image(box_image))


            if self.mapping_new_cams:
                viewpoint_cam = self.mapping_new_cams.pop(0)
                # 현재 프레임의 이미지와 gt_objects 생성 및 설정
                gt_image = viewpoint_cam.original_image.cuda()  # (C,H,W)
                gt_objects = self.generate_objects(gt_image)
                # 초기 마스크 설정 (업데이트 이후 다시 적용될 예정)
                viewpoint_cam.setup_obj(gt_objects.long().cuda())

                # points는 변하지 않는 월드 좌표계의 포인트 클라우드 (GPU tensor)
                points = self.gaussians.get_xyz.detach()  # (N,3)

                # 현재 프레임의 pose
                curr_pose = viewpoint_cam.original_pose

                # 현재 프레임의 gt_objects를 그레이스케일 numpy 배열로 변환
                curr_gt_objects_np = gt_objects.cpu().squeeze().numpy().copy()
                if curr_gt_objects_np.max() <= 1.0:
                    curr_gt_objects_np = (curr_gt_objects_np * 255).astype(np.uint8)
                else:
                    curr_gt_objects_np = curr_gt_objects_np.astype(np.uint8)

                # 현재 프레임의 투영 좌표 계산 (GPU에서 실행)
                u_int_curr, v_int_curr = project_points(points, curr_pose, viewpoint_cam)
                # 현재 프레임의 투영된 마스크 intensity 추출 (numpy)
                curr_mask_colors = curr_gt_objects_np[v_int_curr, u_int_curr]

                # mapping 분석: 이전 프레임 정보(prev_mask_colors)가 있으면 mapping_dict 구성
                mapping_dict = {}
                if prev_mask_colors is not None:
                    common_length = min(prev_mask_colors.shape[0], curr_mask_colors.shape[0])
                    prev_common = prev_mask_colors[:common_length]
                    curr_common = curr_mask_colors[:common_length]
                    unique_prev = np.unique(prev_common)
                    for prev_val in unique_prev:
                        idx_prev = np.where(prev_common == prev_val)[0]
                        if len(idx_prev) == 0:
                            continue
                        curr_vals, counts = np.unique(curr_common[idx_prev], return_counts=True)
                        for curr_val, cnt in zip(curr_vals, counts):
                            ratio = cnt / len(idx_prev)
                            if ratio >= THRESHOLD_PERCENT:
                                mapping_dict[int(curr_val)] = int(prev_val)
                                # print(f"Mapping {prev_val} -> {curr_val} covers {ratio*100:.1f}%")
                else:
                    mapping_dict = {}

                # 현재 프레임의 마스크 업데이트: mapping_dict에 따라 변경
                updated_mask_np = curr_gt_objects_np.copy()
                for curr_val, prev_val in mapping_dict.items():
                    updated_mask_np[updated_mask_np == curr_val] = prev_val
                    self.used_labels.discard(curr_val)

                # 업데이트된 마스크를 (H,W)에서 배치 차원을 추가하여 (1,H,W)로 만듭니다.
                updated_mask_tensor = torch.from_numpy(updated_mask_np)
                # 빠르게 반영: setup_obj 호출 (GPU로 전송)
                viewpoint_cam.setup_obj(updated_mask_tensor.long().cuda())

                # 업데이트 후, 현재 프레임의 정보를 이전 프레임 정보로 저장
                # (mapping 분석은 계속해서 누적해서 진행될 수 있음)
                prev_mask_colors = curr_mask_colors.copy()

                self.training = True
                render_pkg = render_4(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                image_rendered = render_pkg["render"]
                objects = render_pkg["render_object"]
                # depth_image = render_pkg["render_depth"]

                gt_obj = gt_objects
                logits = self.classifier(objects)
                pred_obj = torch.argmax(logits,dim=0)
                pred_obj_mask = self.visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
                rgb_mask = self.feature_to_rgb(objects)
                loss_obj = self.cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
                loss_obj = loss_obj / torch.log(torch.tensor(self.num_classes))

                Ll1_map, Ll1 = l1_loss(image_rendered, gt_image)
                L_ssim_map, L_ssim = ssim(image_rendered, gt_image)
                loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - L_ssim)

                loss_obj_3d = None
                if self.train_iter % 5 == 0:
                    # regularize at certain intervals
                    logits3d = self.classifier(self.gaussians._objects_dc.permute(2,0,1))
                    prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
                    loss_obj_3d = loss_cls_3d(self.gaussians._xyz.squeeze().detach(), prob_obj3d, 5, 2, 300000, 1000)
                    loss = (self.loss_rgb_weight * loss_rgb
                            + self.loss_obj_weight * loss_obj
                            + self.loss_obj_3d_weight * loss_obj_3d)
                    # print(self.loss_rgb_weight * loss_rgb,
                    #         self.loss_obj_weight * loss_obj,
                    #         self.loss_obj_3d_weight * loss_obj_3d)
                else:
                    loss = (self.loss_rgb_weight * loss_rgb
                            + self.loss_obj_weight * loss_obj)
                
                # loss = (self.loss_rgb_weight * loss_rgb
                #             + self.loss_obj_weight * loss_obj)
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

                # rendered_depth_np = depth_image.detach().cpu().numpy().transpose(1, 2, 0)
                # rendered_depth_np = np.clip(rendered_depth_np, 0, 1) * 255
                # rendered_depth_np = rendered_depth_np.astype(np.uint8)
                # rr.log("rendered_depth_image", rr.Image(rendered_depth_np))

                # gt_objects_np = gt_objects.cpu().numpy()
                # rr.log("gt_objects", rr.Image(gt_objects_np))
                # cv2.imwrite(f"/home/irl/Workspace_Hyundo/catkin_ws/saved_masks/{self.train_iter}.png", gt_objects_np)
                gt_rgb_mask = self.visualize_obj(viewpoint_cam.objects.cpu().numpy().astype(np.uint8))
                rr.log("gt_rgb_mask", rr.Image(gt_rgb_mask))

                rr.log("rgb_mask", rr.Image(rgb_mask))
                rr.log("pred_obj_mask", rr.Image(pred_obj_mask))

                new_rotation = viewpoint_cam.R.cpu().numpy()
                new_translation = viewpoint_cam.t.cpu().numpy()
                # print(f"{self.train_iter}: \n{new_rotation}\n{new_translation}")
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
                rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(pts, colors=cols, radii=0.01))
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
                gt_objects = viewpoint_cam.objects.cuda()
                self.training = True
                render_pkg = render_4(scene, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                image_rendered = render_pkg["render"]
                objects = render_pkg["render_object"]

                gt_obj = gt_objects
                logits = self.classifier(objects)
                # pred_obj = torch.argmax(logits,dim=0)
                # pred_obj_mask = self.visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
                # rgb_mask = self.feature_to_rgb(objects)
                loss_obj = self.cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
                loss_obj = loss_obj / torch.log(torch.tensor(self.num_classes))

                Ll1_map, Ll1 = l1_loss(image_rendered, gt_image)
                L_ssim_map, L_ssim = ssim(image_rendered, gt_image)
                loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - L_ssim)

                loss_obj_3d = None
                if self.train_iter % 5 == 0:
                    # regularize at certain intervals
                    logits3d = self.classifier(self.gaussians._objects_dc.permute(2,0,1))
                    prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
                    loss_obj_3d = loss_cls_3d(self.gaussians._xyz.squeeze().detach(), prob_obj3d, 5, 2, 300000, 1000)
                    loss = (self.loss_rgb_weight * loss_rgb
                            + self.loss_obj_weight * loss_obj
                            + self.loss_obj_3d_weight * loss_obj_3d)
                    # print(self.loss_rgb_weight * loss_rgb,
                    #         self.loss_obj_weight * loss_obj,
                    #         self.loss_obj_3d_weight * loss_obj_3d)
                else:
                    loss = (self.loss_rgb_weight * loss_rgb
                            + self.loss_obj_weight * loss_obj)
                    
                # loss = loss_rgb
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
                    object_mask = scene["object_mask"]
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
                        object_mask=object_mask,
                        cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy
                    )
                    viewpoint_cam.on_cuda()
                    viewpoint_cam.setup_cam(new_rotation, new_translation, image_np)

                    gt_image = viewpoint_cam.original_image.cuda()
                    self.training = True
                    render_pkg = render_4(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                    image_rendered = render_pkg["render"]
                    objects = render_pkg["render_object"]

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
    # parser = make_parser()
    # args, unknown = parser.parse_known_args()
    # exp = get_exp(args.exp_file, args.name)
    gs = GaussianSplatting()
    # Run the training loop in a separate thread.
    training_thread = threading.Thread(target=gs.run)
    training_thread.daemon = True
    training_thread.start()
    # Use a multi-threaded spinner to allow callbacks to run concurrently.
    spinner = rospy.spin()
