from cgi import print_directory
from curses import keyname
from email.policy import strict
from http.client import NON_AUTHORITATIVE_INFORMATION
from ntpath import join
from operator import imod
import os
import pickle
from pickletools import optimize
import random
from select import select
from sys import path
from tabnanny import verbose
from tkinter.messagebox import NO
from turtle import color
from typing import KeysView
from unicodedata import name
from cv2 import norm
from core.utils.network_util import MotionBasisComputer
from torch import optim

import numpy as np
import cv2
import torch
import torch.utils.data

import trimesh

from core.utils.image_util import load_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.file_util import list_files, split_path
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox

from configs import cfg
from third_parties.smpl.smpl_numpy import SMPL
from tqdm import trange
MODEL_DIR = 'third_parties/smpl/models'
from core.nets.monohuman.network import Network


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_path,
            index_a,
            index_b,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            skip=1,
            views=('1',),
            **_):

        print('[Dataset Path]', dataset_path) 

        self.smpl_model = SMPL(sex='neutral', model_dir=MODEL_DIR)

        self.dataset_path = dataset_path

        self.image_dir = os.path.join(dataset_path, 'images')

        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()

        if 'motion_weights_priors' in keyfilter:
            self.motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    self.canonical_joints,   
                    self.canonical_bbox['min_xyz'],
                    self.canonical_bbox['max_xyz'],
                    grid_size=cfg.mweight_volume.volume_size).astype('float32')


        self.cameras = self.load_train_cameras()
        self.train_cam = '01'
        self.cam_names = views

        self.mesh_infos = self.load_train_mesh_infos()

        self.framelist = self.load_train_frames()

        ## for training
        # self.framelist = self.framelist[:-(len(self.framelist) // 5)]
        print(f"SKIP: {skip}")
        print(f"BEFORE SKIP: {len(self.framelist)}")
        self.framelist = self.framelist[::skip]
        print(f"AFTER SKIP: {len(self.framelist)}")
        print('test--movement set--')
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]
        print(f' -- Total Frames: {self.get_total_frames()}')


        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        self.ray_shoot_mode = ray_shoot_mode
        self.index_a = index_a
        self.index_b = index_b


        bgcolor = np.zeros(3)

        frame_name_a = f'frame_{self.index_a:06d}'
        frame_name_b = f'frame_{self.index_b:06d}'


        self.in_K = []
        self.in_E = []

        self.in_dst_poses = []
        self.in_dst_tposes_joints = []

        self.in_frame_name = [frame_name_a, frame_name_b]
        self.in_index = [self.index_a, self.index_b]

        for in_idx, in_name in zip(self.in_index, self.in_frame_name):
            cameras = self.cameras

            K_ = cameras[self.train_cam]['intrinsics'].copy()
            K_[:2] *= cfg.resize_img_scale

            in_skel_info = self.query_dst_skeleton(in_name)
            pose_ = in_skel_info['poses']
            tpose_joints_ = in_skel_info['dst_tpose_joints']
            E_ = cameras[self.train_cam]['extrinsics']
            E_ = apply_global_tfm_to_camera(
                    E=E_,
                    Rh=in_skel_info['Rh'],
                    Th=in_skel_info['Th'])
            R_ = E_[:3, :3]
            T_ = E_[:3, 3]

            self.in_K.append(K_.astype('float32'))
            self.in_E.append(E_.astype('float32'))

            self.in_dst_poses.append(pose_)
            self.in_dst_tposes_joints.append(tpose_joints_)


        self.in_dst_Rs = []
        self.in_dst_Ts = []
        for i in range(len(self.in_dst_poses)):
            dst_Rs_, dst_Ts_ = body_pose_to_body_RTs(
                self.in_dst_poses[i], self.in_dst_tposes_joints[i]
            )
            self.in_dst_Rs.append(dst_Rs_)
            self.in_dst_Ts.append(dst_Ts_)

        img_a, _ = self.load_image(self.train_cam, frame_name_a, bgcolor)
        img_a = (img_a / 255.).astype('float32')
        img_b, _ = self.load_image(self.train_cam, frame_name_b, bgcolor)
        img_b = (img_b / 255.).astype('float32')
        self.src_img = np.array([img_a, img_b])


    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        return canonical_joints, canonical_bbox

    def load_train_cameras(self):
        with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f:
            cameras = pickle.load(f)
        return cameras

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):

        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:
            mesh_infos = pickle.load(f)
        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox
        return mesh_infos

    def load_train_frames(self):
        img_paths = list_files(os.path.join(self.dataset_path, 'images', self.train_cam,),
                            exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32')
        }

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, ray_img, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, near, far
    
    def get_patch_ray_indices(
            self, 
            N_patch, 
            ray_mask, 
            subject_mask, 
            bbox_mask,
            patch_size, 
            H, W):

        assert subject_mask.dtype == np.bool
        assert bbox_mask.dtype == np.bool

        bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )

        list_ray_indices = []
        list_mask = []
        list_xy_min = []
        list_xy_max = []

        total_rays = 0
        patch_div_indices = [total_rays]
        for _ in range(N_patch):
            # let p = cfg.patch.sample_subject_ratio
            # prob p: we sample on subject area
            # prob (1-p): we sample on non-subject area but still in bbox
            if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
                candidate_mask = subject_mask
            else:
                candidate_mask = bbox_exclude_subject_mask

            ray_indices, mask, xy_min, xy_max = \
                self._get_patch_ray_indices(ray_mask, candidate_mask, 
                                            patch_size, H, W)

            assert len(ray_indices.shape) == 1
            total_rays += len(ray_indices)

            list_ray_indices.append(ray_indices)
            list_mask.append(mask)
            list_xy_min.append(xy_min)
            list_xy_max.append(xy_max)
            
            patch_div_indices.append(total_rays)

        select_inds = np.concatenate(list_ray_indices, axis=0)
        patch_info = {
            'mask': np.stack(list_mask, axis=0),
            'xy_min': np.stack(list_xy_min, axis=0),
            'xy_max': np.stack(list_xy_max, axis=0)
        }
        patch_div_indices = np.array(patch_div_indices)

        return select_inds, patch_info, patch_div_indices


    def _get_patch_ray_indices(
            self, 
            ray_mask, 
            candidate_mask, 
            patch_size, 
            H, W):

        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool
        assert candidate_mask.dtype == np.bool

        valid_ys, valid_xs = np.where(candidate_mask)

        # determine patch center
        select_idx = np.random.choice(valid_ys.shape[0], 
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        # determine patch boundary
        half_patch_size = patch_size // 2
        x_min = np.clip(a=center_x-half_patch_size, 
                        a_min=0, 
                        a_max=W-patch_size)
        x_max = x_min + patch_size
        y_min = np.clip(a=center_y-half_patch_size,
                        a_min=0,
                        a_max=H-patch_size)
        y_max = y_min + patch_size

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max, x_min:x_max] = True

        #####################################################
        ## Below we determine the selected ray indices
        ## and patch valid mask

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]
        
        inter_mask = inter_mask.reshape(H, W)

        return select_inds, \
                inter_mask[y_min:y_max, x_min:x_max], \
                np.array([x_min, y_min]), np.array([x_max, y_max])

    def load_image(self, cam_name, frame_name, bg_color):

        imagepath = os.path.join(self.image_dir, cam_name, '{}.png'.format(frame_name))
        
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path,
                                'masks',
                                cam_name,
                                '{}.png'.format(frame_name))

        alpha_mask = np.array(load_image(maskpath))

        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask


    def get_total_frames(self):
        return len(self.framelist)

    def sample_patch_rays(self, img, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, ray_img, near, far):

        select_inds, patch_info, patch_div_indices = \
            self.get_patch_ray_indices(
                N_patch=cfg.patch.N_patches, 
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=cfg.patch.size, 
                H=H, W=W)

        rays_o, rays_d, ray_img, near, far = self.select_rays(
            select_inds, rays_o, rays_d, ray_img, near, far)
        
        targets = []
        for i in range(cfg.patch.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, ray_img, near, far, \
                target_patches, patch_masks, patch_div_indices

    def __len__(self):
        return len(self.cam_names) * self.get_total_frames()

    def __getitem__(self, idx):
        cam_idx, frame_idx = idx // self.get_total_frames(), idx % self.get_total_frames()

        cam_name = self.cam_names[cam_idx]
        frame_name = self.framelist[frame_idx]
        results = {
            'cam_name': cam_name,
            'frame_name': frame_name,
            'img_name': f'c{cam_name}_f{frame_name[6:]}'
        }


        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(cam_name, frame_name, bgcolor)
        img = (img / 255.).astype('float32')

        H, W = img.shape[0:2]
        src_img = self.src_img

        poses = self.mesh_infos[frame_name]['poses']
        betas = self.mesh_infos[frame_name]['beats']
        _, _, joints = self.smpl_model(poses, betas)

        # assert frame_name in self.cameras
        K = self.cameras[cam_name]['intrinsics'][:3, :3].copy()
        K[:2] *= cfg.resize_img_scale
        K = K.astype('float32')
        E = self.cameras[cam_name]['extrinsics']
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        E = E.astype('float32')
        R = E[:3, :3]
        T = E[:3, 3]
        
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        ray_img = img.reshape(-1, 3) 
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        if self.ray_shoot_mode == 'image':
            pass
        elif self.ray_shoot_mode == 'patch':
            rays_o, rays_d, ray_img, near, far, \
            target_patches, patch_masks, patch_div_indices = \
                self.sample_patch_rays(img=img, H=H, W=W,
                                       subject_mask=alpha[:, :, 0] > 0.,
                                       bbox_mask=ray_mask.reshape(H, W),
                                       ray_mask=ray_mask,
                                       rays_o=rays_o, 
                                       rays_d=rays_d, 
                                       ray_img=ray_img, 
                                       near=near, 
                                       far=far)
        else:
            assert False, f"Ivalid Ray Shoot Mode: {self.ray_shoot_mode}"
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor, 
                'ori_img': img,
                'src_imgs':src_img,
                'joints': joints,
                'canonical_joints': self.canonical_joints
                })

            if self.ray_shoot_mode == 'patch':
                results.update({
                    'patch_div_indices': patch_div_indices,
                    'patch_masks': patch_masks,
                    'target_patches': target_patches})

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = ray_img

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints
                )
            cnl_gtfms = get_canonical_global_tfms(
                            self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })

            in_dst_Rs = []
            in_dst_Ts = []
            for i in range(len(self.in_dst_poses)):
                dst_Rs_, dst_Ts_ = body_pose_to_body_RTs(
                    self.in_dst_poses[i], self.in_dst_tposes_joints[i]
                )
                in_dst_Rs.append(dst_Rs_)
                in_dst_Ts.append(dst_Ts_)
            results.update(
                {
                    'in_dst_Rs': np.array(in_dst_Rs),
                    'in_dst_Ts': np.array(in_dst_Ts)
                }
            )
        #print('len---in---dataset', len(in_dst_Rs))
        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = self.motion_weights_priors.copy()

        # get the bounding box of canonical volume
        if 'cnl_bbox' in self.keyfilter:
            min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })

        in_dst_posevec = []
        for posevec in self.in_dst_poses:
            in_dst_posevec_69 = posevec[3:] + 1e-2
            in_dst_posevec.append(in_dst_posevec_69)
        results.update({
            'in_dst_posevec': np.array(in_dst_posevec)
        })

        results.update({

            'in_K': np.array(self.in_K),
            'in_E': np.array(self.in_E),
            'E': E,
            'K': K
        }
        )
        return results






