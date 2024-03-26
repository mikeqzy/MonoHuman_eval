import os
import pickle
import json
import numpy as np

from core.utils.camera_util import rotate_camera_by_frame_idx

def get_freeview_camera():
    total_frames = 100

    subject = '377'
    dataset_path = f'dataset/zju_mocap_full/{subject}/'

    with open(os.path.join(dataset_path, 'cameras.pkl'), 'rb') as f:
        cameras = pickle.load(f)

    train_camera = cameras['12']
    extrinsics = train_camera['extrinsics']

    with open(os.path.join(dataset_path, 'mesh_infos.pkl'), 'rb') as f:
        mesh_infos = pickle.load(f)

    trans = mesh_infos['frame_000000']['Th'].astype('float32')

    cam_names = [str(cam_name) for cam_name in range(total_frames)]

    all_cam_params = {'all_cam_names': cam_names}
    for frame_idx, cam_name in enumerate(cam_names):
        E = rotate_camera_by_frame_idx(
            extrinsics=extrinsics,
            frame_idx=frame_idx,
            period=total_frames,
            trans=trans,
            rotate_axis='z',
            inv_angle=True,
        )
        R = E[:3,:3]
        T = E[:3,3:]
        K = train_camera['intrinsics']
        D = train_camera['distortions']

        cam_params = {
            'K': K.tolist(),
            'D': D.tolist(),
            'R': R.tolist(),
            'T': T.tolist(),
        }
        all_cam_params.update({cam_name: cam_params})

    with open(os.path.join(dataset_path, 'freeview_cam_params.json'), 'w') as f:
        json.dump(all_cam_params, f)


if __name__ == '__main__':
    get_freeview_camera()