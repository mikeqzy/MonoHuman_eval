import os
import sys
import glob

from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))

from third_parties.smpl.smpl_numpy import SMPL
from core.utils.file_util import split_path
from core.utils.image_util import load_image, save_image, to_3ch_image
import argparse

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    '387_test.yaml',
                    'the path of config file')
flags.DEFINE_string('seqid',
                    '0',
                    'ood pose sequence name')

MODEL_DIR = '../../third_parties/smpl/models'


seq_dict = {
    '0': 'gBR_sBM_cAll_d04_mBR1_ch05',
    '1': 'gBR_sBM_cAll_d04_mBR1_ch06',
    '2': 'MPI_Limits_03099_op8_poses'
}

def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_mask(subject_dir, img_name):
    msk_path = os.path.join(subject_dir, 'mask',
                            img_name)[:-4] + '.png'
    msk = np.array(load_image(msk_path))[:, :, 0]
    msk = (msk != 0).astype(np.uint8)

    msk_path = os.path.join(subject_dir, 'mask_cihp',
                            img_name)[:-4] + '.png'
    msk_cihp = np.array(load_image(msk_path))[:, :, 0]
    msk_cihp = (msk_cihp != 0).astype(np.uint8)

    msk = (msk | msk_cihp).astype(np.uint8)
    msk[msk == 1] = 255

    return msk


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    sex = cfg['dataset']['sex']
    max_frames = cfg['max_frames']
    seq_name = seq_dict[FLAGS.seqid]

    dataset_dir = cfg['dataset']['zju_mocap_path']
    # dataset_dir = '../../../../data/ZJUMoCap'
    subject_dir = os.path.join(dataset_dir, f"CoreView_{subject}")
    smpl_params_dir = os.path.join(subject_dir, f'{seq_name}_new_params_humannerf')

    # select_view = cfg['training_view']
    # cam_names = [f'{cam_name:02d}' for cam_name in range(1, 24)]
    cam_names = ['01']

    anno_path = os.path.join(subject_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()

    cameras = {}
    for cam_idx, cam_name in enumerate(cam_names):
        # load cameras
        cams = annots['cams']
        cam_Ks = np.array(cams['K'])[cam_idx].astype('float32')
        cam_Rs = np.array(cams['R'])[cam_idx].astype('float32')
        cam_Ts = np.array(cams['T'])[cam_idx].astype('float32') / 1000.
        cam_Ds = np.array(cams['D'])[cam_idx].astype('float32')

        K = cam_Ks     #(3, 3)
        D = cam_Ds[:, 0]
        E = np.eye(4)  #(4, 4)
        cam_T = cam_Ts[:3, 0]
        E[:3, :3] = cam_Rs
        E[:3, 3]= cam_T

        # write camera info
        cameras[cam_name] = {
            'intrinsics': K,
            'extrinsics': E,
            'distortions': D
        }

        # output_path = os.path.join(cfg['output']['dir'],
        #                            subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
        output_path = os.path.join('../../dataset/zju_mocap_full',
                                   subject if 'name' not in cfg['output'].keys() else cfg['output']['name'], seq_name)
        os.makedirs(output_path, exist_ok=True)

        # copy config file
        copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))

        smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)
        smpl_paths = sorted(glob.glob(os.path.join(smpl_params_dir, '*.npy')))

        mesh_infos = {}
        all_betas = []
        for idx_frame, smpl_path in enumerate(tqdm(smpl_paths)):
            out_name = 'frame_{:06d}'.format(idx_frame)

            # load smpl parameters
            smpl_params = np.load(smpl_path, allow_pickle=True).item()

            betas = smpl_params['shapes'][0] #(10,)
            poses = smpl_params['poses'][0]  #(72,)
            Rh = smpl_params['Rh'][0]  #(3,)
            Th = smpl_params['Th'][0]  #(3,)

            all_betas.append(betas)


            # write mesh info
            _, _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
            _, _, joints = smpl_model(poses, betas)
            mesh_infos[out_name] = {
                'Rh': Rh,
                'Th': Th,
                'poses': poses,
                'beats': betas,
                'joints': joints,
                'tpose_joints': tpose_joints
            }

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)
        
    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
        pickle.dump(
            {
                'joints': template_joints,
            }, f)



if __name__ == '__main__':
    app.run(main)
