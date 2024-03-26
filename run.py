from curses import flushinp
import imp
import os

import torch
import numpy as np
from tqdm import tqdm

from configs import cfg, args


from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image

from third_parties import lpips
from third_parties.lpips import LPIPS
import skimage
from core.nets import setup_distributed

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'cam_name', 'img_name',
                       'img_width', 'img_height', 'ray_mask']


def load_network():
    if cfg.ddp:
        setup_distributed()
    model = create_network()
    load_dir = os.path.join('experiments', cfg.category, cfg.task, cfg.subject, cfg.experiment)
    ckpt_path = os.path.join(load_dir, f'latest.tar')
    print(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    return model.cuda()

def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None):
    
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image  = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image


def _freeview(
        data_type='freeview',
        folder_name=None):
    cfg.perturb = 0.

    model = load_network()
    test_loader = create_dataloader(data_type)
    writer = ImageWriter(
                output_dir=os.path.join(cfg.logdir, cfg.load_net),
                exp_name=folder_name)

    model.eval()
    for batch in tqdm(test_loader):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU)

        with torch.no_grad():
            net_output = model(**data, 
                               iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']
        target_rgbs = batch.get('target_rgbs', None)

        rgb_img, alpha_img, _ = unpack_to_image(
            width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
            rgb.data.cpu().numpy(),
            alpha.data.cpu().numpy())

        imgs = [rgb_img]
        if cfg.show_truth and target_rgbs is not None:
            target_rgbs = to_8b_image(target_rgbs.numpy())
            imgs.append(target_rgbs)
        if cfg.show_alpha:
            imgs.append(alpha_img)

        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out)

    writer.finalize()


def run_freeview():
    _freeview(
        data_type='freeview',
        folder_name=f"freeview_{cfg.freeview.frame_idx}" \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_tpose():
    cfg.ignore_non_rigid_motions = True
    _freeview(
        data_type='tpose',
        folder_name='tpose' \
            if not cfg.render_folder_name else cfg.render_folder_name)

def run_text():
    _freeview(
        data_type='text',
        folder_name='text' \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_test(render_folder_name='movement'):
    cfg.perturb = 0.
    cfg.views = ['01']  # hardcoded
    cfg.skip = 1

    seq_dict = {
        '0': 'gBR_sBM_cAll_d04_mBR1_ch05',
        '1': 'gBR_sBM_cAll_d04_mBR1_ch06',
        '2': 'MPI_Limits_03099_op8_poses'
    }

    name_dict = {
        '0': 'dance0',
        '1': 'dance1',
        '2': 'flipping'
    }

    cfg.seqname = seq_dict[args.seqid]

    model = load_network()
    test_loader = create_dataloader('test')
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net),
        exp_name=f'test-{name_dict[args.seqid]}')

    model.eval()
    for idx, batch in enumerate(tqdm(test_loader)):
        img_name = batch['img_name'][0]

        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
            batch,
            exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        with torch.no_grad():
            # print('iter--val', cfg.eval_iter, flush=True)
            net_output = model(**data, iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        rgb_img, alpha_img, _ = \
            unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy())

        imgs = [rgb_img]
        # if cfg.show_truth:
        #     imgs.append(truth_img)
        # if cfg.show_alpha:
        #     imgs.append(alpha_img)

        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=img_name)

    writer.finalize()

def run_movement(render_folder_name='movement'):
    cfg.perturb = 0.
    cfg.views = args.views
    cfg.skip = 1

    model = load_network()
    test_loader = create_dataloader('movement')
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net),
        exp_name='video')

    model.eval()
    for idx, batch in enumerate(tqdm(test_loader)):
        img_name = batch['img_name'][0]

        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        with torch.no_grad():
            #print('iter--val', cfg.eval_iter, flush=True)
            net_output = model(**data, iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        rgb_img, alpha_img, truth_img = \
            unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor)/255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy(),
                batch['target_rgbs'])

        imgs = [rgb_img]
        if cfg.show_truth:
            imgs.append(truth_img)
        if cfg.show_alpha:
            imgs.append(alpha_img)
            
        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=img_name)
    
    writer.finalize()


def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt)**2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr

def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


def get_loss(rgb, target):
    lpips = LPIPS(net='vgg')#.cuda()
    lpips_loss = lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                       scale_for_lpips(target.permute(0, 3, 1, 2)))
    return torch.mean(lpips_loss).cpu().detach().numpy()

def run_eval(render_folder_name='movement'):
    cfg.perturb = 0.
    # cfg.test_view = args.test_view
    cfg.views = [f'{cam_name:02d}' for cam_name in range(2, 24)]
    cfg.skip = 30

    test_loader = create_dataloader('movement')
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, 'latest'),#cfg.load_net
        exp_name=render_folder_name)

    model = load_network()

    model.eval()
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    psnr_all = []
    ssim_all = []
    lpips_all = []
    time_all = []
    for idx, batch in enumerate(tqdm(test_loader)):
        # evaluate at an interval of 30
        img_name = batch['img_name'][0]
        # if int(batch['frame_name']) % 30 != 0:
        #     continue

        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        iter_start.record()

        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter)

        iter_end.record()
        torch.cuda.synchronize()
        elapsed = iter_start.elapsed_time(iter_end)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        pred_img, alpha_img, gt_img = \
            unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor)/255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy(),
                batch['target_rgbs'])

        imgs = [pred_img]
        if cfg.show_truth:
            imgs.append(gt_img)
        if cfg.show_alpha:
            imgs.append(alpha_img)
            
        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=img_name)

        pred_img_norm = pred_img / 255.
        gt_img_norm = gt_img / 255.

        # get evaluation
        psnr_all.append(psnr_metric(pred_img_norm, gt_img_norm))
        ssim_all.append(skimage.metrics.structural_similarity(pred_img_norm, gt_img_norm, multichannel=True))
        lpips_loss = get_loss(rgb=torch.from_numpy(pred_img_norm).float().unsqueeze(0), target=torch.from_numpy(gt_img_norm).float().unsqueeze(0))
        lpips_all.append(lpips_loss)
        time_all.append(elapsed)

    print('psnr: ', np.array(psnr_all).mean())
    print('lpips: ', np.array(lpips_all).mean())
    print('ssim: ', np.array(ssim_all).mean())
    print('time:', np.array(time_all)[1:].mean())

    np.savez(os.path.join(cfg.logdir, 'latest', 'results.npz'),
             psnr=np.array(psnr_all).mean(),
             ssim=np.array(ssim_all).mean(),
             lpips=np.array(lpips_all).mean(),
             time=np.array(time_all)[1:].mean()
             )
    writer.finalize()

if __name__ == '__main__':
    globals()[f'run_{args.type}']()
