"""Optimization-only per-identity avatar (no feed-forward inferer).

Instead of finetuning GUAVA's image-conditioned inferer, this builds an explicit
set of learnable gaussians bound to the canonical EHM mesh (SplattingAvatar-style
barycentric surface binding) and optimizes them directly against every frame of a
tracked video. It reuses GUAVA's own EHM deformation, gaussian renderer and loss,
so the optimization target matches the generalizable model.

Two gaussian sets, identical to GUAVA's representation:
  * vertex gaussians  -- one per SMPL-X vertex, deformed by per-vertex LBS transform.
  * uv-point gaussians -- bound to a face (binding_face) with barycentric coords
    (face_bary) + a local offset, deformed by per-face orientation/scale.

Example:
    PYTHONPATH=. python personalizer/optimize.py \
        --data_path outputs/personalize/tracked/<video_stem> \
        --base_model assets/GUAVA --exp_dir outputs/personalize/<id>/opt
"""
import os
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import lightning
from tqdm import tqdm
from torch.utils.data import DataLoader
from roma import (rotmat_to_unitquat, quat_product,
                  quat_xyzw_to_wxyz, quat_wxyz_to_xyzw)

from models.UbodyAvatar import GaussianRenderer
from models.modules.ehm import EHM
from utils.graphics_utils import compute_face_orientation
from utils.general_utils import (ConfigDict, add_extra_cfgs, device_parser,
                                 find_pt_file, inverse_sigmoid)
from utils.loss_utils import Optimization_Loss, cal_psnr, cal_ssim
from personalizer.sv_dataset import _SVBase


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [to_device(v, device) for v in data]
    return data


class FrameDataset(_SVBase):
    """All frames of the tracked video as bare target records (no source image:
    the optimizer has no inferer to feed)."""

    def __init__(self, cfg, data_path, eval_last_n_frames=32, split='train'):
        super().__init__(cfg, data_path, eval_last_n_frames)
        self.keys = self.eval_keys if split == 'eval' else self.train_keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return self._build_target(self.keys[index])


class ExplicitAvatar(nn.Module):
    """Explicit, optimizable gaussians bound to the canonical EHM mesh."""

    def __init__(self, cfg, color_dim, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.ehm = EHM(cfg.flame_assets_dir, cfg.smplx_assets_dir,
                       add_teeth=cfg.add_teeth, uv_size=cfg.uvmap_size).to(device)
        self.smplx = self.ehm.smplx
        self.faces = self.smplx.faces_tensor.to(device)

        # --- canonical surface binding (GUAVA's UV-grid layout) ---
        uv_mask = self.smplx.uvmap_mask.flatten()
        binding_face = self.smplx.uvmap_f_idx.reshape(uv_mask.shape[0], -1)[uv_mask].squeeze(-1).long()
        face_bary = self.smplx.uvmap_f_bary.reshape(uv_mask.shape[0], -1)[uv_mask].float()
        self.register_buffer('binding_face', binding_face.to(device))
        self.register_buffer('face_bary', face_bary.to(device))

        v_template = self.ehm.v_template.to(device)               # (V, 3)
        n_vert = v_template.shape[0]
        n_uv = binding_face.shape[0]

        # canonical positions of the uv points (on the template surface)
        face_v = v_template[self.faces]                           # (F, 3, 3)
        uv_canon = torch.einsum('nk,nkj->nj', face_bary, face_v[binding_face])

        # scale init from mean mesh edge length (SplattingAvatar uses knn distance;
        # edge length is the cheap, local analog on a fixed-topology mesh).
        edges = torch.cat([face_v[:, 0] - face_v[:, 1],
                           face_v[:, 1] - face_v[:, 2],
                           face_v[:, 2] - face_v[:, 0]], dim=0)
        mean_edge = edges.norm(dim=-1).mean()

        def log_scale(val, n):
            return torch.full((n, 3), float(torch.log(val)), device=device)

        # vertex gaussians (scaling used directly in world units)
        self.vert_offset = nn.Parameter(torch.zeros(n_vert, 3, device=device))
        self.vert_scaling = nn.Parameter(log_scale(mean_edge, n_vert))
        self.vert_rotation = nn.Parameter(self._init_quat(n_vert))
        self.vert_opacity = nn.Parameter(self._init_opacity(n_vert))
        self.vert_color = nn.Parameter(torch.zeros(n_vert, color_dim, device=device))

        # uv-point gaussians (scaling multiplied by per-face scale at deform time;
        # canonical face_scaling == 1, so the stored value is the world scale)
        self.uv_local_pos = nn.Parameter(torch.zeros(n_uv, 3, device=device))
        self.uv_scaling = nn.Parameter(log_scale(mean_edge * 0.5, n_uv))
        self.uv_rotation = nn.Parameter(self._init_quat(n_uv))
        self.uv_opacity = nn.Parameter(self._init_opacity(n_uv))
        self.uv_color = nn.Parameter(torch.zeros(n_uv, color_dim, device=device))
        self.register_buffer('uv_canon', uv_canon)
        self.to(device)  # the quat/opacity inits are built on CPU; unify devices
        print(f'[optimize] gaussians: {n_vert} vertex + {n_uv} uv-point = {n_vert + n_uv}')

    @staticmethod
    def _init_quat(n):
        q = torch.zeros(n, 4)
        q[:, 0] = 1.0
        return q

    @staticmethod
    def _init_opacity(n):
        return inverse_sigmoid(0.1 * torch.ones(n, 1))

    @staticmethod
    def _activate_color(c):
        # match Ubody_Gaussian: first 3 channels are sigmoid RGB, rest are raw features.
        return torch.cat([torch.sigmoid(c[..., :3]), c[..., 3:]], dim=-1)

    def forward(self, batch):
        B = batch['smplx_coeffs']['shape'].shape[0]
        offset = self.vert_offset[None].expand(B, -1, -1)
        deform = self.ehm(batch['smplx_coeffs'], batch['flame_coeffs'], static_offset=offset)
        verts = deform['vertices']                                # (B, V, 3)

        # vertex gaussians
        v_rot_xyzw = rotmat_to_unitquat(deform['ver_transform_mat'][:, :, :3, :3])
        base_v_rot = torch.nn.functional.normalize(self.vert_rotation, dim=-1)[None].expand(B, -1, -1)
        vert_rot = torch.nn.functional.normalize(
            quat_xyzw_to_wxyz(quat_product(v_rot_xyzw, quat_wxyz_to_xyzw(base_v_rot))), dim=-1)
        vert_scale = torch.exp(self.vert_scaling)[None].expand(B, -1, -1)

        # uv-point gaussians (bound to faces)
        face_orien_mat, face_scaling = compute_face_orientation(verts, self.faces, return_scale=True)
        face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(face_orien_mat))
        face_v = verts[:, self.faces]                             # (B, F, 3, 3)
        face_v_nn = face_v[:, self.binding_face]                  # (B, N, 3, 3)
        bary = self.face_bary[None].expand(B, -1, -1)
        center_nn = torch.einsum('bnk,bnkj->bnj', bary, face_v_nn)
        scale_nn = face_scaling[:, self.binding_face]             # (B, N, 1)

        local = self.uv_local_pos[None].expand(B, -1, -1)
        orien_nn = face_orien_mat[:, self.binding_face]
        uv_xyz = torch.einsum('bnij,bnj->bni', orien_nn, local) * scale_nn + center_nn
        base_uv_rot = torch.nn.functional.normalize(self.uv_rotation, dim=-1)[None].expand(B, -1, -1)
        uv_rot = quat_xyzw_to_wxyz(quat_product(
            quat_wxyz_to_xyzw(face_orien_quat[:, self.binding_face]),
            quat_wxyz_to_xyzw(base_uv_rot)))
        uv_scale = torch.exp(self.uv_scaling)[None].expand(B, -1, -1) * scale_nn

        vert_color = self._activate_color(self.vert_color)[None].expand(B, -1, -1)
        uv_color = self._activate_color(self.uv_color)[None].expand(B, -1, -1)

        assets = {
            'xyz': torch.cat([verts, uv_xyz], dim=1),
            'rotation': torch.cat([vert_rot, uv_rot], dim=1),
            'scaling': torch.cat([vert_scale, uv_scale], dim=1),
            'opacity': torch.cat([torch.sigmoid(self.vert_opacity)[None].expand(B, -1, -1),
                                  torch.sigmoid(self.uv_opacity)[None].expand(B, -1, -1)], dim=1),
            'features_color': torch.cat([vert_color, uv_color], dim=1),
            'sh_degree': 0,
        }
        # regularizers consumed by Optimization_Loss
        extra = {'uv_point_xyz': local, 'uv_point_scale': torch.exp(self.uv_scaling)[None].expand(B, -1, -1)}
        return assets, extra

    def save_pointcloud(self, path):
        import open3d as o3d
        xyz = torch.cat([self.ehm.v_template.to(self.device) + self.vert_offset, self.uv_canon], dim=0)
        rgb = torch.cat([torch.sigmoid(self.vert_color[:, :3]), torch.sigmoid(self.uv_color[:, :3])], dim=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())
        o3d.io.write_point_cloud(path, pcd)


class Optimizer:
    def __init__(self, cfg, base_model, device, exp_dir, train_refiner=False, bg=0.0):
        self.cfg = cfg
        self.device = device
        self.exp_dir = exp_dir
        self.bg = bg
        os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
        self.vis_dir = os.path.join(exp_dir, 'vis_results')

        self.avatar = ExplicitAvatar(cfg.MODEL, cfg.MODEL.color_dim, device)
        self.render_model = GaussianRenderer(cfg.MODEL).to(device)

        if base_model is None or os.path.isdir(base_model):
            ckpt_dir = os.path.join(base_model or cfg.MODEL.flame_assets_dir, 'checkpoints')
            base_model = find_pt_file(ckpt_dir, 'best') or find_pt_file(ckpt_dir, 'latest')
        assert base_model and os.path.exists(base_model), f'base model not found: {base_model}'
        state = torch.load(base_model, map_location='cpu', weights_only=True)
        self.render_model.load_state_dict(state['render_model'], strict=False)
        print(f'[optimize] loaded refiner from base model: {base_model}')

        if not train_refiner:
            for p in self.render_model.parameters():
                p.requires_grad = False

        groups = [
            {'params': [self.avatar.vert_offset, self.avatar.uv_local_pos], 'lr': 1e-4},
            {'params': [self.avatar.vert_scaling, self.avatar.uv_scaling], 'lr': 5e-3},
            {'params': [self.avatar.vert_rotation, self.avatar.uv_rotation], 'lr': 1e-3},
            {'params': [self.avatar.vert_opacity, self.avatar.uv_opacity], 'lr': 5e-2},
            {'params': [self.avatar.vert_color, self.avatar.uv_color], 'lr': 1e-2},
        ]
        if train_refiner:
            groups.append({'params': list(self.render_model.parameters()), 'lr': 1e-4})
        self.optimizer = torch.optim.Adam(groups, betas=(0.9, 0.99))
        self.loss_model = Optimization_Loss(cfg).to(device)
        self.best_metric = None

    def _forward(self, batch, it):
        assets, extra = self.avatar(batch)
        render = self.render_model(assets, batch['render_cam_params'], bg=self.bg)
        mask = batch['mask']
        render = dict(render)
        render['renders'] = render['renders'] * mask + (1 - mask) * self.bg
        if 'raw_renders' in render:
            render['raw_renders'] = render['raw_renders'] * mask + (1 - mask) * self.bg
        loss_dict, show = self.loss_model(render, batch, extra, it)
        return render, loss_dict, show

    def fit(self, train_ds, val_ds, n_iters, batch_size, val_interval, log_interval=20):
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)
        self.val_cache = list(DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0))
        if os.path.isdir(self.vis_dir):
            shutil.rmtree(self.vis_dir)
        os.makedirs(self.vis_dir)

        self.validate(0)
        train_iter = iter(loader)
        bar = tqdm(range(1, n_iters + 1), desc='optimize')
        for it in bar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)
            batch = to_device(batch, self.device)

            _, loss_dict, show = self._forward(batch, it)
            loss = sum(loss_dict.values())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if it % log_interval == 0:
                bar.set_postfix({'loss': f'{loss.detach().item():.4f}', **show})
            if it % val_interval == 0 or it == n_iters:
                self.validate(it)
        self.save('latest.pt', n_iters)
        self.avatar.save_pointcloud(os.path.join(self.exp_dir, 'canonical.ply'))

    @torch.no_grad()
    def validate(self, it):
        psnrs, ssims, vis = [], [], []
        for vidx, batch in enumerate(self.val_cache):
            batch = to_device(batch, self.device)
            render, _, _ = self._forward(batch, it)
            pred = render['renders'].clamp(0, 1)
            m = batch['mask']
            gt = (batch['image'] * m + (1 - m) * self.bg).clamp(0, 1)
            psnrs.append(float(cal_psnr(pred, gt).mean()))
            ssims.append(float(cal_ssim(pred, gt).mean()))
            if vidx < 4:
                vis.append(torch.cat([gt[0], pred[0]], dim=2).cpu())
        mpsnr, mssim = float(np.mean(psnrs)), float(np.mean(ssims))
        print(f'[val @ {it}] PSNR={mpsnr:.3f} SSIM={mssim:.4f}')
        if vis:
            grid = torchvision.utils.make_grid(vis, nrow=1, padding=0).clamp(0, 1)
            torchvision.utils.save_image(grid, os.path.join(self.vis_dir, f'val_{it:06d}.png'))
        if self.best_metric is None or mssim >= self.best_metric:
            self.best_metric = mssim
            self.save(f'best_{it}_{mssim:.3f}.pt', it, prune_best=True)

    def save(self, name, it, prune_best=False):
        ckpt_dir = os.path.join(self.exp_dir, 'checkpoints')
        if prune_best:
            for f in os.listdir(ckpt_dir):
                if f.startswith('best'):
                    os.remove(os.path.join(ckpt_dir, f))
        torch.save({
            'avatar': self.avatar.state_dict(),
            'render_model': self.render_model.state_dict(),
            'global_iter': it,
        }, os.path.join(ckpt_dir, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True, help='tracked video dir (EHM-Tracker output)')
    ap.add_argument('--exp_dir', required=True)
    ap.add_argument('--base_model', default='assets/GUAVA',
                    help='dir with config.yaml + checkpoints/ (source of the frozen refiner)')
    ap.add_argument('--config', default=None, help='defaults to <base_model>/config.yaml')
    ap.add_argument('--devices', '-d', default='0')
    ap.add_argument('--n_iters', type=int, default=8000)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--eval_last_n_frames', type=int, default=32)
    ap.add_argument('--val_interval', type=int, default=500)
    ap.add_argument('--train_refiner', action='store_true',
                    help='also finetune the neural render refiner (frozen by default)')
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(args.base_model, 'config.yaml')
    meta_cfg = add_extra_cfgs(ConfigDict(model_config_path=cfg_path))
    lightning.fabric.seed_everything(10)
    torch.set_float32_matmul_precision('high')
    device = f'cuda:{device_parser(args.devices)[0]}'

    os.makedirs(args.exp_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(args.exp_dir, 'config.yaml'))

    train_ds = FrameDataset(meta_cfg, args.data_path, args.eval_last_n_frames, split='train')
    val_ds = FrameDataset(meta_cfg, args.data_path, args.eval_last_n_frames, split='eval')
    print(f'[optimize] train={len(train_ds)} eval={len(val_ds)}')

    opt = Optimizer(meta_cfg, args.base_model, device, args.exp_dir, train_refiner=args.train_refiner)
    opt.fit(train_ds, val_ds, args.n_iters, args.batch_size, args.val_interval)
    print(f'[optimize] done. checkpoints in {args.exp_dir}/checkpoints')


if __name__ == '__main__':
    main()
