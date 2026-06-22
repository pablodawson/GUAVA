"""Per-identity test-time finetuning of the GUAVA inferer.

Loads the pretrained generalizable GUAVA model, freezes the DINO encoder, and
finetunes the gaussian decoders (+ render refiner) on the frames of a single
tracked video. Reuses GUAVA's own forward / loss / renderer so the optimization
target matches generalizable training exactly. Mirrors ELITE's
``FinetuneMeshUNetPriorModel2DGS`` but on GUAVA's 3D-gaussian avatar.
"""
import os
import copy
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.UbodyAvatar import (
    Ubody_Gaussian_inferer, Ubody_Gaussian, GaussianRenderer, configure_optimizers,
)
from utils.loss_utils import Optimization_Loss, cal_psnr, cal_ssim
from utils.general_utils import find_pt_file


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [to_device(v, device) for v in data]
    return data


class Personalizer:
    def __init__(self, meta_cfg, base_model, device, exp_dir,
                 lr=1e-5, freeze_encoder=True, bg=0.0, n_iters=3000):
        self.meta_cfg = meta_cfg
        self.device = device
        self.exp_dir = exp_dir
        self.bg = bg
        os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
        self.vis_dir = os.path.join(exp_dir, 'vis_results')
        os.makedirs(self.vis_dir, exist_ok=True)

        self.infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL).to(device)
        self.render_model = GaussianRenderer(meta_cfg.MODEL).to(device)

        if base_model is None or os.path.isdir(base_model):
            ckpt_dir = os.path.join(base_model or meta_cfg.MODEL.flame_assets_dir, 'checkpoints')
            base_model = find_pt_file(ckpt_dir, 'best') or find_pt_file(ckpt_dir, 'latest')
        assert base_model and os.path.exists(base_model), f'base model not found: {base_model}'
        state = torch.load(base_model, map_location='cpu', weights_only=True)
        self.infer_model.load_state_dict(state['model'], strict=False)
        self.render_model.load_state_dict(state['render_model'], strict=False)
        print(f'[personalize] loaded base model: {base_model}')

        if freeze_encoder:
            for p in self.infer_model.dino_encoder.parameters():
                p.requires_grad = False

        # reuse GUAVA's param grouping; replace scheduler with finetune-appropriate decay
        from omegaconf import OmegaConf
        opt_cfg = copy.deepcopy(meta_cfg.OPTIMIZE)
        OmegaConf.set_readonly(opt_cfg, False)
        opt_cfg.learning_rate = lr
        self.optimizer, _ = configure_optimizers(self.infer_model, opt_cfg, self.render_model)
        # LinearLR: warmup for 5% of iters, then cosine decay to 10% of peak LR
        warmup_iters = max(1, n_iters // 20)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=n_iters - warmup_iters, eta_min=lr * 0.1),
            ],
            milestones=[warmup_iters],
        )

        self.loss_model = Optimization_Loss(meta_cfg).to(device)
        self.best_metric = None

    def _forward(self, batch, iter_idx):
        vertex_gs_dict, uv_point_gs_dict, _ = self.infer_model(batch['source'])
        ubody = Ubody_Gaussian(self.meta_cfg.MODEL, vertex_gs_dict, uv_point_gs_dict, pruning=False)
        ubody.init_ehm(self.infer_model.ehm)
        deform = ubody(batch['target'])
        render = self.render_model(deform, batch['target']['render_cam_params'], bg=self.bg)
        # Pre-mask renders so background doesn't pollute the loss regardless of iter_idx.
        # (loss_utils only masks renders for iter<1000; finetuning needs it always.)
        mask = batch['target']['mask']
        render = dict(render)
        render['renders'] = render['renders'] * mask + (1 - mask) * self.bg
        if 'raw_renders' in render:
            render['raw_renders'] = render['raw_renders'] * mask + (1 - mask) * self.bg
        extra = {
            'uv_point_xyz': uv_point_gs_dict['local_pos'], 'uv_point_scale': uv_point_gs_dict['scales'],
            'vertices': self.infer_model.smplx_deform_res['vertices'],
            'uv_point_opacity': uv_point_gs_dict['opacities'],
            'vertex_opacity': vertex_gs_dict['opacities'], 'vertex_scale': vertex_gs_dict['scales'],
        }
        loss_dict, show = self.loss_model(render, batch['target'], extra, iter_idx)
        return render, loss_dict, show

    def fit(self, train_dataset, val_dataset, n_iters=3000, batch_size=2,
            val_interval=200, log_interval=20):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, drop_last=True)
        self.attach_val(val_dataset)

        # Clear stale vis from previous runs so filenames always correspond to this run
        import shutil
        if os.path.isdir(self.vis_dir):
            shutil.rmtree(self.vis_dir)
        os.makedirs(self.vis_dir)

        # Baseline eval before any weight updates
        self.validate(0)

        self._set_train(True)
        train_iter = iter(train_loader)
        bar = tqdm(range(1, n_iters + 1), desc='personalize')
        for it in bar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = to_device(batch, self.device)

            _, loss_dict, show = self._forward(batch, it)
            loss = sum(loss_dict.values())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for p in self.infer_model.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_()
            self.optimizer.step()
            self.scheduler.step()

            if it % log_interval == 0:
                bar.set_postfix({'loss': f'{loss.detach().item():.4f}', **show})
            if it % val_interval == 0 or it == n_iters:
                self.validate(it)
                self._set_train(True)
        self.save('latest.pt', n_iters)

    @torch.no_grad()
    def validate(self, it):
        self._set_train(False)
        psnrs, ssims, vis = [], [], []
        for vidx, batch in enumerate(self.val_iter_cache):
            batch = to_device(batch, self.device)
            render, _, _ = self._forward(batch, it)
            pred = render['renders'].clamp(0, 1)
            gt_mask = batch['target']['mask']
            gt = (batch['target']['image'] * gt_mask + (1 - gt_mask) * self.bg).clamp(0, 1)
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

    def attach_val(self, val_dataset):
        # cache the (small) held-out set once to avoid re-collating each val
        self.val_iter_cache = list(DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0))

    def save(self, name, it, prune_best=False):
        ckpt_dir = os.path.join(self.exp_dir, 'checkpoints')
        if prune_best:
            for f in os.listdir(ckpt_dir):
                if f.startswith('best'):
                    os.remove(os.path.join(ckpt_dir, f))
        torch.save({
            'model': self.infer_model.state_dict(),
            'render_model': self.render_model.state_dict(),
            'global_iter': it,
        }, os.path.join(ckpt_dir, name))

    def _set_train(self, train):
        self.infer_model.train(train)
        self.render_model.train(train)
