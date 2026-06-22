"""Per-identity personalization entry point (mirrors ELITE's personalize.py).

Stage 1: finetune the GUAVA inferer on the real frames of a tracked video.
Stage 2 (Phase 4): joint finetune on real + DIFIX-refined synthetic frames.

Example:
    PYTHONPATH=. python personalize/personalize.py --stage 1 \
        --data_path outputs/app/tracked_driven_video/<id> \
        --base_model assets/GUAVA \
        --exp_dir outputs/personalize/<id>
"""
import os
import shutil
import argparse
import torch
import lightning

from utils.general_utils import ConfigDict, add_extra_cfgs, device_parser
from personalizer.finetune import Personalizer
from personalizer.sv_dataset import (
    SVVideoTrainDataset, SVVideoTestDataset, SVVideoStage2Dataset,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, default=1, choices=[1, 2])
    ap.add_argument('--data_path', required=True, help='tracked video dir (EHM-Tracker output)')
    ap.add_argument('--exp_dir', required=True)
    ap.add_argument('--base_model', default='assets/GUAVA',
                    help='dir with config.yaml + checkpoints/, or a .pt file')
    ap.add_argument('--config', default=None,
                    help='model config yaml; defaults to <base_model>/config.yaml')
    ap.add_argument('--devices', '-d', default='0')
    ap.add_argument('--n_iters', type=int, default=3000)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--num_real_frames', type=int, default=None,
                    help='evenly subsample this many train frames (None = all)')
    ap.add_argument('--eval_last_n_frames', type=int, default=32)
    ap.add_argument('--val_interval', type=int, default=200)
    ap.add_argument('--freeze_encoder', type=eval, default=True)
    # stage-2 only
    ap.add_argument('--num_synth_frames', type=int, default=64)
    ap.add_argument('--synth_dir', default=None,
                    help='stage-2: dir from infer_synth/refine_synth (has manifest.json)')
    args = ap.parse_args()

    if args.stage == 2 and not args.synth_dir:
        raise SystemExit('stage 2 requires --synth_dir (output of infer_synth/refine_synth)')

    cfg_path = args.config or os.path.join(args.base_model, 'config.yaml')
    meta_cfg = add_extra_cfgs(ConfigDict(model_config_path=cfg_path))
    lightning.fabric.seed_everything(10)
    torch.set_float32_matmul_precision('high')
    device = f'cuda:{device_parser(args.devices)[0]}'

    os.makedirs(args.exp_dir, exist_ok=True)
    # export reads <exp_dir>/config.yaml, so keep a copy alongside the checkpoints
    shutil.copy(cfg_path, os.path.join(args.exp_dir, 'config.yaml'))

    if args.stage == 1:
        train_ds = SVVideoTrainDataset(
            meta_cfg, args.data_path,
            num_frames=args.num_real_frames,
            eval_last_n_frames=args.eval_last_n_frames)
    else:
        train_ds = SVVideoStage2Dataset(
            meta_cfg, args.data_path, args.synth_dir,
            num_real_frames=args.num_real_frames,
            num_synth_frames=args.num_synth_frames,
            eval_last_n_frames=args.eval_last_n_frames)
    val_ds = SVVideoTestDataset(
        meta_cfg, args.data_path,
        eval_last_n_frames=args.eval_last_n_frames)
    print(f'[personalize] stage {args.stage}: train={len(train_ds)} eval={len(val_ds)}')

    trainer = Personalizer(
        meta_cfg, args.base_model, device, args.exp_dir,
        lr=args.lr, freeze_encoder=args.freeze_encoder, n_iters=args.n_iters)
    trainer.fit(train_ds, val_ds, n_iters=args.n_iters,
                batch_size=args.batch_size, val_interval=args.val_interval)
    print(f'[personalize] done. checkpoints in {args.exp_dir}/checkpoints')


if __name__ == '__main__':
    main()
