"""Phase 2: synthetic view generation for stage-2 personalization.

Builds the personalized avatar (from the stage-1 checkpoint) and renders it under
(a) random FLAME expressions at the captured camera, and (b) novel camera views
at captured expressions. The rendered RGBs are saved for the DIFIX refiner
(Phase 3); the exact target params used are saved alongside so stage-2 finetuning
(Phase 4) can re-render them and supervise against the refined images.

Mirrors ELITE's ``infer_randexpr.py`` but on GUAVA's full-body avatar, adding
novel-camera views (per the chosen design: random expressions + novel views).

Example:
    PYTHONPATH=. python personalize/infer_synth.py \
        --exp_dir outputs/personalize/<id> \
        --data_path outputs/personalize/tracked/<id> \
        --num_expr 64 --num_views 64
"""
import os
import json
import pickle
import argparse
import torch
import torchvision
import lightning

from omegaconf import OmegaConf
from utils.general_utils import ConfigDict, add_extra_cfgs, device_parser, find_pt_file
from utils.camera_utils import generate_novel_view_poses
from models.UbodyAvatar import Ubody_Gaussian_inferer, Ubody_Gaussian, GaussianRenderer
from dataset import TrackedData_infer


def to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {k: to_cpu(v) for k, v in data.items()}
    if isinstance(data, list):
        return [to_cpu(v) for v in data]
    return data


def set_data_path(meta_cfg, data_path):
    OmegaConf.set_readonly(meta_cfg._dot_config, False)
    meta_cfg._dot_config.DATASET.data_path = data_path
    OmegaConf.set_readonly(meta_cfg._dot_config, True)


def randomize_expression(target, expr_scale, max_jaw):
    fc = target['flame_coeffs']
    assert 'expression_params' in fc, 'flame_coeffs has no expression_params'
    exp = fc['expression_params']
    fc['expression_params'] = (torch.randn_like(exp) * expr_scale).clamp(-2.0, 2.0)
    jaw = fc.get('jaw_params', None)
    if jaw is not None and jaw.shape[-1] == 3 and max_jaw > 0:
        new_jaw = torch.zeros_like(jaw)
        new_jaw[..., 0] = torch.rand(jaw.shape[:-1], device=jaw.device) * max_jaw
        fc['jaw_params'] = new_jaw
    return target


def full_image_boxes(target, image_size):
    box = torch.tensor([[0, image_size - 1, 0, image_size - 1]],
                       dtype=torch.long, device=target['head_box'].device)
    for k in ('head_box', 'left_hand_box', 'right_hand_box'):
        target[k] = box.clone()
    return target


def save_record(out_dir, name, mode, img, target, records):
    img_path = os.path.join(out_dir, 'images', f'{name}.png')
    tgt_path = os.path.join(out_dir, 'targets', f'{name}.pkl')
    torchvision.utils.save_image(img, img_path)
    tgt = to_cpu(target)
    # GT for stage-2 is the (refined) render, not the base frame's image
    tgt.pop('image', None)
    tgt.pop('mask', None)
    with open(tgt_path, 'wb') as f:
        pickle.dump(tgt, f)
    records.append({'name': name, 'mode': mode,
                    'image': os.path.relpath(img_path, out_dir),
                    'target': os.path.relpath(tgt_path, out_dir)})


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True, help='stage-1 output (has config.yaml + checkpoints/)')
    ap.add_argument('--out_dir', default=None, help='where to write synth/ (default <model_dir>/synth)')
    ap.add_argument('--data_path', required=True, help='tracked video dir (pose/expr source)')
    ap.add_argument('--mode', choices=['expr', 'views', 'both'], default='both',
                    help='expr = random expressions only; views = novel cameras only; both = all')
    ap.add_argument('--num_expr', type=int, default=64)
    ap.add_argument('--num_views', type=int, default=64)
    ap.add_argument('--expr_scale', type=float, default=1.0)
    ap.add_argument('--max_jaw', type=float, default=0.2)
    ap.add_argument('--bg', type=float, default=0.0, help='match finetune bg')
    ap.add_argument('--devices', '-d', default='0')
    args = ap.parse_args()

    cfg_path = os.path.join(args.model_dir, 'config.yaml')
    meta_cfg = add_extra_cfgs(ConfigDict(model_config_path=cfg_path))
    lightning.fabric.seed_everything(10)
    torch.set_float32_matmul_precision('high')
    device = f'cuda:{device_parser(args.devices)[0]}'

    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL).to(device).eval()
    render_model = GaussianRenderer(meta_cfg.MODEL).to(device).eval()
    ckpt_dir = os.path.join(args.model_dir, 'checkpoints')
    ckpt = find_pt_file(ckpt_dir, 'best') or find_pt_file(ckpt_dir, 'latest')
    assert ckpt, f'no personalized checkpoint in {ckpt_dir}'
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(state['model'], strict=False)
    render_model.load_state_dict(state['render_model'], strict=False)
    print(f'[infer_synth] loaded {ckpt}')

    set_data_path(meta_cfg, args.data_path)
    ds = TrackedData_infer(cfg=meta_cfg, split='test', device=device, test_full=True)
    video_id = list(ds.videos_info.keys())[0]
    frames = ds.videos_info[video_id]['frames_keys']

    # build the avatar once from a source frame
    source_info = ds._load_source_info(video_id)
    vertex_gs, uv_gs, _ = infer_model(source_info)
    ubody = Ubody_Gaussian(meta_cfg.MODEL, vertex_gs, uv_gs, pruning=True)
    ubody.init_ehm(infer_model.ehm)
    ubody.eval()

    out_dir = args.out_dir or os.path.join(args.model_dir, 'synth')
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'targets'), exist_ok=True)
    # reference image for DIFIX = first real frame
    ref = ds._load_target_info(video_id, frames[0])['image'][0].clamp(0, 1)
    torchvision.utils.save_image(ref, os.path.join(out_dir, 'ref.png'))

    records = []
    # (a) random expressions at the captured camera
    if args.mode in ('expr', 'both'):
        for i in range(args.num_expr):
            target = ds._load_target_info(video_id, frames[i % len(frames)])
            target = randomize_expression(target, args.expr_scale, args.max_jaw)
            deform = ubody(target)
            render = render_model(deform, target['render_cam_params'], bg=args.bg)
            save_record(out_dir, f'expr_{i:05d}', 'expr',
                        render['renders'][0].clamp(0, 1), target, records)

    # (b) novel camera views at captured expressions (boxes invalid -> full image)
    if args.mode in ('views', 'both'):
        base = ds._load_target_info(video_id, frames[0])
        novel_cams = generate_novel_view_poses(
            base, image_size=ds.image_size, tanfov=ds.tanfov, num_keyframes=args.num_views)
        for i in range(args.num_views):
            target = ds._load_target_info(video_id, frames[i % len(frames)])
            target['render_cam_params'] = novel_cams[i]
            target = full_image_boxes(target, ds.image_size)
            deform = ubody(target)
            render = render_model(deform, target['render_cam_params'], bg=args.bg)
            save_record(out_dir, f'view_{i:05d}', 'view',
                        render['renders'][0].clamp(0, 1), target, records)

    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump({'bg': args.bg, 'records': records}, f, indent=2)
    print(f'[infer_synth] wrote {len(records)} synthetic frames to {out_dir}')
    ds._lmdb_engine.close()


if __name__ == '__main__':
    main()
