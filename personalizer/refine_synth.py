"""Phase 3: refine synthetic frames with ELITE's DIFIX 2D generative prior.

Borrows ELITE's pretrained DIFIX model (``hufix/src/model.py`` + ``2d_prior.pth``)
in-process to clean up the Phase-2 renders, conditioned on a real reference frame.
Reads ``<exp>/synth/images/*.png`` -> writes ``<exp>/synth/images_refined/*.png``
(same filenames, so the stage-2 dataset can map them 1:1).

DIFIX is SD-turbo based and independent of the gaussian rasterizer, so it runs in
the GUAVA env as long as diffusers/transformers/peft are installed.

Example:
    PYTHONPATH=. python personalize/refine_synth.py \
        --exp_dir outputs/personalize/<id> \
        --elite_root /workspace1/pdawson/ELITE
"""
import os
import sys
import glob
import argparse
import torch
from PIL import Image
from torchvision import transforms


def load_difix(elite_root, ckpt):
    # model.py does `from hufix.src.mv_unet import ...`, so both ELITE root and
    # hufix/src must be importable.
    sys.path.insert(0, os.path.join(elite_root, 'hufix', 'src'))
    sys.path.insert(0, elite_root)
    from model import Difix
    model = Difix(pretrained_path=ckpt, timestep=199, mv_unet=True)
    model.set_eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth_dir', required=True, help='dir written by infer_synth.py (has images/, ref.png)')
    ap.add_argument('--elite_root', default='/workspace1/pdawson/ELITE')
    ap.add_argument('--ckpt', default=None, help='defaults to <elite_root>/checkpoints/2d_prior.pth')
    ap.add_argument('--batch_size', type=int, default=4)
    args = ap.parse_args()

    ckpt = args.ckpt or os.path.join(args.elite_root, 'checkpoints', '2d_prior.pth')
    assert os.path.exists(ckpt), f'DIFIX checkpoint not found: {ckpt}'

    synth = args.synth_dir
    in_dir = os.path.join(synth, 'images')
    out_dir = os.path.join(synth, 'images_refined')
    os.makedirs(out_dir, exist_ok=True)
    ref_path = os.path.join(synth, 'ref.png')
    assert os.path.exists(ref_path), f'reference image not found: {ref_path}'

    in_files = sorted(glob.glob(os.path.join(in_dir, '*.png')))
    assert in_files, f'no synthetic images in {in_dir} (run infer_synth.py first)'
    print(f'[refine] {len(in_files)} images, ref={ref_path}, ckpt={ckpt}')

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    images = torch.stack([to_tensor(Image.open(f).convert('RGB')) for f in in_files])  # [B,C,H,W]
    ref = torch.stack([to_tensor(Image.open(ref_path).convert('RGB'))])                # [1,C,H,W]

    model = load_difix(args.elite_root, ckpt)
    outputs = model.sample_batch_multi_tensor(
        image=images, ref_image=ref, batch_size=args.batch_size)  # [B,C,H,W]

    for f, out in zip(in_files, outputs):
        to_pil(out.cpu().clamp(0, 1)).save(os.path.join(out_dir, os.path.basename(f)))
    print(f'[refine] wrote {len(in_files)} refined images to {out_dir}')


if __name__ == '__main__':
    main()
