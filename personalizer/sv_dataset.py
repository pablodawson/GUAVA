"""Single-video datasets for per-identity personalization.

These read the tracked output produced by ``EHM-Tracker/tracking_video.py``
(the ``TrackedData_infer`` on-disk layout: a single video, frame-keyed
``optim_tracking_ehm.pkl`` + flat ``id_share_params.pkl`` + ``img_lmdb``) and
yield ``{'source': ..., 'target': ...}`` records compatible with GUAVA's
trainer forward pass. Mirrors ELITE's ``MeshUNetSVVideo*`` datasets.
"""
import os
import json
import pickle
import random
import numpy as np
import torch
import torchvision

from dataset.data_loader import TrackedData_infer
from utils.graphics_utils import get_full_proj_matrix


def _subsample_keys(keys, num):
    """Evenly pick ``num`` keys from ``keys`` (keep order)."""
    if num is None or num >= len(keys):
        return list(keys)
    idx = np.linspace(0, len(keys) - 1, num).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [keys[i] for i in idx]


def _squeeze_batch(obj):
    """Drop the leading size-1 (collate) dim from every tensor in a saved record,
    so a DataLoader can re-add it consistently with the real-frame records."""
    if torch.is_tensor(obj):
        return obj.squeeze(0) if obj.ndim >= 1 and obj.shape[0] == 1 else obj
    if isinstance(obj, dict):
        return {k: _squeeze_batch(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_squeeze_batch(v) for v in obj]
    return obj


class _SVBase(TrackedData_infer):
    """Shared loading: build a single {source,target} record (un-batched)."""

    def __init__(self, cfg, data_path, eval_last_n_frames, start_frame=0, device='cpu'):
        # TrackedData reads cfg.DATASET.data_path, which resolves through the
        # ConfigDict's internal read-only OmegaConf (_dot_config) -- so set it
        # there. (Mutating cfg['DATASET'] does NOT propagate; verified.)
        from omegaconf import OmegaConf
        OmegaConf.set_readonly(cfg._dot_config, False)
        cfg._dot_config.DATASET.data_path = data_path
        OmegaConf.set_readonly(cfg._dot_config, True)
        super().__init__(cfg, split='test', device=device, test_full=True)

        self.video_id = list(self.videos_info.keys())[0]
        all_keys = self.videos_info[self.video_id]['frames_keys']
        # detect mask channel count from a real frame so synth frames match
        self._init_lmdb_database()
        _, _, sample_mask = self._load_one_info(self.video_id, all_keys[0])
        self.mask_channels = sample_mask.shape[0]
        assert len(all_keys) > eval_last_n_frames, \
            f'video {self.video_id} has {len(all_keys)} frames <= eval_last_n_frames {eval_last_n_frames}'
        self.eval_last_n_frames = eval_last_n_frames
        self.train_keys = all_keys[start_frame: len(all_keys) - eval_last_n_frames]
        self.eval_keys = all_keys[len(all_keys) - eval_last_n_frames:]

    def _load_source(self, source_key):
        if not hasattr(self, '_lmdb_engine'):
            self._init_lmdb_database()
        s_info, s_img, s_mask = self._load_one_info(self.video_id, source_key)
        s_img = s_img * s_mask
        s_img = torchvision.transforms.functional.resize(
            s_img, (self.feature_img_size, self.feature_img_size), antialias=True)
        src = {'image': s_img}
        src.update(s_info)
        return src

    def _build_target(self, target_key):
        if not hasattr(self, '_lmdb_engine'):
            self._init_lmdb_database()
        t_info, t_img, t_mask = self._load_one_info(self.video_id, target_key)
        t_img = torchvision.transforms.functional.resize(
            t_img, (self.image_size, self.image_size), antialias=True)
        t_mask = torchvision.transforms.functional.resize(
            t_mask, (self.image_size, self.image_size), antialias=True)

        view_matrix, full_proj_matrix = get_full_proj_matrix(t_info['w2c_cam'], self.tanfov)
        t_info['render_cam_params'] = {
            "world_view_transform": view_matrix, "full_proj_transform": full_proj_matrix,
            'tanfovx': self.tanfov, 'tanfovy': self.tanfov,
            'image_height': self.image_size, 'image_width': self.image_size,
            'camera_center': t_info['c2w_cam'][:3, 3],
        }
        target = {'image': t_img, 'mask': t_mask}
        target.update(t_info)
        return target

    def _build_record(self, source_key, target_key):
        return {'source': self._load_source(source_key),
                'target': self._build_target(target_key)}


class SVVideoTrainDataset(_SVBase):
    """Real frames of the source video. Target = a train frame; source = a
    (different) train frame used for feature extraction."""

    def __init__(self, cfg, data_path, num_frames=None, eval_last_n_frames=32,
                 start_frame=0, source_mode='random'):
        super().__init__(cfg, data_path, eval_last_n_frames, start_frame)
        self.target_keys = _subsample_keys(self.train_keys, num_frames)
        self.source_mode = source_mode

    def __len__(self):
        return len(self.target_keys)

    def __getitem__(self, index):
        target_key = self.target_keys[index]
        if self.source_mode == 'fixed':
            source_key = self.train_keys[0]
        else:
            cand = [k for k in self.train_keys if k != target_key] or [target_key]
            source_key = random.choice(cand)
        return self._build_record(source_key, target_key)


class SVVideoTestDataset(_SVBase):
    """Held-out last-N frames for evaluation. Source = a fixed train frame."""

    def __init__(self, cfg, data_path, eval_last_n_frames=32, start_frame=0):
        super().__init__(cfg, data_path, eval_last_n_frames, start_frame)
        self.source_key = self.train_keys[0]

    def __len__(self):
        return len(self.eval_keys)

    def __getitem__(self, index):
        return self._build_record(self.source_key, self.eval_keys[index])


class SVVideoStage2Dataset(_SVBase):
    """Joint real + refined-synthetic frames (mirrors ELITE's RandExpr dataset).

    Real frames behave like :class:`SVVideoTrainDataset`. Synthetic frames load the
    target params saved by ``infer_synth.py`` and use the DIFIX-refined render (or
    the raw render if not refined) as the supervision image, with a full mask.
    """

    def __init__(self, cfg, data_path, synth_dir, num_real_frames=None,
                 num_synth_frames=None, eval_last_n_frames=32, start_frame=0,
                 use_refined=True):
        super().__init__(cfg, data_path, eval_last_n_frames, start_frame)
        self.real_keys = _subsample_keys(self.train_keys, num_real_frames)

        self.synth_dir = synth_dir
        self.refined_dir = os.path.join(synth_dir, 'images_refined')
        self.use_refined = use_refined
        with open(os.path.join(synth_dir, 'manifest.json'), 'r') as f:
            recs = json.load(f)['records']
        if num_synth_frames is not None and num_synth_frames < len(recs):
            idx = np.linspace(0, len(recs) - 1, num_synth_frames).round().astype(int)
            recs = [recs[i] for i in sorted(set(idx.tolist()))]
        self.synth = recs

    def __len__(self):
        return len(self.real_keys) + len(self.synth)

    def _build_synth(self, rec):
        with open(os.path.join(self.synth_dir, rec['target']), 'rb') as f:
            target = _squeeze_batch(pickle.load(f))

        img_path = os.path.join(self.refined_dir, os.path.basename(rec['image']))
        if not (self.use_refined and os.path.exists(img_path)):
            img_path = os.path.join(self.synth_dir, rec['image'])
        img = torchvision.io.read_image(img_path)[:3].float() / 255.0
        img = torchvision.transforms.functional.resize(
            img, (self.image_size, self.image_size), antialias=True)

        target['image'] = img
        target['mask'] = torch.ones(self.mask_channels, self.image_size, self.image_size)
        # coerce render cam scalars back to the real-frame types so a mixed batch
        # collates cleanly (real frames store python float/int here)
        rcp = target['render_cam_params']
        rcp['tanfovx'] = float(rcp['tanfovx'])
        rcp['tanfovy'] = float(rcp['tanfovy'])
        rcp['image_height'] = int(rcp['image_height'])
        rcp['image_width'] = int(rcp['image_width'])
        return target

    def __getitem__(self, index):
        source_key = random.choice(self.train_keys)
        if index < len(self.real_keys):
            return self._build_record(source_key, self.real_keys[index])
        rec = self.synth[index - len(self.real_keys)]
        return {'source': self._load_source(source_key), 'target': self._build_synth(rec)}
