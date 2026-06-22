"""Export a trained GUAVA Ubody_Gaussian avatar to a browser-consumable bundle.

The bundle mirrors LAM's packaging (a LAM-style SkinnedMesh `ehm_skin.glb` +
`vertex_order.json`) and extends it for GUAVA's two gaussian groups (vertex-bound
+ UV/face-bound). The browser reproduces the EHM mesh deformation (skin + 52 ARKit
morph targets), then applies GUAVA's two deformation formulas in the splat shader.

Run from the GUAVA repo root (so `models/`, `utils/`, `dataset/` import).

    python export_browser_bundle.py \
        --checkpoint assets/GUAVA \
        --source <source.pkl | tracked_dir> \
        --out outputs/bundle/<avatar> \
        [--arkit_bs path/to/flame_arkit_bs.npy] \
        [--rig head|body] [--motions tracked_dir ...] \
        [--gs_canonical path/to/GS_canonical.ply]

Two rigs:
  --rig head (default): fixed captured body pose + 52 ARKit head morphs; the per-vertex
    frame is constant and baked (vertex_frame.bin).
  --rig body: canonical (rest) base mesh + full 55-joint SMPL-X skeleton + per-vertex
    skin weights + a sidecar motion library (motions.json + clip_*.bin) extracted from
    tracked videos. The viewer derives per-vertex frames from skinning each frame.

See BROWSER_BUNDLE_README.md for the exact output format.

KEY FACTS reproduced here (see models/UbodyAvatar/ubody_gaussian.py):
  * Two gaussian groups, concatenated: vertex group (N = smplx.v_template.shape[0],
    1 splat per EHM vertex) then UV group (M splats, bound to faces).
  * Quaternions are stored wxyz.
  * Because the body is held at a FIXED pose, the per-vertex LBS rotation
    ver_transform_mat[:, :3, :3] is CONSTANT w.r.t. expression (a blended LBS
    rotation depends only on pose rotations + skin weights, not joint positions;
    expression only moves joint positions). So we bake it once as vertex_frame.bin.
"""
import os
import sys
import copy
import json
import struct
import pickle
import argparse

sys.path.insert(0, os.getcwd())  # resolve repo packages (models/, utils/, dataset/) from cwd

import numpy as np
import torch
import torchvision
import lightning
from plyfile import PlyData
from torch.utils.data._utils.collate import default_collate

from models.UbodyAvatar import Ubody_Gaussian_inferer, Ubody_Gaussian
from models.modules.smplx.SMPLX import SMPLX_names as SMPLX_NAMES
from dataset.data_loader import data_to_tensor, squeeze_params, load_dict_pkl
from utils.lmdb import LMDBEngine
from utils.general_utils import ConfigDict, add_extra_cfgs, find_pt_file
from utils.graphics_utils import get_full_proj_matrix, compute_face_orientation
from roma import (rotmat_to_unitquat, rotvec_to_unitquat, quat_xyzw_to_wxyz,
                  quat_product, quat_wxyz_to_xyzw)

# camera convention copied from backview-dataset/stage2_guava_render.py
C2C_MAT = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)

# Canonical ARKit-52 blendshape order (matches LAM bsData.json / iOS ARKit).
ARKIT_52 = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
    'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose',
    'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
    'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight',
    'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
    'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut',
]

# SMPL-X joint indices for the head bones we expose (see models/.../smplx/SMPLX.py).
#   0 pelvis(root)  12 neck  15 head  22 jaw  23 left_eye  24 right_eye
BONES = [('root', 0, -1), ('neck', 12, 0), ('jaw', 22, 1), ('leftEye', 23, 1), ('rightEye', 24, 1)]
BONE_JOINT = [b[1] for b in BONES]          # smplx joint id per bone
BONE_PARENT = [b[2] for b in BONES]         # parent index into BONES (-1 = root)


# --------------------------------------------------------------------------------------
# avatar loading (mirrors stage2_guava_render.py / main/test.py)
# --------------------------------------------------------------------------------------
def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    return data


def load_inferer(model_path, device):
    meta_cfg = add_extra_cfgs(ConfigDict(model_config_path=os.path.join(model_path, 'config.yaml')))
    lightning.fabric.seed_everything(10)
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL).to(device).eval()
    ckpt_dir = os.path.join(model_path, 'checkpoints')
    ckpt = find_pt_file(ckpt_dir, 'best') or find_pt_file(ckpt_dir, 'latest')
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(state['model'], strict=False)
    print(f'[export] loaded GUAVA inferer from {ckpt}', flush=True)
    return meta_cfg, infer_model


def merge_id_params(smplx_coeffs, flame_coeffs, id_params):
    sc, fc = dict(smplx_coeffs), dict(flame_coeffs)
    sc.update({'shape': id_params['smplx_shape'][0], 'joints_offset': id_params['joints_offset'][0],
               'head_scale': id_params['head_scale'][0], 'hand_scale': id_params['hand_scale'][0]})
    fc.update({'shape_params': id_params['flame_shape'][0]})
    return sc, fc


def build_batch(smplx_coeffs, flame_coeffs, id_params, device, tanfov, image_size, image=None):
    """Mirror stage2_guava_render.build_batch: assemble one EHM/render batch."""
    info = {'smplx_coeffs': {}, 'flame_coeffs': {}}
    info['smplx_coeffs'], info['flame_coeffs'] = merge_id_params(smplx_coeffs, flame_coeffs, id_params)
    info = squeeze_params(data_to_tensor(copy.deepcopy(info)))

    RT = info['smplx_coeffs']['camera_RT_params']
    RT_mat = torch.eye(4, dtype=torch.float32)
    RT_mat[:3, :4] = RT
    w2c_cam = C2C_MAT @ RT_mat
    info['w2c_cam'], info['c2w_cam'] = w2c_cam, torch.linalg.inv(w2c_cam)
    view_matrix, full_proj_matrix = get_full_proj_matrix(w2c_cam, tanfov)
    info['render_cam_params'] = {
        'world_view_transform': view_matrix, 'full_proj_transform': full_proj_matrix,
        'tanfovx': tanfov, 'tanfovy': tanfov, 'image_height': image_size, 'image_width': image_size,
        'camera_center': info['c2w_cam'][:3, 3]}
    if image is not None:
        info['image'] = image
    return move_to_device(default_collate([info]), device)


def _to_chw01(img):
    """Normalize a body image to CHW float in [0,1] from numpy HWC-uint8 or CHW tensor."""
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img)
        if t.ndim == 3 and t.shape[2] in (1, 3):   # HWC -> CHW
            t = t.permute(2, 0, 1)
        t = t.float()
    else:
        t = img.float()
    return t / 255.0 if t.max() > 1.5 else t


def load_source(path):
    """Dispatch on a backview source.pkl (file) or a GUAVA tracked dir (with
    optim_tracking_ehm.pkl). Returns a normalized src: raw coeffs/id_params +
    a masked CHW[0,1] `image_masked` tensor."""
    if os.path.isdir(path):
        return load_source_tracked_dir(path)
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    need = ('body_image', 'body_mask', 'smplx_coeffs', 'flame_coeffs', 'id_params')
    missing = [k for k in need if k not in raw]
    if missing:
        raise KeyError(f'source pkl missing keys {missing}; expected backview source.pkl layout')
    img = _to_chw01(raw['body_image'])
    mask = _to_chw01(raw['body_mask'])
    if mask.ndim == 2:
        mask = mask[None]
    return {'smplx_coeffs': raw['smplx_coeffs'], 'flame_coeffs': raw['flame_coeffs'],
            'id_params': raw['id_params'], 'image_masked': img * mask}


def load_source_tracked_dir(path, key_idx=0):
    """Load an avatar from a GUAVA tracked source dir (the app's tracked_source_image/<name>/:
    optim_tracking_ehm.pkl + id_share_params.pkl + videos_info.json + img_lmdb/). Mirrors
    dataset.data_loader.TrackedData_infer._load_one_info / _load_source_info."""
    id_share = load_dict_pkl(os.path.join(path, 'id_share_params.pkl'))
    traked = load_dict_pkl(os.path.join(path, 'optim_tracking_ehm.pkl'))
    with open(os.path.join(path, 'videos_info.json')) as f:
        videos_info = json.load(f)
    video_id = list(videos_info.keys())[0]
    source_key = videos_info[video_id]['frames_keys'][key_idx]
    if video_id in id_share:                      # dataset layout
        idp, frame_info, img_prefix = id_share[video_id], traked[video_id][source_key], f'{video_id}/{source_key}'
    else:                                         # single-subject app layout
        idp, frame_info, img_prefix = id_share, traked[source_key], source_key
    engine = LMDBEngine(os.path.join(path, 'img_lmdb'), write=False)
    img = engine[f'{img_prefix}/body_image'].float() / 255.0      # CHW
    mask = engine[f'{img_prefix}/body_mask'].float() / 255.0       # CHW
    engine.close() if hasattr(engine, 'close') else None
    id_params = {k: idp[k] for k in ('smplx_shape', 'joints_offset', 'head_scale', 'hand_scale', 'flame_shape')}
    return {'smplx_coeffs': copy.deepcopy(frame_info['smplx_coeffs']),
            'flame_coeffs': copy.deepcopy(frame_info['flame_coeffs']),
            'id_params': id_params, 'image_masked': img * mask}


def reconstruct_avatar(meta_cfg, infer_model, src, device):
    """Run inference once to instantiate the live Ubody_Gaussian for this avatar."""
    tanfov = 1.0 / meta_cfg.MODEL.invtanfov
    image_size = meta_cfg.MODEL.image_size
    source_image = torchvision.transforms.functional.resize(
        src['image_masked'], (meta_cfg.MODEL.feature_img_size,) * 2, antialias=True)
    source_batch = build_batch(src['smplx_coeffs'], src['flame_coeffs'], src['id_params'],
                               device, tanfov, image_size, image=source_image)
    vertex_gs_dict, uv_point_gs_dict, _ = infer_model(source_batch)
    # pruning=False so the splat count matches the unpruned GS_canonical.ply (203910 for the test avatar)
    ubody = Ubody_Gaussian(meta_cfg.MODEL, vertex_gs_dict, uv_point_gs_dict, pruning=False)
    ubody.init_ehm(infer_model.ehm)
    ubody.eval()
    return ubody, tanfov, image_size


# --------------------------------------------------------------------------------------
# EHM evaluation at the fixed body pose
# --------------------------------------------------------------------------------------
def neutral_flame(flame_coeffs):
    """Zero the head-expression channels, keep identity (shape) and head pose."""
    fc = copy.deepcopy(flame_coeffs)
    for k in ('expression_params', 'jaw_params', 'eye_pose_params', 'eyelid_params'):
        if k in fc and fc[k] is not None:
            fc[k] = np.zeros_like(np.asarray(fc[k], dtype=np.float32))
    return fc


@torch.no_grad()
def eval_ehm(ubody, src, flame_coeffs, device, tanfov, image_size, smplx_coeffs=None):
    """Call EHM exactly as Ubody_Gaussian.forward does (static_offset = _smplx_offset).
    smplx_coeffs overrides src's body coeffs (used for the canonical/rest base mesh)."""
    sc = smplx_coeffs if smplx_coeffs is not None else src['smplx_coeffs']
    batch = build_batch(sc, flame_coeffs, src['id_params'], device, tanfov, image_size)
    res = ubody.ehm(batch['smplx_coeffs'], batch['flame_coeffs'], static_offset=ubody._smplx_offset)
    return res


def canonical_smplx_coeffs(smplx_coeffs):
    """Zero global/body/hand pose -> rest (bind) pose, keeping the avatar's shape.
    Used as the body-rig bind pose so skinning can re-pose from rest."""
    sc = copy.deepcopy(smplx_coeffs)
    for k in ('global_pose', 'body_pose', 'left_hand_pose', 'right_hand_pose'):
        if k in sc and sc[k] is not None:
            sc[k] = np.zeros_like(np.asarray(sc[k], np.float32))
    return sc


# --------------------------------------------------------------------------------------
# minimal binary glTF (.glb) writer: skinned mesh + morph targets, no extra deps
# --------------------------------------------------------------------------------------
FLOAT, U32, U8 = 5126, 5125, 5121
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


class GLB:
    def __init__(self):
        self.blob = bytearray()
        self.views = []
        self.accessors = []

    def _view(self, data: bytes, target=None):
        while len(self.blob) % 4:
            self.blob.append(0)
        off = len(self.blob)
        self.blob += data
        v = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
        if target is not None:
            v['target'] = target
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, arr, comp_type, type_str, target=None, with_minmax=False):
        arr = np.ascontiguousarray(arr)
        vi = self._view(arr.tobytes(), target)
        acc = {'bufferView': vi, 'componentType': comp_type, 'count': int(arr.shape[0]),
               'type': type_str}
        if with_minmax:
            flat = arr.reshape(arr.shape[0], -1)
            acc['min'] = flat.min(0).astype(float).tolist()
            acc['max'] = flat.max(0).astype(float).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def write(self, path, gltf):
        gltf['buffers'] = [{'byteLength': len(self.blob)}]
        gltf['bufferViews'] = self.views
        gltf['accessors'] = self.accessors
        js = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
        while len(js) % 4:
            js += b' '
        bin_chunk = bytes(self.blob)
        while len(bin_chunk) % 4:
            bin_chunk += b'\x00'
        total = 12 + 8 + len(js) + 8 + len(bin_chunk)
        with open(path, 'wb') as f:
            f.write(struct.pack('<III', 0x46546C67, 2, total))      # glTF, version 2, length
            f.write(struct.pack('<II', len(js), 0x4E4F534A))        # JSON chunk
            f.write(js)
            f.write(struct.pack('<II', len(bin_chunk), 0x004E4942))  # BIN chunk
            f.write(bin_chunk)


def write_skin_glb(path, verts, faces, joints_idx, joints_w, morphs, morph_names,
                   bone_pos, bone_parent, bone_names):
    """verts (N,3) f32, faces (F,3) u32, morphs (K,N,3) f32 deltas, bone_pos (B,3)
    world rest positions, bone_parent (B,) parent index into bones (-1 = root).
    joints_idx / joints_w are (N,W) with W a multiple of 4; split into glTF
    JOINTS_n/WEIGHTS_n sets of 4 (W=8 -> two sets; Three.js native skinning reads
    only set 0, a custom splat shader can use both)."""
    g = GLB()
    pos_a = g.accessor(verts.astype(np.float32), FLOAT, 'VEC3', ARRAY_BUFFER, with_minmax=True)
    idx_a = g.accessor(faces.astype(np.uint32).reshape(-1), U32, 'SCALAR', ELEMENT_ARRAY_BUFFER)

    attributes = {'POSITION': pos_a}
    assert joints_idx.shape[1] % 4 == 0 and joints_idx.shape == joints_w.shape
    for s in range(joints_idx.shape[1] // 4):
        ji = g.accessor(np.ascontiguousarray(joints_idx[:, 4 * s:4 * s + 4]).astype(np.uint8),
                        U8, 'VEC4', ARRAY_BUFFER)
        wi = g.accessor(np.ascontiguousarray(joints_w[:, 4 * s:4 * s + 4]).astype(np.float32),
                        FLOAT, 'VEC4', ARRAY_BUFFER)
        attributes[f'JOINTS_{s}'] = ji
        attributes[f'WEIGHTS_{s}'] = wi
    target_accs = [g.accessor(m.astype(np.float32), FLOAT, 'VEC3', ARRAY_BUFFER, with_minmax=True)
                   for m in morphs]

    # inverse bind matrices: rest joints have no rotation -> IBM = translate(-pos), column-major
    bone_pos = np.asarray(bone_pos, dtype=np.float32)
    ibm = np.tile(np.eye(4, dtype=np.float32), (len(bone_pos), 1, 1))
    ibm[:, :3, 3] = -bone_pos
    ibm_cm = np.transpose(ibm, (0, 2, 1)).reshape(len(bone_pos), 16)
    ibm_a = g.accessor(ibm_cm.astype(np.float32), FLOAT, 'MAT4')

    prim = {'attributes': attributes, 'indices': idx_a}
    if len(target_accs):
        prim['targets'] = [{'POSITION': a} for a in target_accs]
        prim['extras'] = {'targetNames': morph_names}
    mesh = {'primitives': [prim], 'extras': {'targetNames': morph_names}}
    if len(target_accs):
        mesh['weights'] = [0.0] * len(target_accs)

    # nodes: 0 = skinned mesh, 1.. = bone joints (local translation = pos - parent_pos)
    nodes = [{'name': 'avatar', 'mesh': 0, 'skin': 0}]
    joint_node_base = 1
    for i, (name, parent) in enumerate(zip(bone_names, bone_parent)):
        ppos = np.zeros(3) if parent < 0 else bone_pos[parent]
        nodes.append({'name': name, 'translation': (bone_pos[i] - ppos).astype(float).tolist()})
    for i, parent in enumerate(bone_parent):
        children = [joint_node_base + j for j, p in enumerate(bone_parent) if p == i]
        if children:
            nodes[joint_node_base + i]['children'] = children

    skin = {'inverseBindMatrices': ibm_a,
            'joints': [joint_node_base + i for i in range(len(bone_names))],
            'skeleton': joint_node_base}
    root_joint = joint_node_base + bone_parent.index(-1)
    gltf = {'asset': {'version': '2.0', 'generator': 'GUAVA export_browser_bundle'},
            'scene': 0, 'scenes': [{'nodes': [0, root_joint]}],
            'nodes': nodes, 'meshes': [mesh], 'skins': [skin]}
    g.write(path, gltf)


# --------------------------------------------------------------------------------------
# skin weights: reduce SMPL-X 55-joint LBS weights to the 5 exposed bones, top-4 per vertex
# --------------------------------------------------------------------------------------
def build_skin_weights(lbs_weights):
    w5 = lbs_weights[:, BONE_JOINT].clone()          # (N,5)
    leftover = (1.0 - lbs_weights[:, BONE_JOINT].sum(1)).clamp(min=0.0)
    w5[:, 0] = w5[:, 0] + leftover                   # unmodelled joints ride the fixed root
    top = torch.topk(w5, k=4, dim=1)
    idx = top.indices.to(torch.uint8).cpu().numpy()
    val = top.values
    val = (val / val.sum(1, keepdim=True).clamp(min=1e-8)).cpu().numpy()
    return idx, val


# --------------------------------------------------------------------------------------
# body rig: full 55-joint SMPL-X skeleton + top-k (8) per-vertex weights
# --------------------------------------------------------------------------------------
def build_body_skeleton(ubody, base_verts):
    """Rest joints/parents/names for the full SMPL-X skeleton at the canonical mesh.
    base_verts = v_template + offset (N,3). Returns (bone_pos (J,3), bone_parent list, names)."""
    from models.modules.smplx.lbs import vertices2joints
    J = vertices2joints(ubody.smplx.J_regressor, base_verts[None])[0]      # (J,3) rest joints
    parents = ubody.smplx.parents.cpu().numpy().astype(np.int64).copy()
    parents[0] = -1                                                        # root sentinel
    names = list(SMPLX_NAMES[:J.shape[0]])
    return J.cpu().numpy().astype(np.float32), parents.tolist(), names


def build_body_skin_weights(lbs_weights, k=8):
    """Top-k bone influences per vertex over all 55 joints, indices into joint id, renormalized.
    Returns idx (N,k) uint8, w (N,k) f32 (k padded to a multiple of 4)."""
    k = ((k + 3) // 4) * 4
    k = min(k, lbs_weights.shape[1])
    top = torch.topk(lbs_weights, k=k, dim=1)
    idx = top.indices.to(torch.uint8).cpu().numpy()
    val = top.values
    val = (val / val.sum(1, keepdim=True).clamp(min=1e-8)).cpu().numpy()
    if idx.shape[1] % 4:                                                    # pad to multiple of 4
        pad = 4 - idx.shape[1] % 4
        idx = np.pad(idx, ((0, 0), (0, pad)))
        val = np.pad(val, ((0, 0), (0, pad)))
    return idx, val


# --------------------------------------------------------------------------------------
# morph targets
# --------------------------------------------------------------------------------------
@torch.no_grad()
def morphs_from_arkit_bs(ubody, arkit_bs_path, head_scale, N, device):
    """Scatter LAM's per-FLAME-vertex ARKit basis (52,5023,3) onto the SMPL-X head verts.
    EHM stitches the FLAME head with a head_scale factor, so deltas scale by head_scale."""
    bs = np.load(arkit_bs_path).astype(np.float32)
    assert bs.shape[0] == 52 and bs.shape[2] == 3, f'expected (52,V,3), got {bs.shape}'
    flame_ind = ubody.smplx.smplx2flame_ind[:bs.shape[1]]   # first V FLAME verts -> SMPL-X verts
    hs = float(np.mean(np.asarray(head_scale))) if head_scale is not None else 1.0
    morphs = np.zeros((52, N, 3), dtype=np.float32)
    for k in range(52):
        morphs[k, flame_ind] = bs[k] * hs
    return morphs, list(ARKIT_52)


@torch.no_grad()
def morphs_from_flame(ubody, src, base_verts, device, tanfov, image_size, n_exp=10, smplx_coeffs=None):
    """Fallback: raw FLAME basis morphs (no ARKit map). One target per expression dim
    plus jaw-open / eye / eyelid. Driven through EHM and differenced against neutral.
    smplx_coeffs (canonical rest) keeps morph deltas in the pre-skin space for body mode."""
    morphs, names = [], []
    fc0 = neutral_flame(src['flame_coeffs'])

    def delta(fc):
        v = eval_ehm(ubody, src, fc, device, tanfov, image_size, smplx_coeffs=smplx_coeffs)['vertices'][0]
        return (v - base_verts).cpu().numpy().astype(np.float32)

    for j in range(n_exp):
        fc = copy.deepcopy(fc0)
        fc['expression_params'] = np.asarray(fc['expression_params'], np.float32).copy()
        fc['expression_params'][j] = 1.0
        morphs.append(delta(fc)); names.append(f'flameExp{j:02d}')
    fc = copy.deepcopy(fc0); fc['jaw_params'] = np.array([0.3, 0, 0], np.float32)
    morphs.append(delta(fc)); names.append('jawOpen')
    if 'eyelid_params' in fc0 and fc0['eyelid_params'] is not None:
        fc = copy.deepcopy(fc0); fc['eyelid_params'] = np.array([1.0, 1.0], np.float32)
        morphs.append(delta(fc)); names.append('eyeBlink')
    return np.stack(morphs), names


# --------------------------------------------------------------------------------------
# ARKit blendshape refit (mirrors ELITE's morphs_refit_arkit, adapted to GUAVA's EHM space)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def morphs_refit_arkit(ubody, src, device, tanfov, image_size, arkit_bs_path,
                        tracked_frames, head_scale, N,
                        reg=0.1, max_frames=600, max_gain=1.5, underuse_cap=20.0,
                        smplx_coeffs=None):
    """Refit the 52 ARKit morph deltas to this avatar's tracked FLAME expressions.

    Steps:
      (1) Generic basis B0 from LAM arkit_bs.npy; rotate from SMPL-X template space into
          the EHM evaluation space using the per-vertex LBS rotation at the base pose.
      (2) GT per-frame expression vertex delta via EHM (jaw/eye/eyelid zeroed -- those
          are driven by the skeleton bones, not morphs).
      (3) Ridge regression to infer per-frame ARKit weights A from B0.
      (4) Coverage-weighted ridge refit: under-exercised shapes stay close to generic;
          well-exercised shapes are allowed to move toward the GT deltas.
      (5) Per-shape gain clamp: never exceed max_gain x the generic magnitude.

    smplx_coeffs overrides the body coeffs used for every EHM eval (head rig: leave
    None -> the fixed captured pose, deltas are world-space on the posed base mesh;
    body rig: pass the canonical rest coeffs -> at rest R is identity so deltas stay
    in the pre-skin canonical space, matching morphs_from_flame(smplx_coeffs=sc_canon)).
    Returns morph deltas [52, N, 3] additive on the base mesh, consistent with
    morphs_from_flame.
    """
    B0_t_np, names = morphs_from_arkit_bs(ubody, arkit_bs_path, head_scale, N, device)
    B0_t = torch.from_numpy(B0_t_np).to(device)  # [52, N, 3] template space

    fc0 = neutral_flame(src['flame_coeffs'])
    res0 = eval_ehm(ubody, src, fc0, device, tanfov, image_size, smplx_coeffs=smplx_coeffs)
    base_verts = res0['vertices'][0]              # [N, 3] neutral base mesh (posed or rest)
    R = res0['ver_transform_mat'][0, :, :3, :3]  # [N, 3, 3] per-vertex LBS rotation (==I at rest)

    # rotate generic basis into the EHM eval space so it shares a space with the deltas
    B0 = torch.einsum('vji,kvi->kvj', R, B0_t).reshape(52, -1)  # [52, 3N]

    # per-frame expression delta (zero jaw/eyes — those go via the skeleton bones)
    T_frames = len(tracked_frames)
    idx = np.linspace(0, T_frames - 1, min(T_frames, max_frames)).astype(int)
    D = []
    for i in idx:
        frame = tracked_frames[int(i)]
        fc = copy.deepcopy(frame['flame_coeffs'])
        for key in ('jaw_params', 'eye_pose_params', 'eyelid_params'):
            if key in fc:
                fc[key] = np.zeros_like(np.asarray(fc[key], dtype=np.float32))
        v = eval_ehm(ubody, src, fc, device, tanfov, image_size, smplx_coeffs=smplx_coeffs)['vertices'][0]
        D.append((v - base_verts).reshape(-1))
    D = torch.stack(D, 0)  # [F, 3N]

    # (3) ridge: infer per-frame ARKit weights from generic world-space basis
    eye52 = torch.eye(52, device=device)
    G = B0 @ B0.T
    lam_c = (1e-3 * G.diagonal().mean()).clamp(min=1e-8)
    A = torch.linalg.solve(G + lam_c * eye52, B0 @ D.T).T.clamp(0.0, 1.0)  # [F, 52]

    # (4) coverage-weighted ridge refit
    H = A.T @ A
    cov = torch.quantile(A, 0.95, dim=0).clamp(min=1e-3)  # [52]
    w = (1.0 / cov).clamp(1.0, underuse_cap)
    lam_vec = (reg * H.diagonal().mean()).clamp(min=1e-6) * w
    Bstar = torch.linalg.solve(H + torch.diag(lam_vec),
                               A.T @ D + lam_vec[:, None] * B0)  # [52, 3N]

    # (5) per-shape gain clamp
    n0, ns = B0.norm(dim=1), Bstar.norm(dim=1)
    scale = (max_gain * n0 / ns.clamp(min=1e-8)).clamp(max=1.0)
    Bstar = Bstar * scale[:, None]

    eg = (A @ B0 - D).reshape(-1, 3).norm(dim=-1).mean()
    er = (A @ Bstar - D).reshape(-1, 3).norm(dim=-1).mean()
    n_low = int((cov < 0.2).sum())
    n_clamp = int((scale < 0.999).sum())
    print(f'[export] ARKit refit on {len(idx)} frames: recon err '
          f'generic={eg:.3e} -> refit={er:.3e} ({100*(1-er/eg):.0f}% lower) | '
          f'{n_low}/52 under-exercised (pulled to generic), {n_clamp} gain-clamped')

    return Bstar.reshape(52, N, 3).cpu().numpy().astype(np.float32), names


# --------------------------------------------------------------------------------------
# body motion clips (sidecar): per-frame local joint quaternions, SMPL-X joint order
# --------------------------------------------------------------------------------------
def frames_to_quats(frames):
    """List of per-frame info dicts -> (F,55,4) local joint quaternions wxyz, SMPL-X order
    (0 global, 1-21 body, 22 jaw, 23-24 eyes, 25-39 left hand, 40-54 right hand).
    jaw/eyes are left at identity here because the face is driven by the ARKit morphs."""
    n = len(frames)
    aa = np.zeros((n, 55, 3), dtype=np.float32)
    for i, fi in enumerate(frames):
        sc = fi['smplx_coeffs']
        aa[i, 0] = np.asarray(sc['global_pose'], np.float32).reshape(-1)[:3]
        aa[i, 1:22] = np.asarray(sc['body_pose'], np.float32).reshape(21, 3)
        if sc.get('left_hand_pose') is not None:
            aa[i, 25:40] = np.asarray(sc['left_hand_pose'], np.float32).reshape(15, 3)
        if sc.get('right_hand_pose') is not None:
            aa[i, 40:55] = np.asarray(sc['right_hand_pose'], np.float32).reshape(15, 3)
    q = quat_xyzw_to_wxyz(rotvec_to_unitquat(torch.from_numpy(aa).reshape(-1, 3)))
    return q.reshape(n, 55, 4).cpu().numpy().astype(np.float32)


def read_tracked_frames(path):
    """Ordered per-frame info + fps from a tracked dir (multi-frame optim_tracking_ehm.pkl)."""
    traked = load_dict_pkl(os.path.join(path, 'optim_tracking_ehm.pkl'))
    id_share = load_dict_pkl(os.path.join(path, 'id_share_params.pkl'))
    with open(os.path.join(path, 'videos_info.json')) as f:
        videos_info = json.load(f)
    video_id = list(videos_info.keys())[0]
    fkeys = videos_info[video_id]['frames_keys']
    multi = video_id in id_share
    frames = [traked[video_id][k] if multi else traked[k] for k in fkeys]
    fps = videos_info[video_id].get('fps', 30)
    return frames, fps


def export_motions(out_dir, src, motion_paths):
    """Write clip_*.bin + return clip metadata. Always includes 'capturedDefault' (1 frame)."""
    clips, fps = [], 30
    q = frames_to_quats([{'smplx_coeffs': src['smplx_coeffs']}])
    q.tofile(os.path.join(out_dir, 'clip_capturedDefault.bin'))
    clips.append({'name': 'capturedDefault', 'frames': 1, 'file': 'clip_capturedDefault.bin'})
    for mp in motion_paths or []:
        frames, fps = read_tracked_frames(mp)
        name = os.path.basename(mp.rstrip('/')) or 'clip'
        fn = f'clip_{name}.bin'
        frames_to_quats(frames).tofile(os.path.join(out_dir, fn))
        clips.append({'name': name, 'frames': len(frames), 'file': fn})
        print(f'[export] motion clip "{name}": {len(frames)} frames')
    return {'fps': fps, 'rotationOrder': 'wxyz', 'boneOrder': list(SMPLX_NAMES[:55]),
            'jointBin': {'dtype': 'float32', 'shape': ['frames', 55, 4]},
            'rootTranslation': False, 'clips': clips}


# --------------------------------------------------------------------------------------
# gaussian extraction
# --------------------------------------------------------------------------------------
@torch.no_grad()
def extract_gaussians(ubody):
    """Return local/template-frame attributes for both groups, concat order vertex|uv."""
    vx_rot = ubody._smplx_rotation[0]                 # (N,4) wxyz local
    vx_scale = ubody._smplx_scaling[0]                # (N,3)
    vx_op = ubody._smplx_opacity[0]                   # (N,1) in [0,1]
    vx_col = ubody._smplx_features_color[0, :, :3]    # (N,3) sigmoid'd RGB
    N = vx_rot.shape[0]
    vx_local = torch.zeros((N, 3), device=vx_rot.device)  # vertex splats ride their vertex

    uv_local = ubody._uv_local_xyz[0]                 # (M,3) local position in face frame
    uv_rot = ubody._uv_rotation[0]                    # (M,4) wxyz local
    uv_scale = ubody._uv_scaling[0]                   # (M,3) local (pre face_scaling)
    uv_op = ubody._uv_opacity[0]                      # (M,1)
    uv_col = ubody._uv_features_color[0, :, :3]       # (M,3)
    M = uv_rot.shape[0]

    local_xyz = torch.cat([vx_local, uv_local], 0)
    rot = torch.cat([vx_rot, uv_rot], 0)
    scale = torch.cat([vx_scale, uv_scale], 0)
    op = torch.cat([vx_op, uv_op], 0)
    col = torch.cat([vx_col, uv_col], 0)
    cols14 = torch.cat([local_xyz, rot, scale, op, col], 1)  # (N+M, 14)
    binding_face = ubody._uv_binding_face.to(torch.int32)    # (M,)
    face_bary = ubody._uv_face_bary                          # (M,3)
    return cols14.cpu().numpy().astype(np.float32), N, M, \
        binding_face.cpu().numpy(), face_bary.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------
@torch.no_grad()
def reconstruct_canonical_positions(ubody, cols14, N, binding_face, face_bary):
    """Rebuild canonical splat xyz from the EXPORTED bundle and the canonical mesh
    (v_template + offset), exactly as get_canoical_gaussians does. Returns (N+M,3)."""
    off = ubody._smplx_offset[0] if ubody._smplx_offset is not None else 0.0
    v_template = ubody._smplx_xyz[0] + off                           # canonical mesh
    return reconstruct_positions_from_mesh(ubody, cols14, N, binding_face, face_bary, v_template)


def load_ply_xyz(path):
    ply = PlyData.read(path)
    el = ply.elements[0]
    return np.stack([el['x'], el['y'], el['z']], 1).astype(np.float32)


@torch.no_grad()
def reconstruct_positions_from_mesh(ubody, cols14, N, binding_face, face_bary, V):
    """Apply the two GUAVA splat formulas (vertex + UV/face) on an arbitrary deformed mesh V."""
    dev = V.device
    faces = ubody.smplx.faces_tensor
    orien, scaling = compute_face_orientation(V[None], faces, return_scale=True)
    orien, scaling = orien[0], scaling[0]
    uv_local = torch.from_numpy(cols14[N:, 0:3]).to(dev)
    bf = torch.from_numpy(binding_face).long().to(dev)
    bary = torch.from_numpy(face_bary).to(dev)
    center = torch.einsum('mk,mkj->mj', bary, V[faces][bf])
    uv_xyz = torch.einsum('mij,mj->mi', orien[bf], uv_local) * scaling[bf] + center
    return torch.cat([V, uv_xyz], 0)


@torch.no_grad()
def verify_posed_frame(ubody, src, base_verts, cols14, N, binding_face, face_bary,
                       frame_smplx, device, tanfov, image_size):
    """Skin the canonical base by a frame's joint rotations WITHOUT pose blendshapes (what the
    browser does), apply the splat formulas, and compare to ubody.forward (which includes
    posedirs). The reported error is the posedirs gap the viewer will exhibit."""
    from models.modules.smplx.lbs import batch_rigid_transform, batch_rodrigues, vertices2joints
    # full 55-joint local rotations (axis-angle) for this frame; jaw/eyes identity (morph-driven)
    aa = frames_to_quats([{'smplx_coeffs': frame_smplx}])           # reuse joint assembly (-> wxyz)
    # convert the wxyz quats back to rotmats for FK
    q = torch.from_numpy(aa[0]).to(device)                          # (55,4) wxyz
    rot = quat_wxyz_to_xyzw(q)
    from roma import unitquat_to_rotmat
    rot_mats = unitquat_to_rotmat(rot)[None]                        # (1,55,3,3)
    J = vertices2joints(ubody.smplx.J_regressor, base_verts[None])  # (1,55,3) shaped rest joints
    _, A = batch_rigid_transform(rot_mats, J, ubody.smplx.parents)  # (1,55,4,4)
    W = ubody.smplx.lbs_weights[None]                               # (1,N,55)
    T = torch.matmul(W, A.view(1, A.shape[1], 16)).view(1, -1, 4, 4)
    homo = torch.cat([base_verts, torch.ones(N, 1, device=device)], -1)
    V = torch.matmul(T[0], homo[..., None])[:, :3, 0]              # (N,3) skinned, no posedirs
    recon = reconstruct_positions_from_mesh(ubody, cols14, N, binding_face, face_bary, V)
    # ground truth: GUAVA forward (includes posedirs)
    batch = build_batch(frame_smplx, neutral_flame(src['flame_coeffs']), src['id_params'],
                        device, tanfov, image_size)
    gt = ubody({'smplx_coeffs': batch['smplx_coeffs'], 'flame_coeffs': batch['flame_coeffs']})['xyz'][0]
    err = torch.linalg.norm(recon - gt, dim=1)
    return float(err.max()), float(err.mean())


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default='assets/GUAVA', help='dir with config.yaml + checkpoints/')
    ap.add_argument('--source', required=True,
                    help='backview source.pkl OR a GUAVA tracked dir (with optim_tracking_ehm.pkl)')
    ap.add_argument('--out', required=True, help='output bundle directory')
    ap.add_argument('--arkit_bs', default=None,
                    help="LAM flame_arkit_bs.npy (52,5023,3); if omitted, exports raw FLAME morphs")
    ap.add_argument('--refit_arkit', default=True, action=argparse.BooleanOptionalAction,
                    help='refit ARKit morph basis to this avatar\'s tracked expressions (default: on)')
    ap.add_argument('--refit_reg', type=float, default=0.1,
                    help='coverage-weighted regularization strength for ARKit refit (default 0.1)')
    ap.add_argument('--refit_max_gain', type=float, default=1.5,
                    help='max per-shape gain vs generic magnitude (default 1.5)')
    ap.add_argument('--refit_max_frames', type=int, default=600,
                    help='max tracked frames to sample for the refit regression (default 600)')
    ap.add_argument('--rig', choices=['head', 'body'], default='head',
                    help="'head' = fixed body + ARKit morphs (baked vertex frame); "
                         "'body' = canonical rest + full SMPL-X skeleton + motion clips")
    ap.add_argument('--motions', nargs='*', default=None,
                    help='(body rig) tracked dirs to extract motion clips from')
    ap.add_argument('--gs_canonical', default=None,
                    help='GS_canonical.ply to round-trip check against (optional)')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    torch.set_float32_matmul_precision('high')
    os.makedirs(args.out, exist_ok=True)
    device = args.device

    meta_cfg, infer_model = load_inferer(args.checkpoint, device)
    src = load_source(args.source)
    ubody, tanfov, image_size = reconstruct_avatar(meta_cfg, infer_model, src, device)

    # load tracked frames now (needed for ARKit refit); source can be a dir with optim_tracking_ehm.pkl
    tracked_frames = []
    if args.arkit_bs and args.refit_arkit and os.path.isdir(args.source):
        tracked_frames, _ = read_tracked_frames(args.source)
        print(f'[export] loaded {len(tracked_frames)} tracked frames for ARKit refit')
    ubody.get_canoical_gaussians()   # populate _uv_*_cano used by save_gaussian_ply

    # ----- gaussians (local/template-frame) -----
    cols14, N, M, binding_face, face_bary = extract_gaussians(ubody)
    total = N + M
    print(f'[export] numVertexGaussians={N}  numUvGaussians={M}  total={total}')
    if total != 203910:
        print(f'[export] WARNING: total {total} != 203910 (expected for the provided test avatar)')

    faces = ubody.smplx.faces_tensor.cpu().numpy().astype(np.uint32)
    fc_neutral = neutral_flame(src['flame_coeffs'])
    head_scale = src['id_params'].get('head_scale')

    binding = {
        'rig': args.rig,
        'numVertexGaussians': N, 'numUvGaussians': M, 'shDegree': int(meta_cfg.MODEL.sh_degree),
        'quaternion': 'wxyz', 'upAxis': '+Y', 'coordSpace': 'same as GS_canonical.ply (SMPL-X world)',
        'gaussianColumns': ['local_xyz(3)', 'rotation_wxyz(4)', 'scale(3)', 'opacity(1)', 'color_rgb(3)'],
        'gaussiansDtype': 'float32', 'gaussiansStride': 14,
        'uv': {'bindingFace': {'file': 'uv_binding_face.bin', 'dtype': 'int32', 'shape': [M]},
               'faceBary': {'file': 'uv_face_bary.bin', 'dtype': 'float32', 'shape': [M, 3]}},
    }

    if args.rig == 'head':
        # ----- fixed-body + neutral base mesh; constant per-vertex frame is baked -----
        res = eval_ehm(ubody, src, fc_neutral, device, tanfov, image_size)
        base_verts = res['vertices'][0]                                 # (N,3) posed, neutral
        vertex_frame = quat_xyzw_to_wxyz(rotmat_to_unitquat(res['ver_transform_mat'][0, :, :3, :3]))
        joints = res.get('joints_transform')
        joints = joints[0] if joints is not None else res['joints'][0]
        bone_pos = joints[BONE_JOINT].cpu().numpy().astype(np.float32)
        joints_idx, joints_w = build_skin_weights(ubody.smplx.lbs_weights)
        bone_parent, bone_names = BONE_PARENT, [b[0] for b in BONES]
        if args.arkit_bs:
            if tracked_frames:
                morphs, morph_names = morphs_refit_arkit(
                    ubody, src, device, tanfov, image_size, args.arkit_bs,
                    tracked_frames, head_scale, N,
                    reg=args.refit_reg, max_frames=args.refit_max_frames, max_gain=args.refit_max_gain)
            else:
                morphs, morph_names = morphs_from_arkit_bs(ubody, args.arkit_bs, head_scale, N, device)
        else:
            morphs, morph_names = morphs_from_flame(ubody, src, base_verts, device, tanfov, image_size)
        vertex_frame.cpu().numpy().astype(np.float32).tofile(os.path.join(args.out, 'vertex_frame.bin'))
        binding['vertexFrame'] = {'file': 'vertex_frame.bin', 'dtype': 'float32', 'shape': [N, 4],
                                  'note': 'constant per-vertex transform quat (wxyz); body is fixed'}
    else:
        # ----- canonical rest base mesh + full SMPL-X skeleton + motion clips -----
        sc_canon = canonical_smplx_coeffs(src['smplx_coeffs'])
        res = eval_ehm(ubody, src, fc_neutral, device, tanfov, image_size, smplx_coeffs=sc_canon)
        base_verts = res['vertices'][0]                                 # (N,3) shaped rest pose
        bone_pos, bone_parent, bone_names = build_body_skeleton(ubody, base_verts)
        joints_idx, joints_w = build_body_skin_weights(ubody.smplx.lbs_weights)
        if args.arkit_bs:
            if tracked_frames:                                         # learned basis, refit at canonical rest
                morphs, morph_names = morphs_refit_arkit(
                    ubody, src, device, tanfov, image_size, args.arkit_bs,
                    tracked_frames, head_scale, N,
                    reg=args.refit_reg, max_frames=args.refit_max_frames, max_gain=args.refit_max_gain,
                    smplx_coeffs=sc_canon)
            else:                                                      # generic basis (template-space, pose-independent)
                morphs, morph_names = morphs_from_arkit_bs(ubody, args.arkit_bs, head_scale, N, device)
        else:                                                          # diff at canonical rest (pre-skin)
            morphs, morph_names = morphs_from_flame(ubody, src, base_verts, device, tanfov, image_size,
                                                    smplx_coeffs=sc_canon)
        motions = export_motions(args.out, src, args.motions)
        with open(os.path.join(args.out, 'motions.json'), 'w') as f:
            json.dump(motions, f, indent=2)
        binding['numBones'] = len(bone_names)
        binding['boneOrder'] = bone_names
        binding['skinInfluences'] = int(joints_idx.shape[1])
        binding['vertexFrame'] = {'note': 'not baked; derive from the per-vertex skin matrix each frame'}
        binding['motions'] = 'motions.json'

    bbox_min = base_verts.min(0).values.cpu().numpy()
    bbox_max = base_verts.max(0).values.cpu().numpy()
    print(f'[export] rig={args.rig}  verts={N} faces={faces.shape[0]} bones={len(bone_names)} '
          f'morphs={len(morph_names)}  bbox={bbox_min.round(3)}..{bbox_max.round(3)}')
    if args.arkit_bs:
        src_note = f'refit on {len(tracked_frames)} frames' if tracked_frames else 'generic LAM basis'
        print(f'[export] morphs = 52 ARKit blendshapes ({src_note})')
    else:
        print('[export] WARNING: no --arkit_bs given -> morphs are RAW FLAME bases '
              f'{morph_names}, NOT the 52 ARKit names. The viewer must drive these from FLAME '
              'coefficients; re-run with --arkit_bs flame_arkit_bs.npy for an ARKit-named bundle.')

    # ----- write shared bundle -----
    write_skin_glb(os.path.join(args.out, 'ehm_skin.glb'),
                   base_verts.cpu().numpy().astype(np.float32), faces, joints_idx, joints_w,
                   morphs, morph_names, bone_pos, bone_parent, bone_names)
    cols14.tofile(os.path.join(args.out, 'gaussians.bin'))
    off = (ubody._smplx_offset[0].detach().cpu().numpy() if ubody._smplx_offset is not None
           else np.zeros((N, 3), np.float32))
    off.astype(np.float32).tofile(os.path.join(args.out, 'static_offset.bin'))
    binding_face.astype(np.int32).tofile(os.path.join(args.out, 'uv_binding_face.bin'))
    face_bary.astype(np.float32).tofile(os.path.join(args.out, 'uv_face_bary.bin'))
    with open(os.path.join(args.out, 'vertex_order.json'), 'w') as f:
        json.dump(list(range(N)), f)            # identity: GLB verts are in smplx.v_template order
    binding['morphTargets'] = morph_names
    with open(os.path.join(args.out, 'binding.json'), 'w') as f:
        json.dump(binding, f, indent=2)

    # ----- round-trip check vs GS_canonical.ply -----
    recon = reconstruct_canonical_positions(ubody, cols14, N, binding_face, face_bary).cpu().numpy()
    if args.gs_canonical and os.path.exists(args.gs_canonical):
        ref = load_ply_xyz(args.gs_canonical)
        if ref.shape[0] == recon.shape[0]:
            err = np.linalg.norm(recon - ref, axis=1)
            print(f'[export] round-trip vs GS_canonical.ply: max={err.max():.3e} mean={err.mean():.3e}')
        else:
            print(f'[export] cannot compare: ply has {ref.shape[0]} pts, bundle has {recon.shape[0]}')
    else:
        # self-check against the model's own canonical reconstruction
        ref = torch.cat([ubody._smplx_xyz[0], ubody._uv_xyz_cano[0]], 0).detach().cpu().numpy()
        err = np.linalg.norm(recon - ref, axis=1)
        print(f'[export] round-trip vs model canonical: max={err.max():.3e} mean={err.mean():.3e}')

    # ----- body rig: quantify the posedirs gap on a representative posed frame -----
    if args.rig == 'body':
        if args.motions:
            frame_smplx = read_tracked_frames(args.motions[0])[0][0]['smplx_coeffs']
            tag = os.path.basename(args.motions[0].rstrip('/')) + ' frame0'
        else:
            frame_smplx = src['smplx_coeffs']
            tag = 'captured pose'
        mx, mn = verify_posed_frame(ubody, src, base_verts, cols14, N, binding_face, face_bary,
                                    frame_smplx, device, tanfov, image_size)
        print(f'[export] posed-frame check ({tag}) skinning-vs-GUAVA (posedirs gap): '
              f'max={mx:.3e} mean={mn:.3e}')

    print(f'[export] wrote bundle to {args.out}')


if __name__ == '__main__':
    main()
