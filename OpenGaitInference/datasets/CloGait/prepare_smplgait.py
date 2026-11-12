"""Build the input folder of SMPLGait (silhouette + SMPL parameters per sequence).

SMPLGait reads, for every <person>/<clothing>/<view>/, two aligned pickles:
    sils-<view>.pkl   (T, 64, 64) uint8   silhouettes, same pre-processing as pretreatment.py
    smpls-<view>.pkl  (T, 85)     float   [camera(3), pose(72), shape(10)] from 3d_smpl/.../mesh.pkl

Frames that pretreatment.py rejects (empty silhouette) are removed from both
streams, so the two stay frame-aligned.

    python prepare_smplgait.py --sil_root CloGait/silhouettes --smpl_root CloGait/3d_smpl \
                               --output_path data/smplgait_pkl
"""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pretreatment import pretreat_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sil_root', required=True, help='CloGait/silhouettes (PNG)')
    ap.add_argument('--smpl_root', required=True, help='CloGait/3d_smpl (mesh.pkl)')
    ap.add_argument('--output_path', required=True)
    ap.add_argument('--img_size', type=int, default=64)
    args = ap.parse_args()

    sil_root, smpl_root, out_root = Path(args.sil_root), Path(args.smpl_root), Path(args.output_path)
    seq_dirs = sorted(d for d in sil_root.glob('*/*/*') if d.is_dir())
    for sil_dir in tqdm(seq_dirs, desc='Sequences'):
        pid, cloth, view = sil_dir.relative_to(sil_root).parts
        smpl = pickle.load(open(smpl_root / pid / cloth / view / 'mesh.pkl', 'rb'))
        pngs = sorted(sil_dir.glob('*.png'))
        if len(pngs) != len(smpl):
            print(f'[skip] {pid}/{cloth}/{view}: {len(pngs)} silhouettes vs {len(smpl)} SMPL rows')
            continue
        sils, keep = [], []
        for i, png in enumerate(pngs):
            frame, _ = pretreat_frame(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE), args.img_size)
            if frame is not None:
                sils.append(frame)
                keep.append(i)
        out_dir = out_root / pid / cloth / view
        out_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(np.asarray(sils), open(out_dir / f'sils-{view}.pkl', 'wb'))
        pickle.dump(np.asarray(smpl)[keep], open(out_dir / f'smpls-{view}.pkl', 'wb'))


if __name__ == '__main__':
    main()
