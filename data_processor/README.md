# CloGait Data Processor

Preprocessing pipeline that turns raw multi-view gait videos (`.mkv`) into the
inputs used by the gait-recognition models in our paper: person-cropped frames,
binary silhouettes, human-parsing maps (and upper/lower-body silhouettes) and
SMPL parameters. Output sequences are 64x64, in the layout expected by
[OpenGait](https://github.com/ShiqiYu/OpenGait).

> Companion code for the paper *CloGait* (Scientific Data). Add the paper
> citation / dataset DOI here.

## Pipeline

```
original_mkv/<subject>/<clothing>/<subject>_..._Goc<N>.mkv
        |
        |-- convert_mkv_to_mp4.py ............ (optional) .mkv -> .mp4
        v
convert_mkv_to_personframe.py   YOLOv8 person crop -> person_frames_png/<subject>/<clothing>/View<N>/frameXXXXX.png
        |
        |-- RobustVideoMatting/convert_mk4_to_silhouettes_seq.py   -> silhouettes 64x64 (png)
        |-- CDGNet_Parsing/run_inference_gaitdata.sh               -> human parsing maps (12 classes)
        |       `-- CDGNet_Parsing/run_parsing_to_silhouette.sh    -> upper / lower / full-body silhouettes
        |-- ROMP/extract_smpl.py                                   -> per-frame SMPL (.npz)
        `-- POSE/run_extract_coco17.sh                             -> COCO-17 keypoints (RTMPose)
```

Single-video helpers (video in, 64x64 sequence out, no intermediate files):
`onevideo_to_sil64x64pkl.py` (RVM) and `CDGNet_Parsing/onevideo_to_parsing64x64pkl.py` (parsing).

## Installation

```bash
conda env create -f environment.yml
conda activate gait-data-processor
```

`environment.yml` installs PyTorch 2.2 (CUDA 12.1), ffmpeg and `requirements.txt`.
For a CPU-only machine delete the `pytorch-cuda` line. If `simple-romp` fails to
build, make sure `Cython` and `numpy==1.23.5` are installed first
(`pip install Cython numpy==1.23.5 && pip install simple-romp`).

### Model weights (not included in the repo)

| Model | File | Put it at | Source |
|---|---|---|---|
| RobustVideoMatting (ResNet50) | `rvm_resnet50.pth` | `RobustVideoMatting/rvm_resnet50.pth` | [release v1.0.0](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth) |
| CDGNet human parsing | `model_best.pth` | `CDGNet_Parsing/model_best.pth` | [release v0.1](https://github.com/Gait3D/CDGNet-Parsing/releases/download/v0.1/model_best.pth) |
| YOLOv8n | `yolov8n.pt` | auto-downloaded by `ultralytics` | - |
| ROMP | - | auto-downloaded by `simple-romp` on first run | - |

## Usage

All batch scripts read their data location from `GAIT_DATA_ROOT`. Set it to an **absolute path**
(the scripts are run from different sub-folders, so the relative fallback is unreliable):

```bash
export GAIT_DATA_ROOT=/path/to/dataset     # or: cp .env.example .env && source .env
```

Expected input: `$GAIT_DATA_ROOT/original_mkv/<subject>/<clothing>/*Goc<N>.mkv`.

```bash
# 1. Crop the person from every frame
python convert_mkv_to_personframe.py

# 2a. Silhouettes (RobustVideoMatting)
cd RobustVideoMatting && python convert_mk4_to_silhouettes_seq.py && cd ..

# 2b. Human parsing (CDGNet) + body-part silhouettes
cd CDGNet_Parsing
bash run_inference_gaitdata.sh
#   edit INPUT_PATH / PART in run_parsing_to_silhouette.sh, then:
bash run_parsing_to_silhouette.sh
cd ..

# 2c. SMPL (ROMP)
python ROMP/extract_smpl.py

# 2d. COCO-17 pose (RTMPose)
cd POSE && bash run_extract_coco17.sh && cd ..

# Single video -> 64x64 array
python onevideo_to_sil64x64pkl.py            # importable: extract_gait_frames(video_path)
python CDGNet_Parsing/onevideo_to_parsing64x64pkl.py --video_path VIDEO.mkv --gpu 0
```

Optional: `python convert_mkv_to_mp4.py` converts `.mkv` to `.mp4` (needs `ffmpeg`).

## Repository layout

```
convert_mkv_to_mp4.py          mkv -> mp4
convert_mkv_to_personframe.py  YOLOv8 person detection + crop
onevideo_to_sil64x64pkl.py     video -> 64x64 silhouettes (RVM)
RobustVideoMatting/            RVM code (third party, GPL-3.0) + batch silhouette script
CDGNet_Parsing/                CDGNet code (third party) + parsing / body-part silhouette scripts
ROMP/                          SMPL extraction wrapper around simple-romp
POSE/                          COCO-17 keypoint extraction (rtmlib / RTMPose)
environment.yml, requirements.txt, .env.example
```

## Notes

- Versions in `environment.yml` were chosen for compatibility, not exhaustively
  tested on a fresh machine. The original CDGNet release targets Python 3.7 /
  PyTorch 1.9; if you hit incompatibilities, use that environment for the CDGNet step only.
- Hierarchical decoding is enabled in `run_inference_gaitdata.sh` (`HIERARCHICAL='yes'`).

## Third-party code and licenses

- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) - GPL-3.0 (`RobustVideoMatting/LICENSE`)
- [CDGNet](https://github.com/tjpulkl/CDGNet) / [CDGNet-Parsing](https://github.com/Gait3D/CDGNet-Parsing)
- [ROMP / simple-romp](https://github.com/Arthur151/ROMP)
- [rtmlib / RTMPose](https://github.com/Tau-J/rtmlib)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - AGPL-3.0

Because RobustVideoMatting is GPL-3.0, choose a license for this repository
that is compatible with it (e.g. GPL-3.0) and add a top-level `LICENSE` file.

## Citation

```bibtex
@article{clogait,
  title   = {TODO},
  author  = {TODO},
  journal = {Scientific Data},
  year    = {TODO}
}
```
