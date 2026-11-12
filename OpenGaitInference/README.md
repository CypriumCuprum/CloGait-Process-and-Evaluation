# CloGait: baseline evaluation code

Code to reproduce the gait-recognition baselines reported in the **CloGait** paper
(Scientific Data): five models pretrained on Gait3D / Gait3D-Parsing and evaluated, **without
fine-tuning**, on CloGait with the *Casual* condition as gallery and each other clothing condition
(Bag, Coat, Ao Dai, Long Dress) as probe.

| Model | Input modality | Pretrained on | Config |
|---|---|---|---|
| GaitBase | silhouette | Gait3D | `configs/clogait/gaitbase_gait3d.yaml` |
| DeepGaitV2 | silhouette | Gait3D | `configs/clogait/deepgaitv2_gait3d.yaml` |
| SwinGait | silhouette | Gait3D | `configs/clogait/swingait_gait3d.yaml` |
| SMPLGait | silhouette + SMPL | Gait3D | `configs/clogait/smplgait_gait3d.yaml` |
| ParsingGait | parsing map | Gait3D-Parsing | `configs/clogait/parsinggait_gait3d_parsing.yaml` |

This is a trimmed fork of [OpenGait](https://github.com/ShiqiYu/OpenGait) that keeps only what is
needed for this evaluation. The CloGait data itself is hosted on Zenodo (add DOI here). The pipeline
that produced the silhouettes / parsing / pose / SMPL from the raw videos is in
[`../data_processor`](../data_processor).

## 1. Installation

```bash
conda env create -f environment.yml
conda activate clogait-eval
```

A CUDA GPU is required (embedding extraction and the distance computation run on GPU).

## 2. Data

Download CloGait and unpack it. You need three modalities:

```
CloGait/
  silhouettes/   <person_id>/<clothing>/<view>/frameXXXXX.png
  human_parsing/ <person_id>/<clothing>/<view>/frameXXXXX.png
  3d_smpl/       <person_id>/<clothing>/<view>/mesh.pkl
```

`person_id` is `person_00000` ... `person_00193`; clothing is `Casual`, `AoDai`, `Bag`, `Coat`,
`Longdress`; view is `View1` ... `View5`. Convert to the pickle format that OpenGait reads
(run from the repository root, `-n` sets the number of workers):

```bash
# silhouettes -> data/silhouettes_pkl/<id>/<clothing>/<view>/<view>.pkl
python datasets/CloGait/pretreatment.py -i CloGait/silhouettes -o data/silhouettes_pkl -n 8

# parsing maps -> data/parsing_pkl/...
python datasets/CloGait/pretreatment_gps_for_parsing_gait.py -i CloGait/human_parsing -o data/parsing_pkl -p -n 8

# silhouette + SMPL pairs for SMPLGait -> data/smplgait_pkl/<id>/<clothing>/<view>/{sils,smpls}-<view>.pkl
python datasets/CloGait/prepare_smplgait.py --sil_root CloGait/silhouettes --smpl_root CloGait/3d_smpl \
    --output_path data/smplgait_pkl
```

## 3. Pretrained checkpoints

The models are used as released by their authors (OpenGait model zoo / Gait3D-Parsing). Download each
checkpoint and place it at the path OpenGait expects, `output/<dataset>/<model>/<save_name>/checkpoints/<save_name>-<iter>.pt`:

| Model | Place the file at |
|---|---|
| GaitBase | `output/Gait3D/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-60000.pt` |
| DeepGaitV2 | `output/Gait3D/DeepGaitV2/DeepGaitV2/checkpoints/DeepGaitV2-60000.pt` |
| SwinGait | `output/Gait3D/SwinGait/SwinGait3D_B1442_C2/checkpoints/SwinGait3D_B1442_C2-60000.pt` |
| SMPLGait | `output/Gait3D/SMPLGait/SMPLGait/checkpoints/SMPLGait-180000.pt` |
| ParsingGait | `output/Gait3D-Parsing/ParsingGait/ParsingGait/checkpoints/ParsingGait-120000.pt` |

The iteration number is `evaluator_cfg.restore_hint` in the corresponding config; rename the file or edit the config if
your checkpoint has a different name.

## 4. Run the evaluation

```bash
bash extract_features.sh        # embeddings of all 4,850 sequences -> output/pre_extracted_features/*.npy
bash evaluate.sh                # paper protocol (AllViews) -> output/results_AllViews.csv
```

Other protocols supported by `evaluate.sh`:

```bash
bash evaluate.sh View1          # gallery and probe restricted to one view (View1 ... View5)
bash evaluate.sh CrossView      # gallery view i vs probe view j, all clothing (20 ordered pairs)
```

The result CSV has one row per (model, protocol): Rank-1, Rank-5, Rank-10, mAP, mINP.

### Protocol files

`datasets/CloGait/partitions/<protocol>/G_<gallery>__P_<probe>.json` lists the gallery and probe
sequences as `<person_id>-<clothing>-<view>`. In the paper protocol (`AllViews`) the gallery is the 970 Casual sequences
(194 subjects x 5 views) and each probe set is the 970 sequences of one other clothing condition. They can be
regenerated with `datasets/CloGait/create_partition_gallery_probe.py`. The fixed train/val/test subject split
(116/20/58) for users who want to train on CloGait is distributed with the dataset.

### Notes on reproducing the paper numbers

- **Gallery.** The released dataset contains one `Casual` recording per subject and view, and the public
  partitions use it in full as the gallery. The numbers you obtain should match the paper's trends, but small
  deviations from the published tables can occur depending on the checkpoint files and library versions used.
- **Pretrained models only.** No model is trained or fine-tuned on CloGait; the checkpoints must be
  the ones released with Gait3D / Gait3D-Parsing (see section 3).
- **Anonymisation.** Subjects appear only as `person_00000` ... `person_00193`; the gallery/probe lists and
  all scripts in this repository use these identifiers.

## Repository layout

```
opengait/                      OpenGait core (data loading, models, evaluation) trimmed to the 5 models above
  evaluate_gallery_probe.py      gallery/probe evaluation from saved embeddings
  pre_extracted_features.py      embedding extraction
configs/clogait/               one config per model
datasets/CloGait/              pretreatment scripts and gallery/probe partitions
extract_features.sh, evaluate.sh
```

## Acknowledgements

This code is built on **[OpenGait](https://github.com/ShiqiYu/OpenGait)** by the Shiqi Yu Group (SUSTech) and
collaborators. We thank the authors for releasing their framework and pretrained models, and
the teams behind the baselines we evaluate: GaitBase, DeepGaitV2 and SwinGait (OpenGait),
[Gait3D](https://gait3d.github.io/) (SMPLGait) and
[Gait3D-Parsing](https://github.com/Gait3D/Gait3D-Parsing) (ParsingGait). The silhouette pretreatment
follows [GaitSet](https://github.com/AbnerHqC/GaitSet).

## License

OpenGait is released for **academic use only**; this fork inherits that restriction and the original
license terms. Copy the `LICENSE` file from the [upstream repository](https://github.com/ShiqiYu/OpenGait)
into this repository before publishing.
