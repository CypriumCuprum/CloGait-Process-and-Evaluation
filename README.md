# CloGait: data processing and baseline evaluation

Code accompanying the **CloGait** paper (Scientific Data).

| Folder | Purpose |
|---|---|
| [`data_processor/`](data_processor) | Turns raw multi-view videos into the released modalities: person-cropped frames, silhouettes (RobustVideoMatting), human-parsing maps (CDGNet), SMPL (ROMP) and COCO-17 skeletons (RTMPose). |
| [`OpenGaitInference/`](OpenGaitInference) | Reproduces the baseline recognition results (GaitBase, DeepGaitV2, SwinGait, SMPLGait, ParsingGait) on the released CloGait data. Built on [OpenGait](https://github.com/ShiqiYu/OpenGait). |

Each folder has its own README, environment file and instructions. The two parts are independent: to
reproduce the paper's results you only need `OpenGaitInference` and the released dataset (Zenodo, add DOI here).

## Acknowledgements

We thank the authors of [OpenGait](https://github.com/ShiqiYu/OpenGait),
[RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting),
[CDGNet](https://github.com/tjpulkl/CDGNet), [ROMP](https://github.com/Arthur151/ROMP),
[rtmlib](https://github.com/Tau-J/rtmlib), [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics),
[Gait3D](https://gait3d.github.io/) and [GaitSet](https://github.com/AbnerHqC/GaitSet), whose source code
and models this work builds on. See each sub-folder for the specific licenses.

## License

Third-party components carry their own licenses (e.g. OpenGait: academic use only, RobustVideoMatting: GPL-3.0,
Ultralytics: AGPL-3.0). Choose a license for this repository that is compatible with them and add a `LICENSE` file.
