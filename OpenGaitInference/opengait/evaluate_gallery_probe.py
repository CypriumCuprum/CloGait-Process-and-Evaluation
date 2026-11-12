import os
import argparse
import pickle
import torch
import numpy as np
from evaluation.evaluator import evaluate_gallery_probe
def parse_args():
    parser = argparse.ArgumentParser(description='Inference for a single video in OpenGait.')
    parser.add_argument('--cfgs', type=str, required=True, 
                        help="path of config gallery probe file")
    parser.add_argument('--embedding_path', type=str, required=True,
                        help="Path to the saved gallery features (e.g., .npy or .pkl)")
    parser.add_argument('--results_path', type=str, required=True,
                        help="Path to save the evaluation results (e.g., .json)")

    return parser.parse_args()

def load_gallery_features(embedding_path):
    gallery_feat = np.load(embedding_path, allow_pickle=True).item()
    return gallery_feat

def main():
    opt = parse_args()

    model_type = os.path.basename(opt.embedding_path).split('.')[0]
    config_type = os.path.basename(opt.cfgs).split('.')[0]

    # ============================================================
    # Check whether this Model_Type + Config_Type already exists
    # ============================================================
    if os.path.exists(opt.results_path):
        with open(opt.results_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:  # skip header
            if not line.strip():
                continue

            columns = line.strip().split(',')

            if len(columns) >= 2:
                existing_model = columns[0]
                existing_config = columns[1]

                if existing_model == model_type and existing_config == config_type:
                    print(
                        f"[SKIP] Result already exists: "
                        f"Model_Type={model_type}, Config_Type={config_type}"
                    )
                    return

    # ============================================================
    # Check embedding file
    # ============================================================
    if not os.path.exists(opt.embedding_path):
        raise FileNotFoundError(
            f"Gallery features file not found: {opt.embedding_path}"
        )

    print(
        f"[RUN] Model_Type={model_type}, "
        f"Config_Type={config_type}"
    )

    # ============================================================
    # Load features and evaluate
    # ============================================================
    embedding_features = load_gallery_features(opt.embedding_path)

    results = evaluate_gallery_probe(
        embedding_features,
        opt.cfgs,
        metric='euc'
    )

    # ============================================================
    # Create result file + header if needed
    # ============================================================
    if not os.path.exists(opt.results_path):
        with open(opt.results_path, 'w') as f:
            f.write(
                "Model_Type,Config_Type,"
                "Rank-1,Rank-5,Rank-10,mAP,mINP\n"
            )

    # ============================================================
    # Append result
    # ============================================================
    with open(opt.results_path, 'a') as f:
        f.write(
            f"{model_type},{config_type},"
            f"{results['Rank-1']:.2f},"
            f"{results['Rank-5']:.2f},"
            f"{results['Rank-10']:.2f},"
            f"{results['mAP']:.2f},"
            f"{results['mINP']:.2f}\n"
        )

    print(
        f"[DONE] Added result: "
        f"Model_Type={model_type}, Config_Type={config_type}"
    )

if __name__ == '__main__':
    main()