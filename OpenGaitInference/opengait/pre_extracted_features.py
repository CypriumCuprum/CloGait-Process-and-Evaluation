import os
import argparse
import pickle
import torch
import numpy as np
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, get_msg_mgr
from data.transform import BaseSilCuttingTransform
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

def parse_args():
    parser = argparse.ArgumentParser(description='Extract and save embeddings of every CloGait sequence for one model.')
    parser.add_argument('--cfgs', type=str, default='config/default.yaml', 
                        help="path of config file")
    parser.add_argument('--gallery_dir', type=str, default='./output/pre_extracted_features',
                        help="Directory where the extracted features (.npy) are saved")

    return parser.parse_args()

def setup_single_process_ddp():
    """
    Fake môi trường DDP để có thể chạy script trên 1 GPU bằng lệnh python thông thường.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    
    torch.distributed.init_process_group('nccl', init_method='env://')
    torch.cuda.set_device(0)


def main():
    opt = parse_args()
    
    setup_single_process_ddp()
    init_seeds(0)
    
    cfgs = config_loader(opt.cfgs)
    # Naming the pre-extracted features file based on the config file name
    if not os.path.exists(opt.gallery_dir):
        os.makedirs(opt.gallery_dir, exist_ok=True)
    preextracted_features_file = os.path.join(opt.gallery_dir, os.path.basename(opt.cfgs).replace('.yaml', '.npy'))

    msg_mgr = get_msg_mgr()
    msg_mgr.init_logger('output/inference_logs', log_to_file=False)

    model_cfg = cfgs['model_cfg']

    if not os.path.exists(preextracted_features_file):
        # Do code for create gallery features if not exist
        Model = getattr(models, model_cfg['model'])    
        msg_mgr.log_info(f"Initializing {model_cfg['model']} for inference...")
        model = Model(cfgs, False)
        model = get_ddp_module(model, cfgs['trainer_cfg'].get('find_unused_parameters', False))
        model.eval()
        msg_mgr.log_info(f"Creating gallery features at {preextracted_features_file}...")
        gallery_feat = Model.get_feats(model)
        np.save(preextracted_features_file, gallery_feat)
        msg_mgr.log_info(f"Gallery features saved at {preextracted_features_file}")
    else:
        msg_mgr.log_info(f"Gallery is already created at {preextracted_features_file}. Skipping creation.")
    
    

if __name__ == '__main__':
    main()