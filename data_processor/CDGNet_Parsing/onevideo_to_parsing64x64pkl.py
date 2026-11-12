import sys
import os
# 1. Lấy đường dẫn tuyệt đối của thư mục chứa file này (chính là thư mục CDGNet_Parsing)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Đẩy nó lên ưu tiên số 1 trong danh sách tìm kiếm
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from networks.CDGNet import Res_Deeplab
from dataset.datasets import InferenceDataSet
from cdgnet_utils.transforms import transform_parsing
from inference import get_lip_palette, valid

import cv2
from pathlib import Path
from ultralytics import YOLO
import tqdm
import torch
import numpy as np
from copy import deepcopy
from PIL import Image as PILImage
import torchvision.transforms as transforms
from torch.utils import data

# Load model một lần
model = YOLO("yolov8n.pt")
parsing_model_weight = os.path.join(CURRENT_DIR, "model_best.pth")
output_dir = os.path.join(CURRENT_DIR, "output_temp")

def extract_and_save_cropped_frames(
    video_path,
    output_dir,
    model=model,
    padding_percent=0.1,
    rotate_ccw=False,
):
    """
    Crop person from a single video and save all cropped frames.

    Args:
        video_path (str): Path to input video.
        output_dir (str): Directory to save cropped frames.
        model (YOLO): Loaded YOLO model.
        padding_percent (float): Padding around detected bbox.
        rotate_ccw (bool): Rotate frame 90° CCW before detection.

    Returns:
        int: Number of cropped frames saved.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_idx = 1
    saved = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if rotate_ccw:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        results = model.predict(frame, classes=[0], verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            frame_idx += 1
            continue

        # Chọn người có bbox lớn nhất
        best_box = None
        max_area = -1

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                best_box = (x1, y1, x2, y2)

        x1, y1, x2, y2 = best_box

        h, w = frame.shape[:2]

        pad_x = int((x2 - x1) * padding_percent)
        pad_y = int((y2 - y1) * padding_percent)

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))

        cropped = frame[y1:y2, x1:x2]

        save_path = output_dir / f"frame{frame_idx:05d}.png"
        cv2.imwrite(str(save_path), cropped)

        saved += 1
        frame_idx += 1

    cap.release()

    return saved

def process_image_folder(input_folder, output_dir, gpu='0', input_size=(256, 256), vis=True, batch_size=1):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    gpus = [int(i) for i in gpu.split(',')]

    save_lbl_dir = os.path.join(output_dir, 'Pred_parsing_results')
    os.makedirs(save_lbl_dir, exist_ok=True)
    
    save_vis_dir = None
    if vis:
        save_vis_dir = os.path.join(output_dir, 'Pred_parsing_results_vis')
        os.makedirs(save_vis_dir, exist_ok=True)

    # 1. Setup Dataset & DataLoader (Truyền thẳng input_folder vào)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Dataset của bạn sẽ tự động gọi os.listdir(input_folder)
    lip_dataset = InferenceDataSet(root=input_folder, dataset='val', crop_size=input_size, transform=transform)
    num_samples = len(lip_dataset)
    
    if num_samples == 0:
        print(f"[!] No images found in {input_folder}")
        return

    valloader = data.DataLoader(lip_dataset, batch_size=batch_size * len(gpus), shuffle=False, pin_memory=True)

    # 2. Load Model
    model = Res_Deeplab(num_classes=12)
    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(parsing_model_weight)

    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    # 3. Chạy Inference
    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, len(gpus))

    # 4. Lưu ảnh
    palette = get_lip_palette()
    for i, im_lbl_name in tqdm.tqdm(enumerate(lip_dataset.files), total=num_samples, desc="Saving Results"):
        image_path = im_lbl_name['img']
        im_name = im_lbl_name['name']
        im_name = os.path.splitext(im_name)[0] + '.png' 
        
        save_lbl_path = os.path.join(save_lbl_dir, im_name)
        os.makedirs(os.path.dirname(save_lbl_path), exist_ok=True)

        img = PILImage.open(image_path)
        w, h = img.size
        
        pred = transform_parsing(parsing_preds[i], centers[i], scales[i], w, h, input_size)
        
        pred_lbl = PILImage.fromarray(pred)
        pred_lbl.save(save_lbl_path)

        if vis and save_vis_dir:
            save_vis_path = os.path.join(save_vis_dir, im_name)
            pred_vis = PILImage.fromarray(pred)
            pred_vis.putpalette(palette)
            pred_vis = pred_vis.convert("RGB")
            pred_vis.save(save_vis_path)


def pretreat_sequence_to_numpy(folder_input: str, img_size: int = 64) -> np.ndarray:
    """
    Preprocess a sequence of parsing label images into a NumPy array.
    Has been hardcoded to only work with parsing images (using INTER_NEAREST).
    
    Args:
        folder_input (str): Path to the folder containing parsing label images (.png).
        img_size (int): Target size for resizing images (default is 64).
        
    Returns:
        np.ndarray: Matrix of shape (num_frames, img_size, img_size) containing preprocessed images.
    """
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_paths = sorted([
        os.path.join(folder_input, f) for f in os.listdir(folder_input) 
        if f.lower().endswith(valid_exts)
    ])
    
    if not img_paths:
        print(f"[!] No images found in directory: {folder_input}")
        return None

    to_pickle = []
    
    for img_file in img_paths:
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # [CỐ ĐỊNH PARSING] - Luôn tạo silhouette mask từ ảnh label
        img_sil = (img > 0).astype('uint8') * 255
            
        if img_sil.sum() <= 10000:
            print(f"[!] Image sum: {img_sil.sum()}")
            print(f"[!] {img_file} has too few valid pixels. Skipping.")
            continue

        # Tìm giới hạn trên và dưới
        y_sum = img_sil.sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
        
        img = img[y_top: y_btm + 1, :]
        img_sil = img_sil[y_top: y_btm + 1, :]

        # Resize giữ nguyên tỷ lệ dựa trên chiều cao
        ratio = img.shape[1] / img.shape[0]
        ratio_sil = img_sil.shape[1] / img_sil.shape[0]
        assert ratio == ratio_sil
        
        # [CỐ ĐỊNH PARSING] - Luôn dùng INTER_NEAREST để không sinh ra class rác
        img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_NEAREST)
        img_sil = cv2.resize(img_sil, (int(img_size * ratio), img_size), interpolation=cv2.INTER_NEAREST)

        # Tìm trung điểm trục X để lấy tâm người
        x_csum = img_sil.sum(axis=0).cumsum()
        x_center = None
        for idx, csum in enumerate(x_csum):
            if csum > img_sil.sum() / 2:
                x_center = idx
                break

        if x_center is None:
            print(f"[!] {img_file} do not have a valid center. Skipping.")
            continue

        # Padding và cắt trái/phải dựa vào tâm x_center
        half_width = img_size // 2
        left = x_center - half_width
        right = x_center + half_width
        
        if left <= 0 or right >= img.shape[1]:
            left += half_width
            right += half_width
            _pad = np.zeros((img.shape[0], half_width), dtype=img.dtype)
            img = np.concatenate([_pad, img, _pad], axis=1)

        cropped_img = img[:, left: right].astype('uint8')
        to_pickle.append(cropped_img)

    if to_pickle:
        seq_data = np.asarray(to_pickle)
        
        if len(seq_data) < 5:
            print(f"[!] Sequence t   ({len(seq_data)} frames) in {folder_input}.")
            
        return seq_data
    
    return None

def convert_onevideo_to_parsing64x64pkl(video_path: str, gpu='0'):
    """
    Convert a video into a sequence of 64x64 parsing images and save as .pkl.
    
    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save the results.
        gpu (str): GPU to use (default is '0').
    """
    
    temp_cropped_dir = os.path.join(output_dir, "temp_cropped_frames")
    os.makedirs(temp_cropped_dir, exist_ok=True)
    
    # 1. Crop người từ video
    print(f"[INFO] Cropping frames from video: {video_path}")
    num_saved = extract_and_save_cropped_frames(video_path, temp_cropped_dir)
    
    if num_saved == 0:
        print(f"[!] No valid frames found in video: {video_path}")
        return
    
    # 2. Chạy parsing trên các frame đã crop
    print(f"[INFO] Running parsing on cropped frames...")
    process_image_folder(temp_cropped_dir, output_dir, gpu=gpu)
    
    # 3. Tiền xử lý các frame parsing thành mảng numpy
    parsing_results_dir = os.path.join(output_dir, 'Pred_parsing_results')
    seq_data = pretreat_sequence_to_numpy(parsing_results_dir, img_size=64)
    
    if seq_data is None:
        print(f"[!] Invalid data to save from video: {video_path}")
    
    # Cleanup temporary cropped frames
    # for f in os.listdir(temp_cropped_dir):
    #     os.remove(os.path.join(temp_cropped_dir, f))
    # os.rmdir(temp_cropped_dir)

    print(f"[INFO] Finished processing video: {video_path}. Saved parsing data to {parsing_results_dir}.")
    return seq_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a video to parsing 64x64 .pkl")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--gpu", type=str, default='0', help="GPU to use (default is '0').")

    parser.add_argument("--reference_pkl", type=str, default=None, help="Optional .pkl to compare the result against.")

    args = parser.parse_args()

    ## Test working of the function
    seq_data = convert_onevideo_to_parsing64x64pkl(args.video_path, gpu=args.gpu)

    if args.reference_pkl:
        import pickle
        with open(args.reference_pkl, 'rb') as f:
            reference_data = pickle.load(f)
        if seq_data is not None and reference_data is not None:
            if np.array_equal(seq_data, reference_data):
                print("[INFO] The generated parsing data matches the reference data.")
            else:
                print("[WARNING] The generated parsing data does NOT match the reference data.")
