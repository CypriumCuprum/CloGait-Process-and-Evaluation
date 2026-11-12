import cv2
import os
import sys
import torch
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from pathlib import Path
from typing import Optional

# 1. Lấy đường dẫn tuyệt đối của thư mục chứa file script hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Tạo đường dẫn trỏ tới thư mục RobustVideoMatting
rvm_path = os.path.join(current_dir, 'RobustVideoMatting')

# 3. Thêm vào sys.path để Python có thể tìm thấy các module bên trong đó
if rvm_path not in sys.path:
    sys.path.append(rvm_path)

try:
    from model import MattingNetwork
except ImportError:
    print("Import Error: MattingNetwork class not found.")

# Khai báo 2 biến để lưu trữ model trên RAM/VRAM, tránh load lại nhiều lần
_RVM_MODEL = None
_YOLO_MODEL = None
_DEVICE = None

def auto_downsample_ratio(h, w):
    """
    Tự động tính toán tỷ lệ downsample để cạnh dài nhất không vượt quá 512px.
    (Giữ nguyên chất lượng cho ảnh nhỏ)
    """
    return min(512 / max(h, w), 1.0)

def crop_person_yolo(frame: np.ndarray, model: YOLO, padding_percent: float = 0.1):
    results = model.predict(frame, classes=[0], verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None

    max_area = 0
    best_box = None
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    h, w = frame.shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = int(box_w * padding_percent), int(box_h * padding_percent)

    x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    x2, y2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))

    cropped = frame[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else None

def crop_silhouette(
        sil: NDArray[np.uint8], 
        resize1: int = 128, 
        resize2: int = 64, 
        pixel_threshold: int = 800
    ) -> NDArray[np.uint8]:
    
    y = sil.sum(axis=1)
    crop_y1 = (y > pixel_threshold).argmax(axis=0)
    crop_y2 = (y > pixel_threshold).cumsum(axis=0).argmax(axis=0)
    sil = sil[crop_y1:crop_y2 + 1, :]

    ratio = sil.shape[1] / sil.shape[0]
    sil = cv2.resize(sil, (int(resize1 * ratio), resize1), interpolation=cv2.INTER_CUBIC)

    x_cumsum = sil.sum(axis=0).cumsum()
    x_center = np.searchsorted(x_cumsum, sil.sum() / 2)

    half_width = resize1 // 2
    crop_x1 = x_center - half_width
    crop_x2 = x_center + half_width

    if crop_x1 <= 0 or crop_x2 >= sil.shape[1]:
        crop_x1 += half_width
        crop_x2 += half_width
        padding = np.zeros((sil.shape[0], half_width))
        sil = np.concatenate([padding, sil, padding], axis=1)

    sil = sil[:, crop_x1:crop_x2].astype(np.uint8)
    ratio = sil.shape[1] / sil.shape[0]
    sil = cv2.resize(sil, (int(resize2 * ratio), resize2), interpolation=cv2.INTER_CUBIC)
    return sil

def pretreat_single_frame(img: np.ndarray, img_size: int = 64, dataset: str = 'CASIAB') -> Optional[np.ndarray]:
    """Tiền xử lý cho duy nhất 1 ảnh silhouette (Gait frame).

    Args:
        img (np.ndarray): Ảnh đầu vào dạng numpy array (2D grayscale hoặc 3D BGR).
        img_size (int, optional): Kích thước output chuẩn hóa (H, W). Mặc định là 64.
        dataset (str, optional): Tên dataset format ('CASIAB' hoặc 'GREW'). Mặc định là 'CASIAB'.

    Returns:
        Optional[np.ndarray]: Ảnh silhouette kích thước (img_size, img_size) kiểu uint8,
                              hoặc None nếu ảnh không hợp lệ / không có dữ liệu.
    """
    if img is None:
        return None

    # Đảm bảo ảnh là 2D grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Nhị phân hóa ảnh
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    if dataset == 'GREW':
        return cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC).astype('uint8')

    # 2. Kiểm tra nếu frame rỗng / quá ít thông tin
    if img.sum() <= 10000:
        return None

    # Get the upper and lower points
    y_sum = img.sum(axis=1)
    y_top = (y_sum != 0).argmax(axis=0)
    y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top: y_btm + 1, :]

    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)

    # Get the median of the x-axis and take it as the person's x-center.
    x_csum = img.sum(axis=0).cumsum()
    x_center = None
    for idx, csum in enumerate(x_csum):
        if csum > img.sum() / 2:
            x_center = idx
            break

    if x_center is None:
        return None

    # 6. Căn lề và cắt lấy vùng (img_size, img_size) quanh x_center
    half_width = img_size // 2
    left = x_center - half_width
    right = x_center + half_width

    # Padding hai bên nếu bị tràn viền
    if left <= 0 or right >= img.shape[1]:
        left += half_width
        right += half_width
        pad = np.zeros((img.shape[0], half_width))
        img = np.concatenate([pad, img, pad], axis=1)

    cropped = img[:, left:right]

    # Đảm bảo đúng shape đầu ra
    if cropped.shape != (img_size, img_size):
        return None

    return cropped.astype('uint8')

def extract_gait_frames(
    video_path: str, 
    img_size: int = 64
) -> np.ndarray:
    """
    Hàm self-contained: Tự động load model (1 lần duy nhất) và trích xuất frame từ video.
    """
    global _RVM_MODEL, _DEVICE, _YOLO_MODEL

    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO("yolov8n.pt")

    # 1. Tự động load model vào GPU/CPU nếu chưa có
    if _RVM_MODEL is None:
        print("Loading RVM model for the first time...")
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _RVM_MODEL = MattingNetwork('resnet50').to(_DEVICE).eval()
        weights_path = os.path.join(current_dir, 'RobustVideoMatting', 'rvm_resnet50.pth')
        _RVM_MODEL.load_state_dict(torch.load(weights_path, map_location=_DEVICE))
        print("Model loaded successfully!")

    # 2. Xử lý video
    vidcap = cv2.VideoCapture(str(video_path))
    if not vidcap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")

    valid_frames = []
    with torch.no_grad():
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            
            cropped_frame = crop_person_yolo(frame, _YOLO_MODEL)
            if cropped_frame is None:
                print("No person detected in this frame, skipping...")
                continue

            rec = [None] * 4

            frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            tensor_img = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            tensor_img = tensor_img.to(_DEVICE, non_blocking=True)
            h, w = tensor_img.shape[2:]
            ds_ratio = auto_downsample_ratio(h, w)

            fgr, pha, *rec = _RVM_MODEL(tensor_img, *rec, downsample_ratio=ds_ratio)
            alpha = pha[0, 0].cpu().numpy()
            silhouette_mask = (alpha * 255).astype(np.uint8)

            processed_sil = crop_silhouette(silhouette_mask, pixel_threshold=150)
            processed_pretreat_sil = pretreat_single_frame(processed_sil, img_size=img_size)
            
            if processed_pretreat_sil is not None:
                valid_frames.append(processed_pretreat_sil)
    

    vidcap.release()

    if len(valid_frames) == 0:
        return np.array([])
        
    return np.asarray(valid_frames)