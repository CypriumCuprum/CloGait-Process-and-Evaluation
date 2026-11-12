import os
import cv2
import re
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def extract_and_crop_person(input_root, output_root, padding_percent=0.1):
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    mkv_files = list(input_path.rglob('*.mkv'))
    if not mkv_files:
        print("No .mkv files found.")
        return

    # Initialize YOLOv8 model (downloads 'yolov8n.pt' automatically on first run)
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt") 

    for mkv_file in mkv_files:
        relative_path = mkv_file.relative_to(input_path)
        
        match = re.search(r'Goc(\d+)', mkv_file.name, re.IGNORECASE)
        view_folder_name = f"View{match.group(1)}" if match else mkv_file.stem
            
        frames_dir = output_path / relative_path.parent / view_folder_name
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing: {mkv_file.name} -> {frames_dir}")
        
        vidcap = cv2.VideoCapture(str(mkv_file))
        if not vidcap.isOpened():
            print(f"  Error opening video: {mkv_file.name}")
            continue

        count = 1

        while True:
            success, frame = vidcap.read()
            if not success:
                break

            # Predict using YOLO. classes=[0] filters for 'person' only.
            results = model.predict(frame, classes=[0], verbose=False, conf=0.6)
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                # Find the largest bounding box (assuming it is the main subject)
                max_area = 0
                best_box = None
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_box = (x1, y1, x2, y2)
                
                if best_box:
                    x1, y1, x2, y2 = best_box
                    h, w = frame.shape[:2]
                    
                    # Calculate padding based on bounding box size
                    box_w, box_h = x2 - x1, y2 - y1
                    pad_x = int(box_w * padding_percent)
                    pad_y = int(box_h * padding_percent)
                    
                    # Apply padding while ensuring boundaries do not exceed frame dimensions
                    x1 = max(0, int(x1 - pad_x))
                    y1 = max(0, int(y1 - pad_y))
                    x2 = min(w, int(x2 + pad_x))
                    y2 = min(h, int(y2 + pad_y))
                    
                    # Crop the frame
                    cropped_frame = frame[y1:y2, x1:x2]
                    
                    # Save the cropped frame
                    frame_path = frames_dir / f"frame{count:05d}.png"
                    cv2.imwrite(str(frame_path), cropped_frame)
            else:
                # Optional: Handle frames where no person is detected
                pass
            
            count += 1

        vidcap.release()

if __name__ == "__main__":
    INPUT_DIR = os.path.join(os.environ.get("GAIT_DATA_ROOT", "./data"), "original_mkv")
    OUTPUT_DIR = os.path.join(os.environ.get("GAIT_DATA_ROOT", "./data"), "person_frames_png")
    
    extract_and_crop_person(INPUT_DIR, OUTPUT_DIR, padding_percent=0.1)
    print("Extraction and cropping completed.")