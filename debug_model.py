
"""Debug script to check YOLO model classes and test detection"""

import cv2
from ultralytics import YOLO
import numpy as np

def debug_model():
    """Debug YOLO model to understand actual classes"""
    print("DEBUGGING YOLO MODEL")
    print("=" * 50)
    
    model_path = "data/models/yolov11_player_detection.pt" # your best.pt 
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
  
    print(f"\nModel Information:")
    print(f"   Model type: {type(model)}")
    
    # Get class names 
    if hasattr(model, 'names') and model.names:
        print(f"   Classes found: {len(model.names)}")
        for idx, name in model.names.items():
            print(f"   Class {idx}: {name}")
    else:
        print("  No class names found in model")
    
    # provided video 
    video_path = "data/videos/15sec_input_720p.mp4"
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return
            
        ret, frame = cap.read()
        if not ret:
            print("Cannot read first frame")
            cap.release()
            return
            
        print(f"\nVideo Information:")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"   FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"   Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        
     
        print(f"\nTesting Detection on First Frame:")
        results = model(frame, conf=0.1, verbose=False)  
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            print(f"   Total detections: {len(boxes)}")
            
            # Analyze dtections by class
            class_counts = {}
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                class_name = model.names.get(cls, f"class_{cls}") if hasattr(model, 'names') else f"class_{cls}"
                
                if cls not in class_counts:
                    class_counts[cls] = []
                class_counts[cls].append(conf)
                
                if i < 10:  
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    print(f"   Detection {i+1}: Class {cls} ({class_name}) - Conf: {conf:.3f} - Box: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            
            print(f"\nClass Distribution:")
            for cls, confs in class_counts.items():
                class_name = model.names.get(cls, f"class_{cls}") if hasattr(model, 'names') else f"class_{cls}"
                avg_conf = np.mean(confs)
                print(f"   Class {cls} ({class_name}): {len(confs)} detections, avg conf: {avg_conf:.3f}")
        else:
            print("   No detections found")
        
        cap.release()
        
    except Exception as e:
        print(f"Error testing video: {e}")

if __name__ == "__main__":
    debug_model()