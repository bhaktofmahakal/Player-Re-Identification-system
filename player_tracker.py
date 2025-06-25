import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

class PlayerFeatureExtractor:
    """Extract visual features for player re-identification"""
    
    def __init__(self):
        self.hist_bins = 32  # for CPU efficiency
        
    def extract_color_histogram(self, image_patch):
        """Extract robust HSV color histogram from player patch"""
        try:
            if image_patch.size == 0 or image_patch.shape[0] < 5 or image_patch.shape[1] < 5:
               
                return np.ones(self.hist_bins * 3) / (self.hist_bins * 3)
            
           
            hsv = cv2.cvtColor(image_patch, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            h_hist = cv2.calcHist([hsv], [0], None, [self.hist_bins], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [self.hist_bins], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [self.hist_bins], [0, 256])
            
            # smoothing to avoid zero divisions
            h_hist = (h_hist.flatten() + 1e-3) / (h_hist.sum() + 1e-2)
            s_hist = (s_hist.flatten() + 1e-3) / (s_hist.sum() + 1e-2)
            v_hist = (v_hist.flatten() + 1e-3) / (v_hist.sum() + 1e-2)
            
            features = np.concatenate([h_hist, s_hist, v_hist])
            
            # Validate features
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return np.ones(self.hist_bins * 3) / (self.hist_bins * 3)
            
            return features
            
        except Exception as e:
            print(f"Color histogram error: {e}")
            return np.ones(self.hist_bins * 3) / (self.hist_bins * 3)
    
    def extract_features(self, frame, bbox):
        """Extract comprehensive features from player bounding box"""

        x1, y1, x2, y2 = map(int, bbox)
        
        # ensuring bbox is within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.hist_bins * 3 + 2)  # +2 for aspect ratio and area
        
       
        player_patch = frame[y1:y2, x1:x2]
        
        color_features = self.extract_color_histogram(player_patch)
        
        width, height = x2 - x1, y2 - y1
        aspect_ratio = width / (height + 1e-7)
        area = width * height / (w * h)  
        
        geometric_features = np.array([aspect_ratio, area])
        
        return np.concatenate([color_features, geometric_features])

class PlayerTracker:
    """Main player tracking and re-identification system"""
    
    def __init__(self, model_path, max_disappeared=30, max_distance=0.5):
        self.model = YOLO(model_path)
        self.feature_extractor = PlayerFeatureExtractor()
        
        # Permanent ID Registry with Motion Tracking
        self.next_id = 1
        self.id_registry = {}  # PERMANENT storage
        self.candidate_tracks = {}  # Temporary storage for new detections
        
        # Fixed color 
        self.fixed_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 0),    # Dark Green
            (255, 192, 203) # Pink
        ]
        self.color_map = {}  
        
        #
        self.position_threshold = 100   
        self.match_threshold = 0.3      
        
        # Motion tracking 
        self.velocity_history = {}      
        self.max_velocity_history = 5   
        
        # Special handling for fast-moving players 
        self.fast_mover_threshold = 50  
        self.referee_bonus = 0.2        
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.players_per_frame = []
        self.id_switches = 0
        
        # JSON tracking data
        self.tracking_data = {}  
        
    def get_player_color(self, player_id):
        """Get fixed color for player ID - NEVER changes once assigned to prevent flickering"""

        if player_id not in self.color_map:
            
            color_idx = len(self.color_map) % len(self.fixed_colors)
            self.color_map[player_id] = self.fixed_colors[color_idx]
        return self.color_map[player_id]
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IOU) between two bounding boxes"""

        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / (union + 1e-7)
        except:
            return 0.0
    
    def calculate_registry_similarity(self, detection, registry_entry):
        """SIMPLE similarity calculation for reliable ID consistency"""

        try:
            
            color_sim = 0.0
            if detection['features'].shape == registry_entry['features'].shape:
                dot_product = np.dot(detection['features'], registry_entry['features'])
                norm_det = np.linalg.norm(detection['features'])
                norm_reg = np.linalg.norm(registry_entry['features'])
                if norm_det > 0 and norm_reg > 0:
                    color_sim = dot_product / (norm_det * norm_reg)
                    color_sim = max(0, color_sim)
            
            
            current_center = detection['center']
            last_center = registry_entry['center']
            pos_dist = np.sqrt((current_center[0] - last_center[0])**2 + 
                              (current_center[1] - last_center[1])**2)
            
            # Adaptive position for crowded areas 
            adaptive_threshold = self.position_threshold
            
            #  handling for goal event 
            is_goal_event = self.frame_count >= 275  # Last 4 seconds of 15-second video
            
            if is_goal_event:
                adaptive_threshold = self.position_threshold * 2.5  # for fallen players
                
              
                det_bbox = detection['bbox']
                reg_bbox = registry_entry['bbox']
                
                det_width = det_bbox[2] - det_bbox[0]
                det_height = det_bbox[3] - det_bbox[1]
                reg_width = reg_bbox[2] - reg_bbox[0]
                reg_height = reg_bbox[3] - reg_bbox[1]
                
                det_aspect = det_height / det_width if det_width > 0 else 0
                reg_aspect = reg_height / reg_width if reg_width > 0 else 0
                
                if abs(det_aspect - reg_aspect) > 1.0:  
                    adaptive_threshold = self.position_threshold * 3.0  
            elif pos_dist > self.position_threshold * 0.8:  
                adaptive_threshold = self.position_threshold * 1.5  
            
            pos_sim = max(0, 1.0 - pos_dist / adaptive_threshold)
            
            
            frames_since_seen = self.frame_count - registry_entry['last_seen']
            
            
            if is_goal_event:
                time_decay_frames = 40  
            else:
                time_decay_frames = 20  
                
            time_factor = max(0.2, 1.0 - frames_since_seen / time_decay_frames)
            
            # Simple combined similarity
            total_sim = 0.5 * color_sim + 0.3 * pos_sim + 0.2 * time_factor
            
            return total_sim, color_sim, pos_sim, 0.0
            
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(detections) == 0:
            return detections
        
        boxes = []
        scores = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            boxes.append([x1, y1, x2, y2])
            scores.append(det['confidence'])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.15, iou_threshold)
        
        # filtered detections
        if len(indices) > 0:
            
            if isinstance(indices, np.ndarray):
                indices_list = indices.flatten().tolist()
            elif isinstance(indices, (list, tuple)):
                indices_list = list(indices)
            else:
                indices_list = [int(i) for i in indices]
            return [detections[i] for i in indices_list]
        else:
            return []

    def detect_players(self, frame):
        """Detect players using YOLOv11 model"""
    
        original_shape = frame.shape[:2]
        resized_frame = cv2.resize(frame, (640, 360))

        results = self.model(resized_frame, conf=0.2, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):

                # Get class and confidence
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                
                
                # Class 0 = 'ball' (EXCLUDE), Class 1 = 'goalkeeper' (INCLUDE), 
                # Class 2 = 'player' (INCLUDE), Class 3 = 'referee' (EXCLUDE for focus)
                if cls in [1, 2] and conf > 0.2: 
                   
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    scale_x = original_shape[1] / 640
                    scale_y = original_shape[0] / 360
                    
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    
                    # very minimum size thresholds to detect all distant players 
                    if bbox_width > 8 and bbox_height > 15 and bbox_area > 200:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls
                        })
    
        detections = self._apply_nms(detections, iou_threshold=0.7)
        
        return detections
    
    def calculate_similarity(self, features1, features2, bbox1, bbox2):
        """Calculate similarity between two player features"""

      
        feature_sim = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-7
        )
        
        center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        spatial_dist = np.linalg.norm(center1 - center2)
        
        max_dist = np.sqrt(1280**2 + 720**2)  
        spatial_sim = 1 - (spatial_dist / max_dist)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        size_sim = 1 - abs(area1 - area2) / (area1 + area2 + 1e-7)
        
        combined_sim = 0.6 * feature_sim + 0.25 * spatial_sim + 0.15 * size_sim
        return combined_sim
    
    def update_tracks(self, frame, detections):
        """ULTIMATE RE-ID: Registry-based tracking with permanent IDs"""

        self.frame_count += 1
        self.total_detections += len(detections)
        
        #  Prepare detection data for registry matching 
        current_detections = []
        is_goal_event = self.frame_count >= 275  # Last 4 seconds
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            
            # During goal events, filter small objects 
            if is_goal_event:
                if det['confidence'] < 0.25:
                    continue
                
                if bbox_area < 800 and bbox_height < 40:  # Very small = field object
                    continue
                
                # Skip narrow objects (goalpost parts)
                aspect_ratio = bbox_height / bbox_width if bbox_width > 0 else 0
                if aspect_ratio > 5.0:  
                    continue
            
            features = self.feature_extractor.extract_features(frame, det['bbox'])
            x1, y1, x2, y2 = det['bbox']
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            bbox_area = (x2 - x1) * (y2 - y1)
            
            current_detections.append({
                'bbox': det['bbox'],
                'features': features,
                'confidence': det['confidence'],
                'center': center,
                'bbox_area': bbox_area
            })

        tracks = []
        used_registry_ids = set()
        
        for detection in current_detections:
            best_match_id = None
            best_similarity = 0
            
            for registry_id, registry_entry in self.id_registry.items():
                if registry_id in used_registry_ids:
                    continue  # Already matched
                    
                similarity, color_sim, pos_sim, area_sim = self.calculate_registry_similarity(detection, registry_entry)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = registry_id
            
            if best_match_id is not None and best_similarity > self.match_threshold:
                
                self._update_registry_and_add_track(detection, best_match_id, tracks, used_registry_ids)
            else:
                
                candidate_match_id = self._find_candidate_match(detection)
                
                if candidate_match_id is not None:
                    # Update candidate track
                    self.candidate_tracks[candidate_match_id]['consecutive_frames'] += 1
                    self.candidate_tracks[candidate_match_id]['detection_data'] = detection
                    
                    if self.candidate_tracks[candidate_match_id]['consecutive_frames'] >= 1:
                        # Promote to permanent registry
                        self.id_registry[candidate_match_id] = {
                            'features': detection['features'],
                            'last_seen': self.frame_count,
                            'center': detection['center'],
                            'bbox_area': detection['bbox_area'],
                            'bbox': detection['bbox']
                        }
                        # Remove from candidates
                        del self.candidate_tracks[candidate_match_id]
                        
                        tracks.append({
                            'id': candidate_match_id,
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence']
                        })
                    else:
                        tracks.append({
                            'id': candidate_match_id,
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence']
                        })
                else:
                    candidate_id = self.next_id
                    self.next_id += 1
                    
                    self.candidate_tracks[candidate_id] = {
                        'detection_data': detection,
                        'first_seen': self.frame_count,
                        'consecutive_frames': 1
                    }
                    
                    tracks.append({
                        'id': candidate_id,
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence']
                    })

        
        self.players_per_frame.append(len(tracks))
        
        if self.frame_count <= 5:
            print(f"Frame {self.frame_count}: {len(current_detections)} detections → {len(tracks)} tracks, Registry: {len(self.id_registry)} IDs")
        

        self.tracking_data[self.frame_count] = []
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            self.tracking_data[self.frame_count].append({
                'id': int(track['id']),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(track['confidence'])
            })
        
        self._cleanup_candidate_tracks()
        
        return tracks
    
    def _find_candidate_match(self, detection):
        """Find matching candidate track for detection"""
        best_match_id = None
        best_similarity = 0
        
        for candidate_id, candidate_data in self.candidate_tracks.items():
            candidate_detection = candidate_data['detection_data']
            

            similarity = self._calculate_detection_similarity(detection, candidate_detection)
            
            if similarity > 0.4 and similarity > best_similarity:  # Lower threshold for candidates
                best_similarity = similarity
                best_match_id = candidate_id
        
        return best_match_id
    
    def _calculate_detection_similarity(self, det1, det2):
        """Calculate similarity between two detections"""

        # Position similarity
        center1 = det1['center']
        center2 = det2['center']
        pos_dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        pos_sim = max(0, 1.0 - pos_dist / 100)  
        
        # Size similarity
        area_ratio = min(det1['bbox_area'], det2['bbox_area']) / max(det1['bbox_area'], det2['bbox_area'])
        
        return 0.7 * pos_sim + 0.3 * area_ratio
    
    def _cleanup_candidate_tracks(self):
        """Remove candidate tracks that haven't been seen for too long"""

        to_remove = []
        for candidate_id, candidate_data in self.candidate_tracks.items():
            if self.frame_count - candidate_data['first_seen'] > 10:  # Remove after 10 frames
                to_remove.append(candidate_id)
        
        for candidate_id in to_remove:
            del self.candidate_tracks[candidate_id]
    
    def _update_registry_and_add_track(self, detection, player_id, tracks, used_registry_ids):
        """Helper method to update registry and add track"""
        used_registry_ids.add(player_id)
        
        old_features = self.id_registry[player_id]['features']
 
        stability_count = self.id_registry[player_id].get('stability_count', 0)
        is_goal_event = self.frame_count >= 275  # Last 4 seconds
        
        if is_goal_event and stability_count > 3:  
            smoothing_factor = 0.95  
        elif stability_count > 5:  
            smoothing_factor = 0.9  
        else:
            smoothing_factor = 0.8  
            
        new_features = smoothing_factor * old_features + (1 - smoothing_factor) * detection['features']
        
        # Update stability count
        new_stability_count = stability_count + 1
        
        self.id_registry[player_id].update({
            'features': new_features,
            'last_seen': self.frame_count,
            'center': detection['center'],
            'bbox_area': detection['bbox_area'],
            'stability_count': new_stability_count
        })
        
        tracks.append({
            'id': player_id,
            'bbox': detection['bbox'],
            'confidence': detection['confidence']
        })
    
    def draw_tracks(self, frame, tracks):
        """Draw bounding boxes and IDs on frame"""
        annotated_frame = frame.copy()
        
        for track in tracks:
            player_id = track['id']
            bbox = track['bbox']
            confidence = track['confidence']
            
            x1, y1, x2, y2 = bbox
            color = self.get_player_color(player_id)  # Fixed color per ID
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{player_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_statistics(self):
        """Get tracking statistics from permanent registry"""
        avg_players = np.mean(self.players_per_frame) if self.players_per_frame else 0
        unique_players = len(self.id_registry)  # Total unique IDs in registry
        
        total_assignments = sum(self.players_per_frame)
        new_id_assignments = unique_players  
        reused_assignments = total_assignments - new_id_assignments

        consistency_rate = reused_assignments / total_assignments if total_assignments > 0 else 0
        consistency_rate = max(0, min(1, consistency_rate))  # Clamp between 0-1
 
        
        return {
            'total_frames': self.frame_count,
            'unique_players': unique_players,
            'avg_players_per_frame': avg_players,
            'id_consistency_rate': consistency_rate,
            'total_detections': total_assignments,
            'registry_size': len(self.id_registry),
            'reused_assignments': reused_assignments,
            'new_assignments': new_id_assignments
        }

def main():
    """Main function to run player tracking"""

    model_path = "data/models/yolov11_player_detection.pt"
    video_path = "data/videos/15sec_input_720p.mp4"
    output_path = "output/tracked_video.mp4"
    

    os.makedirs("output", exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None, None
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return None, None
    
    # tracker
    tracker = PlayerTracker(model_path)
    
    # video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None
    
    # Get video props
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
 
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # More compatible method
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
     
        detections = tracker.detect_players(frame)
        
        tracks = tracker.update_tracks(frame, detections)
        
        annotated_frame = tracker.draw_tracks(frame, tracks)

        out.write(annotated_frame)
        
        # Progress update
        if frame_num % 30 == 0 or frame_num == total_frames:
            elapsed = time.time() - start_time
            fps_current = frame_num / elapsed
            print(f"Processed {frame_num}/{total_frames} frames "
                  f"({frame_num/total_frames*100:.1f}%) - "
                  f"FPS: {fps_current:.1f}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Get statistics
    stats = tracker.get_statistics()
   
    import json
    json_path = "output/tracking_data.json"
    with open(json_path, 'w') as f:
        json.dump(tracker.tracking_data, f, indent=2)
    
    print("\n=== TRACKING COMPLETED ===")
    print(f"Output saved to: {output_path}")
    print(f"JSON data saved to: {json_path}")
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Unique players detected: {stats['unique_players']}")
    print(f"Average players per frame: {stats['avg_players_per_frame']:.2f}")
    print(f"ID consistency rate: {stats['id_consistency_rate']:.2%}")
    
    return tracker, stats

if __name__ == "__main__":
    tracker, stats = main()