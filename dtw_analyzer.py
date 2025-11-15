
# # Pose Detection with FastDTW Comparison - Shoulder Press Analysis
# 
# This notebook uses BlazePose for pose detection and FastDTW (Fast Dynamic Time Warping) for comparing exercise movements between a trainer and user.
# 
# ## Key Features:
# - Extract multi-dimensional features from pose sequences
# - Use FastDTW for temporal alignment and comparison
# - Generate similarity scores and detailed feedback

# ## 1) Installation
# 
# ```bash
# pip install flask flask-cors mediapipe==0.10.14 protobuf==4.25.3 opencv-python numpy fastdtw scipy
# ```

#import sys
#print(sys.executable)

#/Users/elizabethluo/miniconda3/envs/mpenv/bin/python -m pip install flask flask-cors


import cv2
import numpy as np
from collections import deque
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle

MODEL = "blazepose"   
DRAW_SKELETON = True
SCORE_THRESH = 0.5
ACTION = "shoulder_press"

USE_ANGLES = True
USE_POSITIONS = False
USE_VELOCITIES = False
NORMALIZE_FEATURES = True

# ## 2) PoseDetector Class
# 

class PoseDetector:
    def __init__(self, backend="blazepose"):
        backend = backend.lower()
        self.backend = backend
        if backend == "blazepose":
            self._init_blazepose()
        else:
            raise ValueError("backend must be 'blazepose'")

    def _init_blazepose(self):
        from mediapipe.python.solutions import pose as mp_pose
        self._mp_pose = mp_pose
        self.pose = mp_pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True
        )
        self._edges = list(mp_pose.POSE_CONNECTIONS)

    def _infer_blazepose(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)
        if not res.pose_landmarks:
            return []
        pts = []
        for idx, lm in enumerate(res.pose_landmarks.landmark):
            x = float(lm.x) * w
            y = float(lm.y) * h
            score = float(getattr(lm, "visibility", 0.9))
            pts.append((x, y, score))
        return pts

    def infer(self, frame_bgr):
        if self.backend == "blazepose":
            return self._infer_blazepose(frame_bgr)

    def draw(self, frame_bgr, pts, score_thresh=0.8):
        if not pts:
            return frame_bgr
        for (x, y, sc) in pts:
            if sc >= score_thresh:
                cv2.circle(frame_bgr, (int(x), int(y)), 3, (0,255,0), -1)
        if DRAW_SKELETON and self._edges:
            for a, b in self._edges:
                if a < len(pts) and b < len(pts):
                    if pts[a][2] >= score_thresh and pts[b][2] >= score_thresh:
                        ax, ay, _ = pts[a]; bx, by, _ = pts[b]
                        cv2.line(frame_bgr, (int(ax), int(ay)), (int(bx), int(by)), (0,255,0), 1)
        return frame_bgr
# ## 3) Utility Functions# 
def angle_2pts(a, b):
    a = np.array(a[:2], float); b = np.array(b[:2], float)
    delta_y, delta_x = b[1]-a[1], b[0]-a[0]
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def angle_3pts(a, b, c):
    a = np.array(a[:2], float); b = np.array(b[:2], float); c = np.array(c[:2], float)
    ba, bc = a-b, c-b
    denom = (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def normalize_pose_position(pts):
    # Normalize pose positions relative to hip center and torso scale
    if not pts or len(pts) < 33:
        return pts
    
    hip_left = np.array(pts[23][:2])
    hip_right = np.array(pts[24][:2])
    hip_center = (hip_left + hip_right) / 2.0
    
    shoulder_left = np.array(pts[11][:2])
    shoulder_right = np.array(pts[12][:2])
    shoulder_center = (shoulder_left + shoulder_right) / 2.0
    
    # torso height for scaling
    torso_height = np.linalg.norm(shoulder_center - hip_center)
    if torso_height < 1:
        torso_height = 1

    #shin height for scaling
    knee_left = np.array(pts[25][:2])
    ankle_left = np.array(pts[27][:2])
    knee_right = np.array(pts[26][:2])
    ankle_right = np.array(pts[28][:2])

    shin_left = np.linalg.norm(knee_left - ankle_left)
    shin_right = np.linalg.norm(knee_right - ankle_right)
    scale = (shin_left + shin_right) / 2
    if scale < 1e-6:
        scale = 1

    
    normalized_pts = []
    for x, y, score in pts:
        norm_x = (x - hip_center[0]) / scale
        norm_y = (y - hip_center[1]) / scale
        normalized_pts.append((norm_x, norm_y, score))
    
    return normalized_pts

# ## 4) Feature Extraction for DTW
class FeatureExtractor:
    # Extract features from pose keypoints for DTW comparison
    def __init__(self, action="shoulder_press"):
        self.action = action
        self.prev_features = None
        
    def extract_shoulder_press_features(self, pts):
        features = []
        
        if not pts or len(pts) < 33:
            return None
        
        pts_norm = normalize_pose_position(pts)
        
        if USE_ANGLES:
            angle_r_elbow = angle_3pts(pts[11], pts[13], pts[15])  # shoulder-elbow-wrist
            angle_r_shoulder = angle_3pts(pts[13], pts[11], pts[23])  # elbow-shoulder-hip
            
            angle_l_elbow = angle_3pts(pts[12], pts[14], pts[16])
            angle_l_shoulder = angle_3pts(pts[14], pts[12], pts[24])
            
            # Wrist - elbow line angles
            angle_r_elbow_wrist = angle_2pts(pts[13], pts[15])
            angle_l_elbow_wrist = angle_2pts(pts[14], pts[16])
            
            features.extend([
                angle_r_elbow / 180.0,  # Normalize to 0, 1
                angle_l_elbow / 180.0,
                angle_r_shoulder / 180.0,
                angle_l_shoulder / 180.0
            ])
        
        if USE_POSITIONS:
            # Wrist positions
            features.extend([
                pts_norm[15][0],  # Right wrist X
                pts_norm[15][1],  # Right wrist Y
                pts_norm[16][0],  # Left wrist X
                pts_norm[16][1],  # Left wrist Y
            ])
            
            # Elbow positions
            features.extend([
                pts_norm[13][0],  # Right elbow X
                pts_norm[13][1],  # Right elbow Y
                pts_norm[14][0],  # Left elbow X
                pts_norm[14][1],  # Left elbow Y
            ])
        
        # Symmetry features - Difference between left and right angles
        if USE_ANGLES:
            symmetry_elbow = abs(angle_r_elbow - angle_l_elbow) / 180.0
            symmetry_shoulder = abs(angle_r_shoulder - angle_l_shoulder) / 180.0
            features.extend([symmetry_elbow, symmetry_shoulder])
        
        if USE_VELOCITIES and self.prev_features is not None:
            velocities = np.array(features) - np.array(self.prev_features[:len(features)])
            features.extend(velocities.tolist())
        elif USE_VELOCITIES: # for first frame
            features.extend([0] * len(features))
        
        self.prev_features = features.copy()
        return features
    
    def extract_features(self, pts):
        if self.action == "shoulder_press":
            return self.extract_shoulder_press_features(pts)
        else:
            raise ValueError(f"Unknown action: {self.action}")

# ## 5) DTW-based Movement Analyzer


class DTWMovementAnalyzer:
    # Analyze and compare movements using Dynamic Time Warping
    def __init__(self, action="shoulder_press"):
        self.action = action
        self.trainer_sequence = None
        self.trainer_general_data = {}
        
    def record_trainer_sequence(self, video_path):
        # Record and extract features from trainer video
        print(f"Recording trainer sequence from: {video_path}")
        
        detector = PoseDetector("blazepose")
        extractor = FeatureExtractor(self.action)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open trainer video")
            return False
        
        sequence = []
        frame_count = 0

        min_elbow_angles = []
        max_elbow_angles = []
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            pts = detector.infer(frame)
            if pts and len(pts) == 33:
                features = extractor.extract_features(pts)
                if features:
                    sequence.append(features)
                    
                    angle_r_elbow = angle_3pts(pts[11], pts[13], pts[15])
                    angle_l_elbow = angle_3pts(pts[12], pts[14], pts[16])
                    min_elbow_angles.append(min(angle_r_elbow, angle_l_elbow))
                    max_elbow_angles.append(max(angle_r_elbow, angle_l_elbow))
                    
            frame_count += 1
        
        cap.release()
        
        self.trainer_sequence = np.array(sequence)

        self.trainer_general_data = {
            'frame_count': frame_count,
            'sequence_length': len(sequence),
            'min_elbow_angle': np.min(min_elbow_angles) if min_elbow_angles else 0,
            'max_elbow_angle': np.max(max_elbow_angles) if max_elbow_angles else 180,
            'avg_min_elbow': np.mean(min_elbow_angles) if min_elbow_angles else 0,
            'avg_max_elbow': np.mean(max_elbow_angles) if max_elbow_angles else 180
        }
        
        print(f"{len(sequence)} frames with features")
        print(f"Min elbow angle: {self.trainer_general_data['min_elbow_angle']:.1f}°")
        print(f"Max elbow angle: {self.trainer_general_data['max_elbow_angle']:.1f}°")
        
        return True
    
    def analyze_user_video(self, video_path):
        # Compare user video with trainer
        if self.trainer_sequence is None:
            print("No trainer sequence loaded")
            return None
                
        detector = PoseDetector("blazepose")
        extractor = FeatureExtractor(self.action)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open user video")
            return None
        
        user_sequence = []
        frame_results = []
        reps = 0
        down = False
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            pts = detector.infer(frame)
            if pts and len(pts) == 33:
                features = extractor.extract_features(pts)
                if features:
                    user_sequence.append(features)
                
                    # Count reps
                    angle_r_elbow = angle_3pts(pts[11], pts[13], pts[15])
                    if angle_r_elbow < self.trainer_general_data['avg_min_elbow'] + 15:
                        down = True
                    if down and angle_r_elbow > self.trainer_general_data['avg_min_elbow'] + 15:
                        reps += 1
                        down = False
                    
                    # Visualize
                    vis = detector.draw(frame.copy(), pts, score_thresh=SCORE_THRESH)
                    cv2.putText(vis, f"Reps: {reps}", (12, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.putText(vis, f"Processing for DTW...", (12, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    
                    cv2.imshow("User Performance (ESC to quit)", vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(user_sequence) == 0:
            print("No valid poses detected in user video")
            return None
        
        # DTW comparison
        user_sequence = np.array(user_sequence)
        print(f"User video: {len(user_sequence)} frames")
        print(f"Trainer video: {len(self.trainer_sequence)} frames")
        
        distance, path = fastdtw(self.trainer_sequence, user_sequence, dist=euclidean)
        normalized_distance = distance / len(path)
        
        # Scaling determined empyrically by watching the video
        max_expected_distance = 5.0
        similarity_score = max(0, min(100, 100 * (1 - normalized_distance / max_expected_distance)))
        
        path_array = np.array(path)
        trainer_indices = path_array[:, 0]
        user_indices = path_array[:, 1]
        
        # Segments with high distance represent poor form
        segment_distances = []
        window_size = 10 # windows of 10 frames
        for i in range(0, len(path) - window_size, window_size):
            segment_path = path[i:i+window_size]
            segment_dist = sum([euclidean(self.trainer_sequence[t], user_sequence[u]) 
                               for t, u in segment_path]) / window_size
            segment_distances.append(segment_dist)
        
        # Generate feedback
        results = {
            'similarity_score': similarity_score,
            'dtw_distance': distance,
            'normalized_distance': normalized_distance,
            'reps_counted': reps,
            'user_frames': len(user_sequence),
            'trainer_frames': len(self.trainer_sequence),
            'alignment_path_length': len(path),
            'worst_segments': np.argsort(segment_distances)[-3:][::-1] if segment_distances else [],
            'feedback': self._generate_feedback(similarity_score, segment_distances)
        }
        
        return results
    
    def _generate_feedback(self, score, segment_distances):
        # Generate form feedback based on DTW analysis
        feedback = []
        
        if score >= 90:
            feedback.append("Great form")
        elif score >= 75:
            feedback.append("Good form")
        elif score >= 60:
            feedback.append("Ok form")
        else:
            feedback.append("Room for improvement")
        
        if segment_distances:
            avg_dist = np.mean(segment_distances)
            std_dist = np.std(segment_distances)
            
            # Find problematic segments
            problem_segments = [i for i, d in enumerate(segment_distances) if d > avg_dist + std_dist]
            
            if problem_segments:
                if problem_segments[0] < len(segment_distances) * 0.3:
                    feedback.append("Focus on the starting position and initial movement.")
                elif problem_segments[0] > len(segment_distances) * 0.7:
                    feedback.append("Focus on the  form at the end of each rep.")
                else:
                    feedback.append("Work on maintaining consistent form throughout the movement.")
        
        # Specific shoulder press feedback
        if self.action == "shoulder_press":
            if score < 70:
                feedback.append("Tips: Keep elbows at proper angle, maintain symmetry between arms.")
                feedback.append("Watch the trainer's elbow depth and wrist positioning.")
        
        return feedback

# ## 6) Main Execution Pipeline

def save_trainer_data(analyzer, filename="trainer_data.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({
            'sequence': analyzer.trainer_sequence,
            'general_data': analyzer.trainer_general_data,
            'action': analyzer.action
        }, f)
    print(f"Trainer data saved to {filename}")

def load_trainer_data(analyzer, filename="trainer_data.pkl"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    analyzer.trainer_sequence = data['sequence']
    analyzer.trainer_general_data = data['general_data']
    analyzer.action = data['action']
    print(f"Trainer data loaded from {filename}")
    return True

def main():    
    analyzer = DTWMovementAnalyzer(action=ACTION)
    
    trainer_video = "Videos/shoulder_press/shoulder_press_trainer.mp4"
    user_video = "Videos/shoulder_press/shoulder_press_jimena.mp4"
    
    if analyzer.record_trainer_sequence(trainer_video):
        save_trainer_data(analyzer, f"{ACTION}_trainer.pkl")
    else:
        print("Failed to get trainer features")
        return
    
    results = analyzer.analyze_user_video(user_video)
    
    if results:
        print(f"Similarity Score: {results['similarity_score']:.1f}/100")
        print(f"Repetitions Counted: {results['reps_counted']}")
        print(f"DTW Distance: {results['dtw_distance']:.1f}")
        print(f"Normalized Distance: {results['normalized_distance']:.2f}")
        
        print("Feedback:")
        for fb in results['feedback']:
            print(f"- {fb}")
        
        print("Analysis complete!")
    else:
        print("Analysis failed")

if __name__ == "__main__":
    main()

