import cv2
import numpy as np
import requests
import os
import json
import pickle
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MODEL = "blazepose"   
DRAW_SKELETON = True
SCORE_THRESH = 0.5

USE_ANGLES = True
USE_POSITIONS = False
USE_VELOCITIES = False
NORMALIZE_FEATURES = True

ACTION = "shoulder_press"  # Options: "shoulder_press", "squat", "jumping_jacks"

## 2) PoseDetector Class

# --- 1. Pose Detector ---
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
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
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

# --- 2. Utility Functions ---
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

ALIGN_JOINTS = [11, 12, 23, 24, 25, 26]  # shoulders, hips, knees
VIS_THRESH = 0.5

def align_skeleton(user_pts, trainer_pts,
                   joints_to_use=ALIGN_JOINTS,
                   vis_thresh=VIS_THRESH):
    if not user_pts or len(user_pts) < 33 or not trainer_pts or len(trainer_pts) < 33:
        return user_pts  # skip if invalid

    # Convert to numpy (x, y, score)
    user = np.array(user_pts, float)
    trainer = np.array(trainer_pts, float)

    # 1) Rough scale normalization
    def body_size(p):
        return np.linalg.norm(p[11, :2] - p[23, :2])
    usz = body_size(user)
    trsz = body_size(trainer)
    if usz > 1e-6 and trsz > 1e-6:
        user[:, :2] *= (trsz / usz)

    # 2) Select reliable torso joints
    src, tgt = [], []
    for j in joints_to_use:
        if user[j, 2] > vis_thresh and trainer[j, 2] > vis_thresh:
            src.append(user[j, :2])
            tgt.append(trainer[j, :2])

    src, tgt = np.array(src), np.array(tgt)
    if src.shape[0] < 3:
        return user_pts  # skip if not enough

    # 3) Center
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c, tgt_c = src - src_mean, tgt - tgt_mean

    # 4) Solve for 2D rotation
    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_mean - R @ src_mean

    # 5) Apply transform
    aligned_xy = (R @ user[:, :2].T).T + t
    return [(float(x), float(y), s) for (x, y), (_, _, s) in zip(aligned_xy, user)]

# --- 3. Feature Extractor ---
class FeatureExtractor:
    # Extract features from pose keypoints for DTW comparison
    def __init__(self, ACTION):
        self.action = ACTION
        self.prev_features = None
    
    @property
    def features(self):
        """
        Dynamically generates the list of feature names based on:
        1. The current action (Squat vs Press)
        2. The global configuration flags (USE_ANGLES, USE_POSITIONS)
        """
        names = []

        # 1. ADD ANGLES
        if globals().get("USE_ANGLES", True):
            if self.action == "shoulder_press":
                names.extend([
                    "right_elbow_angle", "left_elbow_angle",
                    "right_shoulder_angle", "left_shoulder_angle"
                ])
            elif self.action == "squat":
                names.extend([
                    "right_hip_angle", "left_hip_angle",
                    "right_knee_angle", "left_knee_angle"
                ])
            elif self.action == "jumping_jacks":
                names.extend([
                    "right_shoulder_abduct", "left_shoulder_abduct"
                ])

        # 2. ADD POSITIONS
        if globals().get("USE_POSITIONS", False):
            if self.action == "shoulder_press":
                # Track wrists and elbows for press
                names.extend([
                    "right_wrist_x", "right_wrist_y",
                    "left_wrist_x", "left_wrist_y",
                    "right_elbow_x", "right_elbow_y",
                    "left_elbow_x", "left_elbow_y"
                ])
            elif self.action == "squat":
                # Track knees and hips for squat
                names.extend([
                    "right_knee_x", "right_knee_y",
                    "left_knee_x", "left_knee_y",
                    "right_hip_x", "right_hip_y",
                    "left_hip_x", "left_hip_y"
                ])
            # Add jumping jacks positions if needed...

        # 3. ADD VELOCITIES
        # This automatically creates velocity names for whatever features exist so far
        if globals().get("USE_VELOCITIES", False):
            names += [f"{n}_velocity" for n in names]

        return names
    
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
    
    def extract_jumping_jacks_features(self, pts):
        features = []
        
        if not pts or len(pts) < 33:
            return None
        
        pts_norm = normalize_pose_position(pts)
        
        if USE_ANGLES:
            # Arm angles 
            angle_r_shoulder_abduction = angle_3pts(pts[23], pts[11], pts[13]) # angle hip-shoulder-elbow
            angle_l_shoulder_abduction = angle_3pts(pts[24], pts[12], pts[14]) # angle hip-shoulder-elbow
            
            # Elbow angles
            angle_r_elbow = angle_3pts(pts[11], pts[13], pts[15])
            angle_l_elbow = angle_3pts(pts[12], pts[14], pts[16])
            
            # Leg angles
            angle_r_hip_abduction = angle_3pts(pts[23], pts[24], pts[26]) # angle left_hip-right_hip-right_knee
            angle_l_hip_abduction = angle_3pts(pts[24], pts[23], pts[25]) # angle right_hip-left_hip-left_knee
            
            # Knee angles (should be somewhat straight)
            angle_r_knee = angle_3pts(pts[24], pts[26], pts[28])
            angle_l_knee = angle_3pts(pts[23], pts[25], pts[27])
            
            features.extend([
                angle_r_shoulder_abduction / 180.0,
                angle_l_shoulder_abduction / 180.0,
                angle_r_elbow / 180.0,
                angle_l_elbow / 180.0,
                angle_r_hip_abduction / 180.0,
                angle_l_hip_abduction / 180.0,
                angle_r_knee / 180.0,
                angle_l_knee / 180.0
            ])
        
        if USE_POSITIONS:
            # Wrist positions
            features.extend([
                pts_norm[15][0],  # Right wrist X
                pts_norm[15][1],  # Right wrist Y
                pts_norm[16][0],  # Left wrist X
                pts_norm[16][1],  # Left wrist Y
            ])
            
            # Ankle positions
            features.extend([
                pts_norm[27][0],  # Right ankle X
                pts_norm[27][1],  # Right ankle Y
                pts_norm[28][0],  # Left ankle X
                pts_norm[28][1],  # Left ankle Y
            ])
        
        # Symmetry features
        if USE_ANGLES:
            symmetry_shoulder = abs(angle_r_shoulder_abduction - angle_l_shoulder_abduction) / 180.0
            symmetry_elbow = abs(angle_r_elbow - angle_l_elbow) / 180.0
            symmetry_hip = abs(angle_r_hip_abduction - angle_l_hip_abduction) / 180.0
            symmetry_knee = abs(angle_r_knee - angle_l_knee) / 180.0
            features.extend([symmetry_shoulder, symmetry_elbow, symmetry_hip, symmetry_knee])
        
        if USE_VELOCITIES and self.prev_features is not None:
            velocities = np.array(features) - np.array(self.prev_features[:len(features)])
            features.extend(velocities.tolist())
        elif USE_VELOCITIES:
            features.extend([0] * len(features))
        
        self.prev_features = features.copy()
        return features
    
    def extract_squats_features(self, pts):
        features = []
        
        if not pts or len(pts) < 33:
            return None
        
        pts_norm = normalize_pose_position(pts)
        
        if USE_ANGLES:
            # Knee angles
            angle_r_knee = angle_3pts(pts[24], pts[26], pts[28])  # hip-knee-ankle
            angle_l_knee = angle_3pts(pts[23], pts[25], pts[27])
            
            # Hip angles
            angle_r_hip = angle_3pts(pts[12], pts[24], pts[26])  # shoulder-hip-knee
            angle_l_hip = angle_3pts(pts[11], pts[23], pts[25])
            
            # Ankle angles
            angle_r_ankle = angle_3pts(pts[26], pts[28], pts[32])  # knee-ankle-foot
            angle_l_ankle = angle_3pts(pts[25], pts[27], pts[31])
            
            # Torso angle shoulder-hip-vertical
            # Approximate by angle between shoulder midpoint, hip midpoint, and a point directly below hip
            shoulder_mid = ((pts[11][0] + pts[12][0]) / 2, (pts[11][1] + pts[12][1]) / 2, 1.0)
            hip_mid = ((pts[23][0] + pts[24][0]) / 2, (pts[23][1] + pts[24][1]) / 2, 1.0)
            # virtual point directly below hip for vertical reference
            vertical_ref = (hip_mid[0], hip_mid[1] + 100, 1.0)
            torso_angle = angle_3pts(shoulder_mid, hip_mid, vertical_ref)
            
            features.extend([
                angle_r_knee / 180.0,
                angle_l_knee / 180.0,
                angle_r_hip / 180.0,
                angle_l_hip / 180.0,
                angle_r_ankle / 180.0,
                angle_l_ankle / 180.0,
                torso_angle / 180.0
            ])
        
        if USE_POSITIONS:
            # Hip positions (track squat depth)
            features.extend([
                pts_norm[23][0],  # Left hip X
                pts_norm[23][1],  # Left hip Y
                pts_norm[24][0],  # Right hip X
                pts_norm[24][1],  # Right hip Y
            ])
            
            # Knee positions (track knee travel)
            features.extend([
                pts_norm[25][0],  # Left knee X
                pts_norm[25][1],  # Left knee Y
                pts_norm[26][0],  # Right knee X
                pts_norm[26][1],  # Right knee Y
            ])
        
        # Symmetry features
        if USE_ANGLES:
            symmetry_knee = abs(angle_r_knee - angle_l_knee) / 180.0
            symmetry_hip = abs(angle_r_hip - angle_l_hip) / 180.0
            symmetry_ankle = abs(angle_r_ankle - angle_l_ankle) / 180.0
            features.extend([symmetry_knee, symmetry_hip, symmetry_ankle])
        
        if USE_VELOCITIES and self.prev_features is not None:
            velocities = np.array(features) - np.array(self.prev_features[:len(features)])
            features.extend(velocities.tolist())
        elif USE_VELOCITIES:
            features.extend([0] * len(features))
        
        self.prev_features = features.copy()
        return features

    def extract_features(self, pts):
        if self.action == "shoulder_press":
            return self.extract_shoulder_press_features(pts)
        if self.action == "jumping_jacks":
            return self.extract_jumping_jacks_features(pts)
        if self.action == "squat":
            return self.extract_squats_features(pts)
        else:
            raise ValueError(f"Unknown action: {self.action}")

# --- 4. DTW Analyzer ---
class DTWMovementAnalyzer:
    def __init__(self, action):
        self.action = action
        self.trainer_sequence = None
        self.trainer_general_data = {}
        self.trainer_ref_pts = None
        #New
        self.extractor = FeatureExtractor(action)

        # AI-related config
        self.feature_names = self.extractor.features
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_url = "https://api.openai.com/v1/chat/completions"
    
    def _init_feature_names(self):
        names = []
        if USE_ANGLES:
            names.extend([
                "right_elbow_angle",
                "left_elbow_angle",
                "right_shoulder_angle",
                "left_shoulder_angle",
                "elbow_symmetry",
                "shoulder_symmetry"
            ])

        if USE_POSITIONS:
            names.extend([
                "right_wrist_x", "right_wrist_y",
                "left_wrist_x", "left_wrist_y",
                "right_elbow_x", "right_elbow_y",
                "left_elbow_x", "left_elbow_y"
            ])

        if USE_VELOCITIES:
            base_count = len(names)
            for i in range(base_count):
                names.append(f"{names[i]}_velocity")

        return names
    
    def _get_rep_metric(self, pts):
        if self.action == "shoulder_press":
            # Rep depends on elbow extension
            r = angle_3pts(pts[11], pts[13], pts[15])
            l = angle_3pts(pts[12], pts[14], pts[16])
            return (r + l) / 2  # Average of both arms
            
        elif self.action == "squat":
            # Rep depends on knee flexion (going down)
            r = angle_3pts(pts[23], pts[25], pts[27])
            l = angle_3pts(pts[24], pts[26], pts[28])
            return (r + l) / 2
            
        elif self.action == "jumping_jacks":
            # Rep depends on shoulder abduction (arms going up)
            r = angle_3pts(pts[23], pts[11], pts[13])
            l = angle_3pts(pts[24], pts[12], pts[14])
            return (r + l) / 2
            
        return 0
    
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
        metrics = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pts = detector.infer(frame)
            if pts and len(pts) == 33:
                # Capture the first valid frame as the reference for alignment
                if self.trainer_ref_pts is None:
                    self.trainer_ref_pts = pts.copy()

                features = extractor.extract_features(pts)
                
                if features:
                    sequence.append(features)
                    
                    # Calculate the dynamic metric for this frame
                    metric = self._get_rep_metric(pts)
                    metrics.append(metric)

                frame_count += 1
        
        cap.release()

        if not metrics or not sequence: 
            print("No valid metrics or features found in trainer video.")
            return False
        
        # Calculate dynamic range based on the specific exercise
        min_metric = np.min(metrics)
        max_metric = np.max(metrics)

        self.trainer_sequence = np.array(sequence)

        # Store dynamic thresholds (20% / 80% of range)
        self.trainer_general_data = {
            "frame_count": frame_count,
            "sequence_length": len(sequence),
            "rep_metric_min": min_metric,
            "rep_metric_max": max_metric,
            "rep_threshold_low": min_metric + (max_metric - min_metric) * 0.2,
            "rep_threshold_high": min_metric + (max_metric - min_metric) * 0.8
        }




        # --- Rep Counting Initialization ---
        trainer_reps = 0
        rep_state = "start"  # States: 'start', 'mid'
        
        # 1. Retrieve dynamic range from trainer data, or set defaults
        # Note: 'rep_metric_min/max' should ideally be set in record_trainer_sequence
        t_min = self.trainer_general_data.get("rep_metric_min", 70) 
        t_max = self.trainer_general_data.get("rep_metric_max", 170)
        
        # 2. Define thresholds (e.g., 25% and 75% of the range of motion)
        r_range = t_max - t_min
        thresh_low = t_min + (r_range * 0.25)
        thresh_high = t_min + (r_range * 0.75)

        # 3. Define movement direction
        # Squats start High (180), go Low (<90). Press starts Low, goes High.
        target_is_low = (self.action == "squat") 

        cap = cv2.VideoCapture(video_path)
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pts_raw = detector.infer(frame)
            if pts_raw and len(pts_raw) == 33:
                # align for features (Kabsch / 4th-iter behavior)
                if self.trainer_ref_pts is not None:
                    pts_aligned = align_skeleton(pts_raw, self.trainer_ref_pts)
                else:
                    pts_aligned = pts_raw

                features = extractor.extract_features(pts_aligned)
                if features:
                    sequence.append(features)

                # --- Dynamic Rep Counting Logic ---
                metric = self._get_rep_metric(pts_raw)
                
                if target_is_low: 
                    # Case: SQUAT (Start High -> Go Low -> Return High)
                    if rep_state == "start":
                        if metric < thresh_low: # User went down
                            rep_state = "mid"
                    elif rep_state == "mid":
                        if metric > thresh_high: # User stood back up
                            trainer_reps += 1
                            rep_state = "start"
                else:
                    # Case: PRESS / JUMPING JACKS (Start Low -> Go High -> Return Low)
                    if rep_state == "start":
                        if metric > thresh_high: # User pushed up
                            rep_state = "mid"
                    elif rep_state == "mid":
                        if metric < thresh_low: # User came back down
                            trainer_reps += 1
                            rep_state = "start"

        cap.release()
        self
        print(f"{len(sequence)} frames with features")
        print(f"Trainer Metric Range ({self.action}): {min_metric:.1f} to {max_metric:.1f}")

        return trainer_reps

    # --- analyze_user_video ---
    def analyze_user_video(self, video_path, visualize=True, trainer_reps=0):
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
        
        # --- Rep Counting Initialization ---
        reps = 0
        rep_state = "start"  # States: 'start', 'mid'
        
        # 1. Retrieve dynamic range from trainer data, or set defaults
        # Note: 'rep_metric_min/max' should ideally be set in record_trainer_sequence
        t_min = self.trainer_general_data.get("rep_metric_min", 70) 
        t_max = self.trainer_general_data.get("rep_metric_max", 170)
        
        # 2. Define thresholds (e.g., 25% and 75% of the range of motion)
        r_range = t_max - t_min
        thresh_low = t_min + (r_range * 0.25)
        thresh_high = t_min + (r_range * 0.75)

        # 3. Define movement direction
        # Squats start High (180), go Low (<90). Press starts Low, goes High.
        target_is_low = (self.action == "squat") 

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pts_raw = detector.infer(frame)
            if pts_raw and len(pts_raw) == 33:
                # align for features (Kabsch / 4th-iter behavior)
                if self.trainer_ref_pts is not None:
                    pts_aligned = align_skeleton(pts_raw, self.trainer_ref_pts)
                else:
                    pts_aligned = pts_raw

                features = extractor.extract_features(pts_aligned)
                if features:
                    user_sequence.append(features)

                # --- Dynamic Rep Counting Logic ---
                metric = self._get_rep_metric(pts_raw)
                
                if target_is_low: 
                    # Case: SQUAT (Start High -> Go Low -> Return High)
                    if rep_state == "start":
                        if metric < thresh_low: # User went down
                            rep_state = "mid"
                    elif rep_state == "mid":
                        if metric > thresh_high: # User stood back up
                            reps += 1
                            rep_state = "start"
                else:
                    # Case: PRESS / JUMPING JACKS (Start Low -> Go High -> Return Low)
                    if rep_state == "start":
                        if metric > thresh_high: # User pushed up
                            rep_state = "mid"
                    elif rep_state == "mid":
                        if metric < thresh_low: # User came back down
                            reps += 1
                            rep_state = "start"

                if visualize:
                    vis = detector.draw(
                        frame.copy(), pts_raw, score_thresh=SCORE_THRESH
                    )
                    cv2.putText(
                        vis,
                        f"Reps: {reps}",
                        (12, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (60, 60, 255),
                        2,
                    )
                    # Optional: Visualize the metric being tracked
                    cv2.putText(vis, f"{self.action} metric: {int(metric)}", (12, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    scale = 0.6
                    vis_small = cv2.resize(
                        vis,
                        (int(vis.shape[1] * scale), int(vis.shape[0] * scale)),
                    )
                    cv2.imshow("User Performance (ESC to quit)", vis_small)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

        cap.release()
        if visualize:
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

        max_expected_distance = 5.0
        similarity_score = max(
            0, min(100, 100 * (1 - normalized_distance / max_expected_distance))
        )

        # segments for feedback
        segment_distances = []
        window_size = 10
        for i in range(0, len(path) - window_size, window_size):
            segment_path = path[i : i + window_size]
            segment_dist = (
                sum(
                    [
                        euclidean(self.trainer_sequence[t], user_sequence[u])
                        for t, u in segment_path
                    ]
                )
                / window_size
            )
            segment_distances.append(segment_dist)

        # AI analysis and feedback
        feature_analysis = self._analyze_feature_differences(
            self.trainer_sequence, user_sequence, path
        )
        timing_analysis = self._analyze_timing_coordination(
            self.trainer_sequence, user_sequence, path, reps, trainer_reps
        )
        movement_patterns = self._identify_movement_patterns(
            feature_analysis, timing_analysis
        )
        chatgpt_prompt = self._generate_chatgpt_prompt(
            movement_patterns, similarity_score, reps
        )
        ai_feedback = self.get_chatgpt_feedback(chatgpt_prompt)

        results = {
            "similarity_score": similarity_score,
            "dtw_distance": distance,
            "normalized_distance": normalized_distance,
            "reps_counted": reps,
            "user_frames": len(user_sequence),
            "trainer_frames": len(self.trainer_sequence),
            "alignment_path_length": len(path),
            "worst_segments": np.argsort(segment_distances)[-3:][::-1]
            if segment_distances
            else [],
            "feedback": self._generate_feedback(
                similarity_score, segment_distances
            ),
            "detailed_analysis": {
                "feature_differences": feature_analysis,
                "timing_coordination": timing_analysis,
                "movement_patterns": movement_patterns,
                "chatgpt_prompt": chatgpt_prompt,
                "ai_feedback": ai_feedback,
            },
        }

        return results

    # --- Rule-based Feedback ---
    def _generate_feedback(self, score, segment_distances):
        # Generate form feedback based on DTW analysis
        feedback = []
        
        if score == 0:
            feedback.append("No full repetitions detected.")
            feedback.append("Try to complete the full range of motion.")
            return feedback
        
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
            problem_segments = [
                i
                for i, d in enumerate(segment_distances)
                if d > avg_dist + std_dist
            ]

            if problem_segments:
                if problem_segments[0] < len(segment_distances) * 0.3:
                    feedback.append(
                        "Focus on the starting position and initial movement."
                    )
                elif problem_segments[0] > len(segment_distances) * 0.7:
                    feedback.append(
                        "Focus on the form at the end of each rep."
                    )
                else:
                    feedback.append(
                        "Work on maintaining consistent form throughout the movement."
                    )

        # Specific shoulder press feedback
        if self.action == "shoulder_press":
            if score < 70:
                feedback.append(
                    "Tips: Keep elbows at proper angle and maintain symmetry between arms."
                )
                feedback.append(
                    "Watch the trainer's elbow depth and wrist positioning."
                )

        return feedback


    # --- Helper Analysis Methods ---
    def _analyze_feature_differences(self, trainer_seq, user_seq, path):
        feature_diffs = {name: [] for name in self.feature_names}

        # Analyze differences along DTW path
        for trainer_idx, user_idx in path:
            trainer_features = trainer_seq[trainer_idx]
            user_features = user_seq[user_idx]

            for i, name in enumerate(self.feature_names):
                if i < len(trainer_features) and i < len(user_features):
                    diff = abs(trainer_features[i] - user_features[i])
                    feature_diffs[name].append(diff)

        # Stats per feature
        feature_stats = {}
        for name, diffs in feature_diffs.items():
            if diffs:
                feature_stats[name] = {
                    "mean_diff": np.mean(diffs),
                    "max_diff": np.max(diffs),
                    "std_diff": np.std(diffs),
                    "percentile_95": np.percentile(diffs, 95),
                }

        return {
            "feature_stats": feature_stats,
            "most_problematic_features": sorted(
                feature_stats.items(),
                key=lambda x: x[1]["mean_diff"],
                reverse=True,
            )[:5],  # top 5
        }

    def _analyze_timing_coordination(self, trainer_seq, user_seq, path, user_reps, trainer_reps):
        # Pace
        trainer_frames = len(trainer_seq)
        user_frames = len(user_seq)
        user_pace = user_frames / user_reps if user_reps > 0 else user_frames
        trainer_pace = trainer_frames / trainer_reps if trainer_reps > 0 else trainer_frames
        pace_ratio = user_pace / trainer_pace if trainer_pace > 0 else 1.0

        # Synchronization over sliding windows
        sync_scores = []
        window_size = 20

        for i in range(0, len(path) - window_size, window_size // 2):
            window_path = path[i : i + window_size]

            trainer_indices = [p[0] for p in window_path]
            user_indices = [p[1] for p in window_path]

            trainer_rate = (trainer_indices[-1] - trainer_indices[0]) / len(
                trainer_indices
            )
            user_rate = (user_indices[-1] - user_indices[0]) / len(user_indices)

            if trainer_rate > 0:
                ratio = (
                    user_rate / trainer_rate
                    if user_rate < trainer_rate
                    else trainer_rate / user_rate
                )
                sync_score = min(1.0, ratio)
            else:
                sync_score = 0

            sync_scores.append(sync_score)

        # Left-right coordination (angles)
        if USE_ANGLES:
            left_right_coordination = []
            for _, user_idx in path:
                user_features = user_seq[user_idx]
                if len(user_features) >= 2:
                    lr_diff = abs(user_features[0] - user_features[1])
                    left_right_coordination.append(lr_diff)
            coordination_score = (
                1.0 - np.mean(left_right_coordination)
                if left_right_coordination
                else 1.0
            )
        else:
            coordination_score = 1.0

        if pace_ratio < 0.1:
            pace_description = "significantly faster than trainer"
        elif pace_ratio < 0.14:
            pace_description = "slightly faster than trainer"
        elif pace_ratio <= 0.2:
            pace_description = "matching trainer pace"
        elif pace_ratio <= 0.25:
            pace_description = "slightly slower than trainer"
        else:
            pace_description = "significantly slower than trainer"

        return {
            "pace_ratio": pace_ratio,
            "pace_description": pace_description,
            "avg_synchronization": np.mean(sync_scores) if sync_scores else 1.0,
            "coordination_score": coordination_score,
            "timing_consistency": np.std(sync_scores) if sync_scores else 0,
        }

    def _identify_movement_patterns(self, feature_analysis, timing_analysis):
        patterns = {
            "primary_issues": [],
            "secondary_issues": [],
            "strengths": [],
            "specific_corrections": [],
        }

        # Most problematic features
        for feature_name, stats in feature_analysis["most_problematic_features"][:3]:
            if "elbow_angle" in feature_name:
                if stats["mean_diff"] > 0.15:
                    side = "right" if "right" in feature_name else "left"
                    patterns["primary_issues"].append(
                        f"{side} arm elbow angle inconsistent"
                    )
                    patterns["specific_corrections"].append(
                        f"Maintain proper {side} elbow bend throughout the movement"
                    )
            elif "shoulder_angle" in feature_name:
                if stats["mean_diff"] > 0.15:
                    side = "right" if "right" in feature_name else "left"
                    patterns["primary_issues"].append(
                        f"{side} shoulder positioning needs adjustment"
                    )
                    patterns["specific_corrections"].append(
                        f"Keep {side} shoulder stable and aligned with the torso"
                    )
            elif "symmetry" in feature_name:
                if stats["mean_diff"] > 0.1:
                    patterns["primary_issues"].append(
                        "Asymmetric movement between arms"
                    )
                    patterns["specific_corrections"].append(
                        "Move both arms in sync with similar range of motion"
                    )
            elif "wrist" in feature_name and USE_POSITIONS:
                if stats["mean_diff"] > 0.2:
                    side = "right" if "right" in feature_name else "left"
                    patterns["secondary_issues"].append(
                        f"{side} wrist path deviates from ideal"
                    )

        # Timing + coordination
        if timing_analysis["pace_ratio"] < 0.8 or timing_analysis["pace_ratio"] > 1.2:
            patterns["secondary_issues"].append(
                f"Movement pace is {timing_analysis['pace_description']}"
            )
            patterns["specific_corrections"].append(
                "Match your tempo more closely to the trainer"
            )

        if timing_analysis["coordination_score"] < 0.7:
            patterns["primary_issues"].append(
                "Poor coordination between left and right sides"
            )
            patterns["specific_corrections"].append(
                "Practice moving both arms together with synchronized timing"
            )

        # Strengths
        if timing_analysis["avg_synchronization"] > 0.85:
            patterns["strengths"].append("Good overall timing")

        if timing_analysis["coordination_score"] > 0.85:
            patterns["strengths"].append("Excellent bilateral coordination")

        # Well-matching features
        for feature_name, stats in feature_analysis["feature_stats"].items():
            if stats["mean_diff"] < 0.05:
                if "elbow" in feature_name and "angle" in feature_name:
                    patterns["strengths"].append(
                        f"Good {feature_name.replace('_', ' ')}"
                    )
                    break

        return patterns

    def _generate_chatgpt_prompt(self, patterns, similarity_score, rep_count):
        prompt = f"""
You are a fitness coach giving SHORT, DIRECT feedback for a {self.action.replace('_',' ')}.

Keep it:
- concise (max 6 sentences)
- specific
- NOT emotional
- NOT a letter
- NO greetings or sign-offs
- NO fluff

USER PERFORMANCE:
- Similarity: {similarity_score:.1f}/100
- Reps: {rep_count}

PRIMARY ISSUES:
{chr(10).join(f"- {i}" for i in patterns['primary_issues']) if patterns['primary_issues'] else "- None"}

SECONDARY ISSUES:
{chr(10).join(f"- {i}" for i in patterns['secondary_issues']) if patterns['secondary_issues'] else "- None"}

STRENGTHS:
{chr(10).join(f"- {s}" for s in patterns['strengths']) if patterns['strengths'] else "- None"}

CORRECTIONS NEEDED:
{chr(10).join(f"- {c}" for c in patterns['specific_corrections'])}

Now give one paragraph of feedback:
- Start with what went well
- Give 2–3 specific corrections
- Keep it technical and short.
"""
        return prompt

    def get_chatgpt_feedback(self, prompt: str) -> str:
        if not self.openai_api_key:
            return "ChatGPT API key not configured. Set OPENAI_API_KEY to enable AI feedback."

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert fitness coach. "
                        "Give concise, technical feedback only. No greetings, no sign-offs."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 300,
        }

        try:
            response = requests.post(self.openai_api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            feedback = result["choices"][0]["message"]["content"]
            return feedback
        except requests.exceptions.RequestException as e:
            return f"Error getting ChatGPT feedback: {str(e)}"

# --- Global Helper for Persistence ---
def save_trainer_data(analyzer, filename="trainer_data.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({
            'sequence': analyzer.trainer_sequence,
            'general_data': analyzer.trainer_general_data,
            'action': analyzer.action,
            'trainer_ref_pts': analyzer.trainer_ref_pts,
        }, f)
    print(f"Trainer data saved to {filename}")

def load_trainer_data(analyzer, filename="trainer_data.pkl"):
    if not os.path.exists(filename): return False
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    analyzer.trainer_sequence = data['sequence']
    analyzer.trainer_general_data = data['general_data']
    analyzer.action = data['action']
    analyzer.trainer_ref_pts = data.get('trainer_ref_pts')
    print(f"Trainer data loaded from {filename}")
    return True

## 6) Main Execution Pipeline
def fit_into_cell(frame, cell_w, cell_h):
    h, w = frame.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    y_off = (cell_h - new_h) // 2
    x_off = (cell_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas
def save_trainer_data(analyzer, filename="trainer_data.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({
            'sequence': analyzer.trainer_sequence,
            'general_data': analyzer.trainer_general_data,
            'action': analyzer.action,
            'trainer_ref_pts': analyzer.trainer_ref_pts,
        }, f)
    print(f"Trainer data saved to {filename}")

def load_trainer_data(analyzer, filename="trainer_data.pkl"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    analyzer.trainer_sequence = data['sequence']
    analyzer.trainer_general_data = data['general_data']
    analyzer.action = data['action']
    analyzer.trainer_ref_pts = data.get('trainer_ref_pts')
    print(f"Trainer data loaded from {filename}")
    return True
def show_side_by_side_with_pose(trainer_path, user_path,
                                analyzer,
                                label_left="trainer", label_right="user",
                                width_each=480, height_each=360):
    tr_cap = cv2.VideoCapture(trainer_path)
    us_cap = cv2.VideoCapture(user_path)

    trainer_detector = PoseDetector("blazepose")
    user_detector = PoseDetector("blazepose")

    cv2.namedWindow("Comparison 1x2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Comparison 1x2", width_each*2, height_each)

    # user rep state
    user_reps = 0
    user_down = False

    while True:
        ok_tr, tr_frame = tr_cap.read()
        ok_us, us_frame = us_cap.read()

        if not ok_tr and not ok_us:
            break

        # ---------- TRAINER SIDE ----------
        if ok_tr:
            tr_pts = trainer_detector.infer(tr_frame)
            if tr_pts and len(tr_pts) == 33:
                tr_frame = trainer_detector.draw(tr_frame.copy(), tr_pts, score_thresh=SCORE_THRESH)
        else:
            tr_frame = np.zeros((height_each, width_each, 3), dtype=np.uint8)

        # ---------- USER SIDE ----------
        if ok_us:
            us_pts = user_detector.infer(us_frame)
            if us_pts and len(us_pts) == 33:
                # rep counting like before
                angle_r_elbow = angle_3pts(us_pts[11], us_pts[13], us_pts[15])
                if angle_r_elbow < analyzer.trainer_general_data['avg_min_elbow'] + 15:
                    user_down = True
                if user_down and angle_r_elbow > analyzer.trainer_general_data['avg_min_elbow'] + 15:
                    user_reps += 1
                    user_down = False

                us_frame = user_detector.draw(us_frame.copy(), us_pts, score_thresh=SCORE_THRESH)
        else:
            us_frame = np.zeros((height_each, width_each, 3), dtype=np.uint8)

        # ---------- NOW resize ----------
        tr_fit = fit_into_cell(tr_frame, width_each, height_each)
        us_fit = fit_into_cell(us_frame, width_each, height_each)

        # ---------- add labels (this is what the old version did) ----------
        # left label
        cv2.rectangle(tr_fit, (5, 5), (5 + 150, 35), (0, 0, 0), -1)
        cv2.putText(tr_fit, label_left, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # right label
        cv2.rectangle(us_fit, (5, 5), (5 + 150, 35), (0, 0, 0), -1)
        cv2.putText(us_fit, label_right, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # ---------- extra user info ----------
        cv2.putText(us_fit, f"Reps: {user_reps}", (12, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)

        # ---------- show ----------
        grid = cv2.hconcat([tr_fit, us_fit])
        cv2.imshow("Comparison 1x2", grid)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    tr_cap.release()
    us_cap.release()
    cv2.destroyAllWindows()

def format_ai_feedback(text: str) -> str:
    """
    Take the AI feedback paragraph and break it into bullet-like lines
    so it doesn't show as one long wrapped line in the notebook output.
    """
    # Split on periods, keep only non-empty trimmed sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    # Reattach the period and put each on its own line with a dash
    return "\n".join(f"- {s}." for s in sentences)

def main():
    analyzer = DTWMovementAnalyzer(action=ACTION)

    trainer_video = "Videos/shoulder_press/shoulder_press_trainer.mp4"
    user_videos = [
        "Videos/jumping_jacks/1_jumping_jacks_jimena_0_deg.mp4",
        "Videos/jumping_jacks/1_jumping_jacks_jimena_45_deg.mp4",
        "Videos/jumping_jacks/1_jumping_jacks_jimena_90_deg.MOV",
    ]

    # 1) Extract trainer features once
    trainer_reps = analyzer.record_trainer_sequence(trainer_video)

    if not trainer_reps:
        print("Failed to get trainer features")
        return
    save_trainer_data(analyzer, f"{ACTION}_trainer.pkl")

    # 2) Analyze each user video with rule-based and AI feedback
    for user_video in user_videos:
        print("\n==============================")
        print(f"Analyzing: {user_video}")
        print(trainer_reps)

        res = analyzer.analyze_user_video(user_video, visualize=False, trainer_reps=trainer_reps)

        if not res:
            print("Analysis failed")
            continue

        # Basic numeric metrics
        print(f"Similarity Score: {res['similarity_score']:.1f}/100")
        print(f"Repetitions Counted: {res['reps_counted']}")
        print(f"DTW Distance: {res['dtw_distance']:.1f}")
        print(f"Normalized Distance: {res['normalized_distance']:.2f}")

        # Rule-based feedback
        print("\nFeedback:")
        for fb in res['feedback']:
            print(f"- {fb}")

        # AI feedback (formatted into bullet-style lines)
        ai_fb = res.get("detailed_analysis", {}).get("ai_feedback")
        if ai_fb:
            print("\nAI Feedback:")
            print(format_ai_feedback(ai_fb))

    # 3) Visual comparison after analysis
    angles = ["0 deg", "45 deg", "90 deg"]
    for user_video, angle in zip(user_videos, angles):
        show_side_by_side_with_pose(
            trainer_video,
            user_video,
            analyzer,
            label_left="trainer",
            label_right=angle,
            width_each=480,
            height_each=360
        )

    print("\nAll analyses done.")
    

if __name__ == "__main__":
    main()