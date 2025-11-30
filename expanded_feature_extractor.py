class FeatureExtractor:
    # Extract features from pose keypoints for DTW comparison
    def __init__(self, ACTION):
        self.action = ACTION
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
        if self.action == "squats":
            return self.extract_squats_features(pts)
        else:
            raise ValueError(f"Unknown action: {self.action}")