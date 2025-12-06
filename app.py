
'''
**brew install ffmpeg**
'''
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
import time
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import traceback

# Import components from dtw2
from dtw2 import (
    DTWMovementAnalyzer, PoseDetector, angle_3pts, align_skeleton
)

load_dotenv() 

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = os.path.join('static', 'output')
MODEL = "blazepose"

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 32 MB limit

# --- Global State ---
analyzers = {}  # Stores DTWMovementAnalyzer instances
detector = None # Global PoseDetector instance

# Map internal action names to their trainer video filenames in /static/
TRAINER_VIDEOS = {
    "shoulder_press": "trainer_shoulder_press.mp4",
    "squat": "trainer_squats.mp4",
    "jumping_jacks": "trainer_jumping_jacks.mp4"
}

def load_analyzers():
    """Initializes the PoseDetector and loads trainer data for all defined exercises."""
    global detector, analyzers
    
    print("Initializing PoseDetector...")
    detector = PoseDetector(MODEL)
    
    print("Initializing Analyzers...")
    for action, video_filename in TRAINER_VIDEOS.items():
        video_path = os.path.join("static", video_filename)
        
        if os.path.exists(video_path):
            print(f"Loading {action} from {video_path}...")
            try:
                # Initialize analyzer for this specific action
                analyzer = DTWMovementAnalyzer(action)
                analyzer.record_trainer_sequence(video_path)
                analyzers[action] = analyzer
            except Exception as e:
                print(f"Error loading {action}: {e}")
                traceback.print_exc()
        else:
            print(f"Warning: Trainer video for {action} not found at {video_path}")

# Initialize on startup
try:
    load_analyzers()
    if not analyzers:
        print("WARNING: No analyzers were loaded. Check your 'static' folder.")
except Exception as e:
    print(f"FATAL ERROR during initialization: {e}")
    traceback.print_exc()

# ------------------------

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/output/<filename>')
def serve_output_video(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/static/<filename>')
def serve_trainer_video(filename):
    return send_from_directory('static', filename)

@app.route('/upload_and_analyze', methods=['POST'])
def upload_and_analyze():
    # 1. Get the Exercise Type
    action = request.form.get("action") 
    
    if not action:
        action = request.form.get("exercise")

    if not action:
        return jsonify({"error": "No exercise action specified"}), 400
    
    if action not in analyzers:
        return jsonify({"error": f"Action '{action}' is not supported or trainer video is missing."}), 400

    # 2. Validate Video File
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 3. Save User Video
    base_filename = secure_filename(file.filename or 'user_video')
    timestamp = int(time.time())
    input_filename = f"{os.path.splitext(base_filename)[0]}_{timestamp}.webm"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    file.save(input_path)
    
    # 4. Select the Correct Analyzer
    current_analyzer = analyzers[action]
    
    # 5. Process
    output_filename = f"analyzed_{action}_{timestamp}.mp4"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    try:
        # Pass the specific analyzer and the global detector
        results = process_video_file(input_path, output_path, current_analyzer, detector)
        
        # Add URLs
        results['analyzed_video_url'] = f'/static/output/{output_filename}'
        results['trainer_video_url'] = f'/static/{TRAINER_VIDEOS[action]}'
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error processing {action}:")
        traceback.print_exc()
        return jsonify({"error": f"Failed to process video: {str(e)}"}), 500

def process_video_file(input_path, output_path, analyzer, detector):
    """
    Processes video using the PASSED analyzer instance (specific to the exercise).
    This keeps app.py agnostic to the specific exercise math.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open user video at {input_path}")

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60: fps = 20.0
        
    # Writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    user_sequence = []
    
    # --- Rep Counting State ---
    reps = 0
    rep_state = "start" # start, mid
    
    # Get dynamic thresholds from the loaded analyzer
    # These are calculated in analyzer.record_trainer_sequence()
    t_data = analyzer.trainer_general_data
    if t_data and "rep_threshold_low" in t_data:
        thresh_low = t_data["rep_threshold_low"]
        thresh_high = t_data["rep_threshold_high"]
    else:
        # Fallback defaults if trainer video failed to load properly
        thresh_low, thresh_high = 75, 160 

    # Determine direction: Squats are "Low" target, Press/Jacks are "High" target
    target_is_low = (analyzer.action == "squat")

    # Access the extractor directly from the analyzer instance
    extractor = analyzer.extractor 
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        pts_raw = detector.infer(frame)
        vis_frame = frame.copy() 

        if pts_raw and len(pts_raw) == 33:
            # Align user skeleton to trainer's first frame
            if analyzer.trainer_ref_pts is not None:
                pts_aligned = align_skeleton(pts_raw, analyzer.trainer_ref_pts)
            else:
                pts_aligned = pts_raw
            
            # Extract features
            features = extractor.extract_features(pts_aligned)
            if features:
                user_sequence.append(features)
            
            # --- Dynamic Rep Counting ---
            # 1. Get the metric relevant to this exercise (Knee angle, Elbow angle, etc.)
            metric = analyzer._get_rep_metric(pts_raw)

            # 2. Update State Machine
            if target_is_low:
                # Squat: Start High -> Go Low -> Return High
                if rep_state == "start" and metric < thresh_low:
                    rep_state = "mid"
                elif rep_state == "mid" and metric > thresh_high:
                    reps += 1
                    rep_state = "start"
            else:
                # Press/Jacks: Start Low -> Go High -> Return Low
                if rep_state == "start" and metric > thresh_high:
                    rep_state = "mid"
                elif rep_state == "mid" and metric < thresh_low:
                    reps += 1
                    rep_state = "start"
            
            # Draw
            vis_frame = detector.draw(vis_frame, pts_raw, score_thresh=0.5)
            cv2.putText(vis_frame, f"Reps: {reps}", (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            # Optional: Show the metric being tracked
            cv2.putText(vis_frame, f"Metric: {int(metric)}", (12, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        writer.write(vis_frame)

    cap.release()
    writer.release()

    # --- DTW Analysis ---
    if len(user_sequence) == 0:
        # Cleanup if failed
        if os.path.exists(output_path): os.remove(output_path)
        raise ValueError("No valid poses detected in user video")
    
    user_sequence_np = np.array(user_sequence)
    trainer_sequence_np = analyzer.trainer_sequence
    
    # Run DTW
    distance, path = fastdtw(trainer_sequence_np, user_sequence_np, dist=euclidean)
    normalized_distance = distance / len(path) if len(path) > 0 else 0
    
    # Calculate Score
    max_expected_distance = 5.0 
    similarity_score = max(0, min(100, 100 * (1 - 3 * normalized_distance / max_expected_distance)))
    
    # Generate Feedback
    feature_analysis = analyzer._analyze_feature_differences(trainer_sequence_np, user_sequence_np, path)
    
    trainer_reps = analyzer.trainer_general_data.get('trainer_reps', 0)
    timing_analysis = analyzer._analyze_timing_coordination(trainer_sequence_np, user_sequence_np, path, reps, trainer_reps)
    patterns = analyzer._identify_movement_patterns(feature_analysis, timing_analysis)
    prompt = analyzer._generate_chatgpt_prompt(patterns, similarity_score, reps)
    feedback = analyzer.get_chatgpt_feedback(prompt)
    
    results = {
        'similarity_score': round(similarity_score, 1),
        'dtw_distance': round(distance, 2),
        'normalized_distance': round(normalized_distance, 2),
        'user_frames': len(user_sequence_np),
        'trainer_frames': len(trainer_sequence_np),
        'reps_counted': reps,
        'feedback': feedback
    }
    return results

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("Starting Flask server on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080)