'''
**brew install ffmpeg**
'''
'''
# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
import time
from werkzeug.utils import secure_filename

from dotenv import load_dotenv
load_dotenv() # Loads environment variables from .env file

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = os.path.join('static', 'output')

# Import components from dtw_analyzer
from dtw2 import (
    DTWMovementAnalyzer, PoseDetector, FeatureExtractor, 
    load_trainer_data, angle_3pts, align_skeleton
)

MODEL = "blazepose"         # The model name for PoseDetector
ACTION = "shoulder_press"   # The action name for Analyzer/Extractor
SCORE_THRESH = 0.5  


app = Flask(__name__)
CORS(app)

# --- Flask Configuration ---
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 32 MB upload limit

# --- Global Variables ---

def load_analyzers():
    actions = {
        "shoulder_press": "static/shoulder_press_trainer.mp4",
        "squat": "static/squat_trainer.mp4", 
        "jumping_jacks": "static/jumping_jacks_trainer.mp4"
    }
    
    print("Initializing analyzers...")
    for action, video_path in actions.items():
        if os.path.exists(video_path):
            print(f"Loading {action} from {video_path}...")
            analyzer = DTWMovementAnalyzer(action)
            analyzer.record_trainer_sequence(video_path)
            analyzers[action] = analyzer
        else:
            print(f"Warning: Trainer video for {action} not found at {video_path}")


try:
    print("Initializing components...")
    detector = PoseDetector(MODEL)
    extractor = FeatureExtractor(ACTION)
    analyzer = DTWMovementAnalyzer(ACTION)
    
    trainer_video_path = os.path.join("static", "shoulder_press_trainer.mp4")
    print(f"Loading trainer data from: {trainer_video_path}")
    if not os.path.exists(trainer_video_path):
        raise FileNotFoundError(f"Trainer video not found at {trainer_video_path}")
        
    analyzer.record_trainer_sequence(trainer_video_path)
    print("Trainer data loaded successfully.")

    analyzers = {}
    load_analyzers()
    print("Loaded analyzers")

    
except Exception as e:
    print(f"FATAL ERROR during initialization: {e}")
    print("Please ensure 'static/shoulder_press_trainer.mp4' exists and 'mediapipe' is installed.")
    exit()
# ------------------------

# Main endpoint to serve the index.html page
@app.route('/')
def index():
    # Renders the index.html file
    return send_from_directory('static', 'index.html')

# Endpoint to serve static video files (like the analyzed output)
@app.route('/static/output/<filename>')
def serve_output_video(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# Endpoint to serve the trainer video
@app.route('/static/<filename>')
def serve_trainer_video(filename):
    return send_from_directory('static', filename)


@app.route('/upload_and_analyze', methods=['POST'])
def upload_and_analyze():
    """
    Receives an uploaded video, processes it, and returns
    the analysis and a link to the new video.
    """
    exercise = request.form.get("exercise")
    if exercise is None:
        return jsonify({"error": "No exercise provided"}), 400
    
    if exercise not in analyzers:
        return jsonify({"error": f"Action '{exercise}' is not supported."}), 400

    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400


    if file:
        # 1. Save the user's raw video
        # Use secure_filename for safety
        current_analyzer = analyzers[exercise]
        base_filename = secure_filename(file.filename or 'user_video')
        filename = f"{os.path.splitext(base_filename)[0]}_{int(time.time())}.webm"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        print(f"User video saved to: {input_path}")

        # 2. Process the video and get analysis
        output_filename = f"analyzed_{int(time.time())}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        try:
            results = process_video_file(input_path, output_path)
            
            # 3. Add the URL to the results
            results['analyzed_video_url'] = f'/static/output/{output_filename}'
            results['trainer_video_url'] = f'/static/shoulder_press_trainer.mp4'
            
            return jsonify(results)
        
        except Exception as e:
            print(f"Error during video processing: {e}")
            # Import traceback to get more details
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Failed to process video: {str(e)}"}), 500

def process_video_file(input_path, output_path):
    """
    This function combines all the logic from dtw_analyzer's 
    main() and analyze_user_video() functions, using the new DTW2 classes.
    """
    global analyzer, detector, extractor
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open user video at {input_path}")

    # Get video properties for output writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 60: # Handle edge cases
        print(f"Warning: Invalid FPS ({fps}), defaulting to 20.0")
        fps = 20.0
        
    # --- Video Writer Setup ---
    # Using 'avc1' (H.264) for MP4 format, which is very web-friendly
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        # Fallback to mp4v if avc1 fails
        print("avc1 codec failed, falling back to mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not writer.isOpened():
            raise IOError(f"Could not open video writer at {output_path} with avc1 or mp4v")
    
    print(f"Writing analyzed video to {output_path} at {fps} FPS")

    user_sequence = []
    reps = 0
    down = False
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        pts_raw = detector.infer(frame)
        vis_frame = frame.copy() # Start with the original frame

        if pts_raw and len(pts_raw) == 33:
            
            # --- NEW: Align skeleton for feature extraction ---
            if analyzer.trainer_ref_pts is not None:
                pts_aligned = align_skeleton(pts_raw, analyzer.trainer_ref_pts)
            else:
                pts_aligned = pts_raw # Use raw points if no ref
            
            features = extractor.extract_features(pts_aligned)
            if features:
                user_sequence.append(features)
            
            # --- Rep Counting (using raw points) ---
            avg_min_elbow = analyzer.trainer_general_data.get('avg_min_elbow', 90)
            angle_r_elbow = angle_3pts(pts_raw[11], pts_raw[13], pts_raw[15])
            
            # Using the logic from your original app.py for rep counting
            if angle_r_elbow < avg_min_elbow + 15:
                down = True
            if down and angle_r_elbow > avg_min_elbow + 25:
                reps += 1
                down = False
            
            # --- Draw on the frame (using raw points) ---
            vis_frame = detector.draw(vis_frame, pts_raw, score_thresh=SCORE_THRESH)
            cv2.putText(vis_frame, f"Reps: {reps}", (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # Write the (possibly drawn) frame to the new video
        writer.write(vis_frame)

    cap.release()
    writer.release()
    print("Video processing and writing complete.")

    # --- DTW Analysis (after video is saved) ---
    if len(user_sequence) == 0:
        return {"error": "No valid poses detected in user video"}
    
    user_sequence_np = np.array(user_sequence)
    trainer_sequence_np = analyzer.trainer_sequence
    
    print(f"Analyzing... User frames: {len(user_sequence_np)}, Trainer frames: {len(trainer_sequence_np)}")

    distance, path = fastdtw(trainer_sequence_np, user_sequence_np, dist=euclidean)
    normalized_distance = distance / len(path) if len(path) > 0 else 0
    
    # max_expected_distance is a tuning parameter. 
    # 5.0 was from your new code, I'll keep it.
    max_expected_distance = 5.0 
    similarity_score = max(0, min(100, 100 * (1 - normalized_distance / max_expected_distance)))
    
    # --- NEW: AI Feedback Generation ---
    print("Generating detailed AI feedback...")
    
    # 1. Analyze feature differences
    feature_analysis = analyzer._analyze_feature_differences(trainer_sequence_np, user_sequence_np, path)
    
    # 2. Analyze timing
    timing_analysis = analyzer._analyze_timing_coordination(trainer_sequence_np, user_sequence_np, path)
    
    # 3. Identify patterns
    patterns = analyzer._identify_movement_patterns(feature_analysis, timing_analysis)
    
    # 4. Generate prompt
    prompt = analyzer._generate_chatgpt_prompt(patterns, similarity_score, reps)
    
    # 5. Get feedback from AI (or fallback)
    feedback = analyzer.get_chatgpt_feedback(prompt)
    
    # --- ---
    
    results = {
        'similarity_score': round(similarity_score, 1),
        'dtw_distance': round(distance, 2),
        'normalized_distance': round(normalized_distance, 2),
        'user_frames': len(user_sequence_np),
        'trainer_frames': len(trainer_sequence_np),
        'reps_counted': reps,
        'feedback': feedback,
        'analysis_details': { # Adding extra details for potential frontend use
            'feature_stats': feature_analysis.get('feature_stats', {}),
            'timing_stats': timing_analysis,
            'identified_patterns': patterns
        }
    }
    return results

if __name__ == "__main__":
    # Ensure all our directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    print("Starting Flask server on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080)'''

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
    "shoulder_press": "shoulder_press_trainer.mp4",
    "squat": "squat_trainer.mp4",
    "jumping_jacks": "jumping_jacks_trainer.mp4"
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
    similarity_score = max(0, min(100, 100 * (1 - normalized_distance / max_expected_distance)))
    
    # Generate Feedback
    feature_analysis = analyzer._analyze_feature_differences(trainer_sequence_np, user_sequence_np, path)
    timing_analysis = analyzer._analyze_timing_coordination(trainer_sequence_np, user_sequence_np, path)
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