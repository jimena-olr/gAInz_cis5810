'''
**brew install ffmpeg**
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
    
except Exception as e:
    print(f"FATAL ERROR during initialization: {e}")
    print("Please ensure 'static/shoulder_press_trainer.mp4' exists and 'mediapipe' is installed.")
    exit()
# ------------------------

# Main endpoint to serve the index.html page
@app.route('/')
def index():
    # Renders the index.html file
    return render_template('index.html')

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
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # 1. Save the user's raw video
        # Use secure_filename for safety
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
    app.run(host="0.0.0.0", port=8080)