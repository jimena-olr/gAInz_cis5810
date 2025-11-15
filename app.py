'''
**brew install ffmpeg**
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os

# Import the classes, loader, AND angle_3pts function
from dtw_analyzer import (
    DTWMovementAnalyzer, PoseDetector, FeatureExtractor, 
    load_trainer_data, angle_3pts 
)

app = Flask(__name__)
CORS(app)

# --- Global Variables ---
try:
    analyzer = DTWMovementAnalyzer(action="shoulder_press")
    current_directory = os.getcwd()
    file_name = "shoulder_press_trainer.pkl"
    full_path = os.path.join(current_directory, file_name)
    print(f"Loading data from full path: {full_path}")
    load_trainer_data(analyzer, full_path) 
    print("Trainer data loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Could not find '{file_name}'.")
    print("Please run dtw_analyzer.py first to generate this file.")
    exit()

detector = PoseDetector("blazepose")
extractor = FeatureExtractor("shoulder_press")

user_sequence_features = []
# --- NEW: Globals for rep counting ---
rep_counter = 0
rep_down_state = False
# ------------------------------------


@app.route("/start_session", methods=["POST"])
def start_session():
    """
    Clears the previous session data to start a new recording.
    """
    global user_sequence_features, rep_counter, rep_down_state
    user_sequence_features = []
    # --- NEW: Reset rep counters ---
    rep_counter = 0
    rep_down_state = False
    print("New session started. Feature list cleared.")
    return jsonify({"status": "session_started"})


@app.route("/process_frame", methods=["POST"])
def process_frame():
    """
    Receives a single frame, processes it, draws the pose,
    counts reps, and sends the drawn frame back.
    """
    global user_sequence_features, detector, extractor, rep_counter, rep_down_state
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image data"}), 400

    try:
        img_data = base64.b64decode(data["image"].split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return jsonify({"status": "image_error", "image": data["image"]})

    if frame is None:
        return jsonify({"status": "image_error", "image": data["image"]})

    pts = detector.infer(frame)
    image_to_send = data["image"] 
    feedback_status = "no_pose"

    if pts and len(pts) == 33:
        features = extractor.extract_features(pts)
        if features:
            user_sequence_features.append(features)
            feedback_status = "pose_detected"
        
        # --- NEW: Rep Counting Logic (from your dtw_analyzer)
        try:
            # Use the trainer's data to set rep thresholds
            avg_min_elbow = analyzer.trainer_general_data.get('avg_min_elbow', 90) # Default 90
            
            angle_r_elbow = angle_3pts(pts[11], pts[13], pts[15])
            
            if angle_r_elbow < avg_min_elbow + 15:
                rep_down_state = True
            # Check for "up" state, add a buffer to prevent bouncing
            if rep_down_state and angle_r_elbow > avg_min_elbow + 25: 
                rep_counter += 1
                rep_down_state = False
        except Exception as e:
            print(f"Error in rep counting: {e}")
        # --- End Rep Counting ---

        vis_frame = detector.draw(frame.copy(), pts, score_thresh=0.5)
        
        # Add rep count to the visual frame
        cv2.putText(vis_frame, f"Reps: {rep_counter}", (12, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        _, buffer = cv2.imencode('.jpg', vis_frame)
        vis_img_b64 = base64.b64encode(buffer).decode('utf-8')
        image_to_send = f"data:image/jpeg;base64,{vis_img_b64}"
        
    return jsonify({"status": feedback_status, "image": image_to_send})


@app.route("/analyze_session", methods=["POST"])
def analyze_session():
    """
    Analyzes the complete recorded session against the trainer's data.
    """
    global user_sequence_features, analyzer, rep_counter
    
    if analyzer.trainer_sequence is None:
        return jsonify({"error": "Trainer data not loaded"}), 500
        
    if len(user_sequence_features) < 20: 
        return jsonify({
            "error": "Not enough pose data collected. Please try recording for longer.",
            "similarity_score": 0,
            "feedback": ["Not enough data to analyze. Please record a full set of reps."]
        })

    user_sequence_np = np.array(user_sequence_features)
    trainer_sequence_np = analyzer.trainer_sequence
    
    print(f"Analyzing... User frames: {len(user_sequence_np)}, Trainer frames: {len(trainer_sequence_np)}")

    distance, path = fastdtw(trainer_sequence_np, user_sequence_np, dist=euclidean)
    normalized_distance = distance / len(path)
    
    max_expected_distance = 5.0 
    similarity_score = max(0, min(100, 100 * (1 - normalized_distance / max_expected_distance)))
    
    feedback = analyzer._generate_feedback(similarity_score, []) 
    
    # --- UPDATED: Send all data back ---
    results = {
        'similarity_score': round(similarity_score, 1),
        'dtw_distance': round(distance, 2),
        'normalized_distance': round(normalized_distance, 2),
        'user_frames': len(user_sequence_np),
        'trainer_frames': len(trainer_sequence_np),
        'reps_counted': rep_counter, # Send the reps we counted
        'feedback': feedback
    }
    # -----------------------------------
    
    print(f"Analysis complete. Score: {similarity_score}")
    
    # Clear the sequence for the next session
    user_sequence_features = []

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

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

# Import components from dtw_analyzer
from dtw_analyzer import (
    DTWMovementAnalyzer, PoseDetector, FeatureExtractor, 
    load_trainer_data, angle_3pts
)

app = Flask(__name__)
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = os.path.join('static', 'output')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 32 MB upload limit

# --- Global Variables ---
try:
    analyzer = DTWMovementAnalyzer(action="shoulder_press")
    full_path = os.path.join(os.getcwd(), "shoulder_press_trainer.pkl")
    print(f"Loading data from full path: {full_path}")
    load_trainer_data(analyzer, full_path) 
    print("Trainer data loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Could not find 'shoulder_press_trainer.pkl'.")
    print("Please run dtw_analyzer.py first to generate this file.")
    exit()

detector = PoseDetector("blazepose")
extractor = FeatureExtractor("shoulder_press")
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
        filename = f"user_video_{int(time.time())}.webm"
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
            return jsonify({"error": f"Failed to process video: {e}"}), 500

def process_video_file(input_path, output_path):
    """
    This function combines all the logic from dtw_analyzer's 
    main() and analyze_user_video() functions.
    """
    global analyzer, detector, extractor
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open user video at {input_path}")

    # Get video properties for output writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Handle edge cases
        fps = 20.0
        
    # --- Video Writer Setup ---
    # Using 'mp4v' for MP4 format, which is web-friendly
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise IOError(f"Could not open video writer at {output_path}")
    
    print(f"Writing analyzed video to {output_path} at {fps} FPS")

    user_sequence = []
    reps = 0
    down = False
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        pts = detector.infer(frame)
        vis_frame = frame.copy() # Start with the original frame

        if pts and len(pts) == 33:
            features = extractor.extract_features(pts)
            if features:
                user_sequence.append(features)
            
            # --- Rep Counting ---
            avg_min_elbow = analyzer.trainer_general_data.get('avg_min_elbow', 90)
            angle_r_elbow = angle_3pts(pts[11], pts[13], pts[15])
            if angle_r_elbow < avg_min_elbow + 15:
                down = True
            if down and angle_r_elbow > avg_min_elbow + 25:
                reps += 1
                down = False
            
            # --- Draw on the frame ---
            vis_frame = detector.draw(vis_frame, pts, score_thresh=0.5)
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
    normalized_distance = distance / len(path)
    max_expected_distance = 5.0 
    similarity_score = max(0, min(100, 100 * (1 - normalized_distance / max_expected_distance)))
    
    feedback = analyzer._generate_feedback(similarity_score, []) 
    
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
    # Ensure all our directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(host="0.0.0.0", port=8080)