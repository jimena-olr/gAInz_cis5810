# 2) Imports
import cv2, numpy as np
from collections import deque

# 3) Configuration
MODEL = "blazepose"   # choose: "blazepose" or "movenet"
SOURCE = 0            # 0 = webcam; or a path string like "video.mp4"
DRAW_SKELETON = True
SCORE_THRESH = 0.5
print(f"Backend selected: {MODEL}, source: {SOURCE}")


# Pose Detector Model Initialiation and Inference
class PoseDetector:
    def __init__(self, backend="blazepose"):
        backend = backend.lower()
        self.backend = backend
        if backend == "blazepose":
            self._init_blazepose()
        elif backend == "movenet":
            self._init_movenet()
        else:
            raise ValueError("backend must be 'blazepose' or 'movenet'")

    # ---------- BlazePose (solutions API) ----------
    def _init_blazepose(self):
        # Import only the solutions submodule to avoid TensorFlow/tasks imports
        from mediapipe.python.solutions import pose as mp_pose
        self._mp_pose = mp_pose
        # model_complexity: 0=lite, 1=full, 2=heavy
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
        for lm in res.pose_landmarks.landmark:  # 33 landmarks
            x = float(lm.x) * w
            y = float(lm.y) * h
            score = float(getattr(lm, "visibility", 0.9))
            pts.append((x, y, score))
        return pts

    # ---------- MoveNet (TF Hub) ----------
    def _init_movenet(self):
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except Exception as e:
            raise RuntimeError(
                "MoveNet requires TensorFlow + tensorflow-hub. Install them before selecting MODEL='movenet'."
            ) from e
        self.tf = tf
        self.hub = hub
        self.input_size = 192  # Lightning; use 256 for Thunder
        self.model = self.hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self._edges = [
            (5,7),(7,9), (6,8),(8,10), (5,6),
            (5,11),(6,12), (11,12), (11,13),(13,15), (12,14),(14,16),
            (0,1),(0,2),(1,3),(2,4)
        ]

    def _infer_movenet(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tf = self.tf
        img_tf = tf.convert_to_tensor(img_rgb)
        img_tf = tf.image.resize_with_pad(img_tf, self.input_size, self.input_size)
        input_img = tf.cast(img_tf, dtype=tf.int32)[tf.newaxis, ...]  # (1,H,W,3)
        out = self.model.signatures["serving_default"](input_img)["output_0"].numpy()  # (1,1,17,3)
        kps = out[0,0,:,:]  # (y, x, score) normalized
        pts = []
        for (yy, xx, sc) in kps:
            x = float(xx) * w
            y = float(yy) * h
            pts.append((x, y, float(sc)))
        return pts

    # ---------- Unified API ----------
    def infer(self, frame_bgr):
        if self.backend == "blazepose":
            return self._infer_blazepose(frame_bgr)
        else:
            return self._infer_movenet(frame_bgr)

    def draw(self, frame_bgr, pts, score_thresh=0.5):
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
    

# Utility functions for pose analysis
def movement_magnitude(prev_pts, curr_pts):
    if not prev_pts or not curr_pts:
        return 0.0
    n = min(len(prev_pts), len(curr_pts))
    if n == 0: return 0.0
    disps = []
    for i in range(n):
        x1,y1,_ = prev_pts[i]
        x2,y2,_ = curr_pts[i]
        disps.append(((x2-x1)**2 + (y2-y1)**2) ** 0.5)
    return float(np.mean(disps)) if disps else 0.0

def angle_3pts(a, b, c):
    a = np.array(a[:2], float); b = np.array(b[:2], float); c = np.array(c[:2], float)
    ba, bc = a-b, c-b
    denom = (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def torso_scale(pts):
    try:
        # BlazePose indices: shoulders 11/12, hips 23/24
        # MoveNet indices:   shoulders 5/6,   hips 11/12
        if len(pts) >= 25:
            sL, sR, hL, hR = pts[11], pts[12], pts[23], pts[24]
        else:
            sL, sR, hL, hR = pts[5], pts[6], pts[11], pts[12]
        s_mid = ((sL[0]+sR[0])/2.0, (sL[1]+sR[1])/2.0)
        h_mid = ((hL[0]+hR[0])/2.0, (hL[1]+hR[1])/2.0)
        d = ((s_mid[0]-h_mid[0])**2 + (s_mid[1]-h_mid[1])**2)**0.5
        return max(d, 1.0)
    except:
        return 1.0

def main():
    # Sanity Check
    from mediapipe.python.solutions import pose as mp_pose
    p = mp_pose.Pose(); print('BlazePose solutions API imported OK')


    #Video Inference
    def run_demo(model_name, source=0):
        print(f"Starting with backend={model_name}, source={source}")
        detector = PoseDetector(model_name)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Could not open video source. Try SOURCE=0 for webcam or a valid file path.")
            return

        prev_pts = None
        mov_hist = deque(maxlen=30)
        LOW, HIGH = 70, 160  # squat-like thresholds
        down = False
        reps = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pts = detector.infer(frame)

            move = movement_magnitude(prev_pts, pts)
            scale = torso_scale(pts) if pts else 1.0
            move_norm = move / scale
            mov_hist.append(move_norm)
            smooth = float(np.mean(mov_hist)) if mov_hist else 0.0
            prev_pts = pts

            knee_angle = None
            if pts:
                if len(pts) >= 28:  # BlazePose
                    hip, knee, ankle = pts[23], pts[25], pts[27]
                else:               # MoveNet
                    hip, knee, ankle = pts[11], pts[13], pts[15]
                knee_angle = angle_3pts(hip, knee, ankle)
                if knee_angle < LOW:
                    down = True
                if down and knee_angle is not None and knee_angle > HIGH:
                    reps += 1
                    down = False

            vis = detector.draw(frame.copy(), pts, score_thresh=SCORE_THRESH)
            cv2.putText(vis, f"Movement (norm): {smooth:.2f}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if knee_angle is not None:
                cv2.putText(vis, f"Knee angle: {knee_angle:.1f} deg  Reps: {reps}",
                            (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow("Pose (ESC to quit)", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    # Run it using the config values
    source = "Videos\shoulder_press\shoulder_press_trainer.mp4"
    run_demo(MODEL, source)

if __name__ == "__main__":
    main()