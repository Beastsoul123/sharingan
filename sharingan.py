import cv2
import numpy as np
import dlib
import math
import time
import random
import os
import sys
import urllib.request
import bz2
from scipy.spatial import distance as dist

class UltimateSharingan:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(self.model_path):
            print("Downloading facial landmark model...")
            self.download_model()
        try:
            self.predictor = dlib.shape_predictor(self.model_path)
        except Exception as e:
            print(f"Error loading shape predictor model: {e}")
            sys.exit(1)
        self.eye_state = "open"
        self.last_blink_time = time.time()
        self.blink_cooldown = 0.5
        self.sharigan_active = False
        self.activation_time = 0
        self.tomoe_count = 3
        self.mangekyo_pattern = 0
        self.rotation_angle = 0
        self.pupil_dilation = 1.0
        self.blood_effect = 0.0
        self.glow_effect = 0.0
        self.gaze_direction = (0, 0)
        self.emotion = "neutral"
        self.emotion_intensity = 0.0
        self.performance_mode = "high"
        self.frame_skip = 0
        self.frame_counter = 0
        self.recording = False
        self.video_writer = None
        self.genjutsu_active = False
        self.genjutsu_type = "none"
        self.mangekyo_patterns = self.load_mangekyo_patterns()
        
    def download_model(self):
        model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = self.model_path + ".bz2"
        try:
            urllib.request.urlretrieve(model_url, compressed_path)
            with open(self.model_path, 'wb') as new_file, open(compressed_path, 'rb') as file:
                decompressor = bz2.BZ2Decompressor()
                for data in iter(lambda: file.read(100 * 1024), b''):
                    new_file.write(decompressor.decompress(data))
            os.remove(compressed_path)
            print("Model downloaded and extracted successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please manually download the model from:")
            print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Extract it and place in the same directory as this script.")
            sys.exit(1)
    
    def load_mangekyo_patterns(self):
        patterns = {
            0: {"name": "Standard", "complexity": 1, "color": (0, 0, 255)},
            1: {"name": "Itachi", "complexity": 3, "color": (0, 0, 200)},
            2: {"name": "Sasuke", "complexity": 3, "color": (0, 0, 220)},
            3: {"name": "Kakashi", "complexity": 2, "color": (0, 0, 240)},
            4: {"name": "Madara", "complexity": 4, "color": (0, 0, 180)},
        }
        return patterns
    
    def get_eye_aspect_ratio(self, eye_points):
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C)
    
    def get_eye_center_and_radius(self, landmarks, eye_indices):
        points = []
        for i in eye_indices:
            point = (landmarks.part(i).x, landmarks.part(i).y)
            points.append(point)
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        center = (sum(x) // len(points), sum(y) // len(points))
        width = abs(points[3][0] - points[0][0])
        height = abs(points[1][1] - points[5][1])
        radius = int((width + height) / 4)
        return center, radius, points
    
    def detect_gaze_direction(self, landmarks, eye_points, eye_center):
        iris_x = (landmarks.part(eye_points[0]).x + landmarks.part(eye_points[3]).x) // 2
        iris_y = (landmarks.part(eye_points[1]).y + landmarks.part(eye_points[5]).y) // 2
        dx = (iris_x - eye_center[0]) / max(1, eye_center[0] * 0.5)
        dy = (iris_y - eye_center[1]) / max(1, eye_center[1] * 0.5)
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))
        return (dx, dy)
    
    def detect_emotion(self, landmarks):
        try:
            mouth_width = abs(landmarks.part(54).x - landmarks.part(48).x)
            mouth_height = abs(landmarks.part(57).y - landmarks.part(51).y)
            eyebrow_height = (landmarks.part(19).y + landmarks.part(24).y) / 2
            nose_ref = landmarks.part(27).y
            emotion = "neutral"
            intensity = 0.0
            if mouth_width > 30 and eyebrow_height < nose_ref:
                emotion = "happy"
                intensity = min(1.0, mouth_width / 50)
            elif mouth_height > 15 and eyebrow_height > nose_ref:
                emotion = "angry"
                intensity = min(1.0, mouth_height / 25)
            elif eyebrow_height < nose_ref - 10:
                emotion = "surprised"
                intensity = min(1.0, (nose_ref - eyebrow_height) / 20)
        except:
            emotion = "neutral"
            intensity = 0.0
        return emotion, intensity
    
    def draw_mangekyo_pattern(self, frame, center, size, angle, pattern_type):
        x, y = center
        if pattern_type == 0:
            self.draw_animated_tomoe(frame, center, size, angle, self.tomoe_count)
        elif pattern_type == 1:
            for i in range(3):
                current_angle = angle + i * (2 * math.pi / 3)
                for j in range(2):
                    arm_angle = current_angle + j * (math.pi / 3)
                    end_x = int(x + size * 0.8 * math.cos(arm_angle))
                    end_y = int(y + size * 0.8 * math.sin(arm_angle))
                    cv2.line(frame, center, (end_x, end_y), (0, 0, 0), 3)
        elif pattern_type == 2:
            for i in range(6):
                current_angle = angle + i * (math.pi / 3)
                end_x = int(x + size * 0.7 * math.cos(current_angle))
                end_y = int(y + size * 0.7 * math.sin(current_angle))
                cv2.line(frame, center, (end_x, end_y), (0, 0, 0), 3)
            cv2.circle(frame, center, int(size * 0.3), (0, 0, 0), 2)
        cv2.circle(frame, center, int(size * 0.2), (0, 0, 0), -1)
        highlight_x = int(x + size * 0.1)
        highlight_y = int(y - size * 0.1)
        cv2.circle(frame, (highlight_x, highlight_y), int(size * 0.05), (255, 255, 255), -1)
    
    def draw_animated_tomoe(self, frame, center, size, angle, tomoe_num=3, filled=True):
        x, y = center
        for i in range(tomoe_num):
            current_angle = angle + i * (2 * math.pi / tomoe_num)
            tomoe_x = int(x + (size * 0.6) * math.cos(current_angle))
            tomoe_y = int(y + (size * 0.6) * math.sin(current_angle))
            if filled:
                cv2.circle(frame, (tomoe_x, tomoe_y), size // 3, (0, 0, 0), -1)
            highlight_x = int(tomoe_x + size * 0.1 * math.cos(current_angle + math.pi/4))
            highlight_y = int(tomoe_y - size * 0.1 * math.sin(current_angle + math.pi/4))
            cv2.circle(frame, (highlight_x, highlight_y), size // 6, (255, 255, 255), -1)
            tail_length = size // 2
            tail_x = int(tomoe_x + tail_length * math.cos(current_angle + math.pi/2))
            tail_y = int(tomoe_y + tail_length * math.sin(current_angle + math.pi/2))
            cv2.line(frame, (tomoe_x, tomoe_y), (tail_x, tail_y), (0, 0, 0), 2)
    
    def apply_genjutsu_effect(self, frame, genjutsu_type):
        if genjutsu_type == "tsukuyomi":
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 100), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            center = (w // 2, h // 2)
            for i in range(10, 0, -1):
                radius = int(min(w, h) * i / 10)
                cv2.circle(frame, center, radius, (0, 0, 0), 2)
        elif genjutsu_type == "kotoamatsukami":
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (100, 50, 0), -1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame
    
    def apply_glow_effect(self, frame, center, radius, intensity):
        glow_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(glow_mask, center, radius + 10, 255, -1)
        glow_mask = cv2.GaussianBlur(glow_mask, (15, 15), 0)
        glow = np.zeros_like(frame)
        glow[glow_mask > 0] = (0, 0, 255 * intensity)
        frame = cv2.addWeighted(frame, 1.0, glow, 0.5, 0)
        return frame
    
    def start_recording(self, filename="sharingan_recording.avi"):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
        self.recording = True
        print(f"Started recording to {filename}")
    
    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        print("Recording stopped")
    
    def draw_sharingan(self, frame, center, radius, rotation_angle, eye_state="open"):
        x, y = center
        if eye_state == "closed":
            cv2.line(frame, (x - radius, y), (x + radius, y), (0, 0, 0), 2)
            return frame
        elif eye_state == "half":
            radius = int(radius * 0.7)
        cv2.circle(frame, center, radius, (255, 255, 255), -1)
        if self.blood_effect > 0:
            for i in range(5):
                angle = rotation_angle + i * (2 * math.pi / 5)
                length = radius * (0.7 + 0.3 * self.blood_effect)
                end_x = int(x + length * math.cos(angle))
                end_y = int(y + length * math.sin(angle))
                cv2.line(frame, center, (end_x, end_y), (0, 0, 200), 1)
        for r in range(radius - 5, radius - 15, -1):
            intensity = int(255 * (r / (radius - 5)))
            cv2.circle(frame, center, r, (0, 0, intensity), 1)
        pupil_offset_x = int(self.gaze_direction[0] * radius * 0.3)
        pupil_offset_y = int(self.gaze_direction[1] * radius * 0.3)
        pupil_center = (x + pupil_offset_x, y + pupil_offset_y)
        pupil_size = int((radius // 2) * self.pupil_dilation)
        cv2.circle(frame, pupil_center, pupil_size, (0, 0, 0), -1)
        highlight_x = pupil_center[0] + pupil_size // 3
        highlight_y = pupil_center[1] - pupil_size // 3
        cv2.circle(frame, (highlight_x, highlight_y), pupil_size // 4, (255, 255, 255), -1)
        if self.sharigan_active and self.mangekyo_pattern > 0:
            self.draw_mangekyo_pattern(frame, pupil_center, radius, rotation_angle, self.mangekyo_pattern)
        elif self.sharigan_active:
            self.draw_animated_tomoe(frame, pupil_center, radius, rotation_angle, self.tomoe_count)
        cv2.circle(frame, center, radius, (0, 0, 0), 2)
        if eye_state == "half":
            cv2.ellipse(frame, center, (radius, radius//2), 0, 0, 180, (0, 0, 0), 2)
        if self.glow_effect > 0:
            frame = self.apply_glow_effect(frame, center, radius, self.glow_effect)
        return frame
    
    def update_eye_state(self, left_ear, right_ear):
        current_time = time.time()
        if left_ear < 0.2 and right_ear < 0.2:
            if current_time - self.last_blink_time > self.blink_cooldown:
                self.eye_state = "closed"
                self.last_blink_time = current_time
                if current_time - self.last_blink_time < 1.0:
                    self.sharigan_active = not self.sharigan_active
                    if self.sharigan_active:
                        self.activation_time = current_time
                        self.blood_effect = 1.0
                        self.glow_effect = 1.0
        elif left_ear < 0.25 or right_ear < 0.25:
            self.eye_state = "half"
        else:
            self.eye_state = "open"
        if self.sharigan_active:
            self.blood_effect = max(0, self.blood_effect - 0.02)
            self.glow_effect = max(0.3, self.glow_effect - 0.01)
        else:
            self.glow_effect = max(0, self.glow_effect - 0.05)
        if random.random() < 0.01:
            self.pupil_dilation = 0.8 + 0.4 * random.random()
    
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cv2.namedWindow("Ultimate Sharingan", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ultimate Sharingan", 800, 600)
        print("Ultimate Sharingan Controls:")
        print("- Double blink: Toggle Sharingan activation")
        print("- Press '1'-'5': Change Mangekyo pattern")
        print("- Press 'g': Toggle Genjutsu effects")
        print("- Press 'r': Start/stop recording")
        print("- Press 'p': Change performance mode")
        print("- Press 'q': Quit")
        while True:
            if self.performance_mode == "low" and self.frame_counter % 3 != 0:
                self.frame_counter += 1
                continue
            ret, frame = cap.read()
            if not ret:
                break
            if self.performance_mode == "low":
                frame = cv2.resize(frame, (320, 240))
            elif self.performance_mode == "medium":
                frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            if self.genjutsu_active:
                frame = self.apply_genjutsu_effect(frame, self.genjutsu_type)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            for face in faces:
                try:
                    landmarks = self.predictor(gray, face)
                    left_eye_center, left_eye_radius, left_eye_points = self.get_eye_center_and_radius(landmarks, range(36, 42))
                    right_eye_center, right_eye_radius, right_eye_points = self.get_eye_center_and_radius(landmarks, range(42, 48))
                    left_ear = self.get_eye_aspect_ratio(left_eye_points)
                    right_ear = self.get_eye_aspect_ratio(right_eye_points)
                    left_gaze = self.detect_gaze_direction(landmarks, range(36, 42), left_eye_center)
                    right_gaze = self.detect_gaze_direction(landmarks, range(42, 48), right_eye_center)
                    self.gaze_direction = ((left_gaze[0] + right_gaze[0]) / 2, 
                                          (left_gaze[1] + right_gaze[1]) / 2)
                    self.emotion, self.emotion_intensity = self.detect_emotion(landmarks)
                    self.update_eye_state(left_ear, right_ear)
                    frame = self.draw_sharingan(frame, left_eye_center, left_eye_radius, self.rotation_angle, self.eye_state)
                    frame = self.draw_sharingan(frame, right_eye_center, right_eye_radius, self.rotation_angle, self.eye_state)
                except Exception as e:
                    continue
            self.rotation_angle += 0.03
            status = f"Sharingan: {'ACTIVE' if self.sharigan_active else 'INACTIVE'}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            mangekyo_name = self.mangekyo_patterns[self.mangekyo_pattern]["name"]
            cv2.putText(frame, f"Pattern: {mangekyo_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Emotion: {self.emotion} ({self.emotion_intensity:.2f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            perf_mode = f"Performance: {self.performance_mode}"
            cv2.putText(frame, perf_mode, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.recording:
                cv2.putText(frame, "RECORDING", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.genjutsu_active:
                cv2.putText(frame, f"GENJUTSU: {self.genjutsu_type.upper()}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.recording and self.video_writer:
                self.video_writer.write(frame)
            cv2.imshow("Ultimate Sharingan", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in [ord(str(i)) for i in range(1, 6)]:
                self.mangekyo_pattern = int(chr(key)) - 1
            elif key == ord('g'):
                if not self.genjutsu_active:
                    self.genjutsu_active = True
                    self.genjutsu_type = random.choice(["tsukuyomi", "kotoamatsukami"])
                else:
                    self.genjutsu_active = False
            elif key == ord('r'):
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording()
            elif key == ord('p'):
                modes = ["high", "medium", "low"]
                current_idx = modes.index(self.performance_mode)
                self.performance_mode = modes[(current_idx + 1) % len(modes)]
            self.frame_counter += 1
        if self.recording:
            self.stop_recording()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sharingan = UltimateSharingan()
    sharingan.run()