import cv2
import mediapipe as mp
import numpy as np
from math import dist, atan2, degrees

class SignLanguageDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.labels = [chr(i) for i in range(65, 91)]  # A-Z alphabet
        self.prev_detections = []
        self.smoothing_window = 5
        
    def _get_landmark_coords(self, landmarks, idx):
        return (landmarks[idx].x, landmarks[idx].y)
        
    def _calculate_distance(self, p1, p2):
        return dist(p1, p2)
        
    def _calculate_angle(self, p1, p2, p3):
        # Convert all points to numpy arrays if they aren't already
        a = np.array([p1[0], p1[1]]) if isinstance(p1, tuple) else np.array([p1.x, p1.y])
        b = np.array([p2[0], p2[1]]) if isinstance(p2, tuple) else np.array([p2.x, p2.y])
        c = np.array([p3[0], p3[1]]) if isinstance(p3, tuple) else np.array([p3.x, p3.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return degrees(angle)
        
    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmarks.landmark
                
                # Get coordinates of key landmarks
                thumb_tip = self._get_landmark_coords(landmarks, 4)
                index_tip = self._get_landmark_coords(landmarks, 8)
                middle_tip = self._get_landmark_coords(landmarks, 12)
                ring_tip = self._get_landmark_coords(landmarks, 16)
                pinky_tip = self._get_landmark_coords(landmarks, 20)
                
                # Get base joints for angle calculations
                thumb_base = self._get_landmark_coords(landmarks, 2)
                index_base = self._get_landmark_coords(landmarks, 5)
                middle_base = self._get_landmark_coords(landmarks, 9)
                ring_base = self._get_landmark_coords(landmarks, 13)
                pinky_base = self._get_landmark_coords(landmarks, 17)
                
                # Calculate distances between key points
                thumb_index_dist = self._calculate_distance(thumb_tip, index_tip)
                index_middle_dist = self._calculate_distance(index_tip, middle_tip)
                middle_ring_dist = self._calculate_distance(middle_tip, ring_tip)
                ring_pinky_dist = self._calculate_distance(ring_tip, pinky_tip)
                
                # Calculate finger angles
                thumb_angle = self._calculate_angle(thumb_base, landmarks[3], thumb_tip)
                index_angle = self._calculate_angle(index_base, landmarks[6], index_tip)
                middle_angle = self._calculate_angle(middle_base, landmarks[10], middle_tip)
                ring_angle = self._calculate_angle(ring_base, landmarks[14], ring_tip)
                pinky_angle = self._calculate_angle(pinky_base, landmarks[18], pinky_tip)
                
                # A: Thumb across palm, other fingers closed
                if (thumb_angle > 160 and 
                    all(angle > 120 for angle in [index_angle, middle_angle, ring_angle, pinky_angle])):
                    return "A"
                
                # B: All fingers extended, thumb alongside
                elif (all(angle < 30 for angle in [index_angle, middle_angle, ring_angle, pinky_angle]) and
                      thumb_angle > 120):
                    return "B"
                
                # C: Curved hand shape
                elif (ring_pinky_dist > 0.25 and
                      abs(index_tip[1] - pinky_tip[1]) < 0.05):
                    return "C"
                
                # D: Index finger extended, others closed
                elif (index_angle < 30 and
                      all(angle > 120 for angle in [middle_angle, ring_angle, pinky_angle])):
                    return "D"
                
                # E: All fingers closed
                elif all(angle > 120 for angle in [index_angle, middle_angle, ring_angle, pinky_angle]):
                    return "E"
                
                # F: Thumb and index touching, others extended
                elif (thumb_index_dist < 0.05 and
                      all(angle < 30 for angle in [middle_angle, ring_angle, pinky_angle])):
                    return "F"
                
                # G: Index finger pointing, thumb alongside
                elif (index_angle < 30 and
                      thumb_index_dist > 0.15 and
                      all(angle > 120 for angle in [middle_angle, ring_angle, pinky_angle])):
                    return "G"
                
                # H: Index and middle fingers extended together
                elif (index_angle < 30 and
                      middle_angle < 30 and
                      index_middle_dist < 0.1 and
                      all(angle > 120 for angle in [ring_angle, pinky_angle])):
                    return "H"
                
                # I: Pinky extended, others closed
                elif (pinky_angle < 30 and
                      all(angle > 120 for angle in [index_angle, middle_angle, ring_angle])):
                    return "I"
                
                # J: Pinky extended with circular motion (special case)
                elif (pinky_angle < 30 and
                      all(angle > 120 for angle in [index_angle, middle_angle, ring_angle])):
                    return "J"
                
                # K: Index and middle extended, thumb between them
                elif (index_angle < 30 and
                      middle_angle < 30 and
                      thumb_angle < 60 and
                      thumb_tip[0] > index_tip[0] and
                      thumb_tip[0] < middle_tip[0]):
                    return "K"
                
                # L: Index and thumb extended, others closed
                elif (index_angle < 30 and
                      thumb_angle < 60 and
                      all(angle > 120 for angle in [middle_angle, ring_angle, pinky_angle])):
                    return "L"
                
                # M: Thumb under three closed fingers
                elif (thumb_angle > 160 and
                      all(angle > 120 for angle in [index_angle, middle_angle, ring_angle, pinky_angle])):
                    return "M"
                
                # N: Thumb under two closed fingers
                elif (thumb_angle > 160 and
                      index_angle > 120 and
                      middle_angle > 120 and
                      ring_angle < 30 and
                      pinky_angle < 30):
                    return "N"
                
                # O: Fingers curved to touch thumb
                elif (thumb_index_dist < 0.05 and
                      thumb_angle < 60 and
                      index_angle < 60):
                    return "O"
                
                # P: Index pointing down, thumb extended
                elif (index_angle > 160 and
                      thumb_angle < 60 and
                      all(angle > 120 for angle in [middle_angle, ring_angle, pinky_angle])):
                    return "P"
                
                # Q: Index pointing down, thumb alongside
                elif (index_angle > 160 and
                      thumb_tip[0] > thumb_base[0] and
                      all(angle > 120 for angle in [middle_angle, ring_angle, pinky_angle])):
                    return "Q"
                
                # R: Crossed index and middle fingers
                elif (index_angle < 30 and
                      middle_angle < 30 and
                      index_tip[0] > middle_tip[0]):
                    return "R"
                
                # S: Fist with thumb over fingers
                elif (thumb_angle < 60 and
                      all(angle > 120 for angle in [index_angle, middle_angle, ring_angle, pinky_angle])):
                    return "S"
                
                # T: Thumb between index and middle fingers
                elif (thumb_angle < 60 and
                      index_angle > 120 and
                      middle_angle > 120 and
                      thumb_tip[0] > index_tip[0] and
                      thumb_tip[0] < middle_tip[0]):
                    return "T"
                
                # U: Index and middle fingers extended together
                elif (index_angle < 30 and
                      middle_angle < 30 and
                      index_middle_dist < 0.1 and
                      all(angle > 120 for angle in [ring_angle, pinky_angle])):
                    return "U"
                
                # V: Index and middle fingers extended apart
                elif (index_angle < 30 and
                      middle_angle < 30 and
                      index_middle_dist > 0.15 and
                      all(angle > 120 for angle in [ring_angle, pinky_angle])):
                    return "V"
                
                # W: Index, middle and ring fingers extended
                elif (index_angle < 30 and
                      middle_angle < 30 and
                      ring_angle < 30 and
                      pinky_angle > 120):
                    return "W"
                
                # X: Index finger bent
                elif (index_angle > 160 and
                      all(angle < 30 for angle in [middle_angle, ring_angle, pinky_angle])):
                    return "X"
                
                # Y: Thumb and pinky extended
                elif (thumb_angle < 60 and
                      pinky_angle < 30 and
                      all(angle > 120 for angle in [index_angle, middle_angle, ring_angle])):
                    return "Y"
                
                # Z: Index finger drawing Z shape (special case)
                elif (index_angle < 30 and
                      all(angle > 120 for angle in [middle_angle, ring_angle, pinky_angle])):
                    return "Z"
                
                return None
        return None
