import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class HandTracker:
    def __init__(self, maxHands=2, detection_conf=0.7, tracking_conf=0.7):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )

    def get_hands(self, frame):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        all_hands = []

        if results.multi_hand_landmarks:
            for hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                handType = handedness.classification[0].label  # "Left" or "Right"

                # Tip of index finger = landmark 8
                h, w, c = frame.shape
                x = int(hand.landmark[8].x * w)
                y = int(hand.landmark[8].y * h)

                # Thumb tip 4, index tip 8 → distance small = thumbs up
                thumb_tip = hand.landmark[4]
                index_tip = hand.landmark[8]
                dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
                thumbs_up = dist < 0.07  # tuned value

                all_hands.append({
                    "type": handType,
                    "fingertip": (x, y),
                    "thumbs_up": thumbs_up,
                    "landmarks": hand
                })

        return all_hands
