import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

STABLE_FRAMES_REQUIRED = 10 
last_prediction = None
stable_frames= 0
letter_locked = False 
output_text = ""
MAX_CHARS = 19
NO_HAND_FRAMES_REQUIRED = 10
no_hand_frames = 0
space_locked = False
BACKSPACE_FRAMES_REQUIRED = 10
backspace_frames = 0
backspace_locked = False

model = joblib.load("asl_model.pkl")

mp_hands = mp.solutions.hands 
mp_draw = mp.solutions.drawing_utils

def is_open_hand(handLms):
    tips = [4, 8, 12, 16, 20]   # thumb + finger tips
    pips = [3, 6, 10, 14, 18]  # thumb IP + finger PIPs

    for tip, pip in zip(tips, pips):
        if handLms.landmark[tip].y > handLms.landmark[pip].y:
            return False
    thumb_tip = handLms.landmark[4]
    index_mcp = handLms.landmark[5]

    thumb_dist = abs(thumb_tip.x - index_mcp.x)
    if thumb_dist < 0.04:
        return False
    
    spread = abs(handLms.landmark[8].x - handLms.landmark[20].x)
    if spread < 0.12:
        return False
    return True

    

hands = mp_hands.Hands(
    static_image_mode =False, 
    max_num_hands = 1,
    min_detection_confidence = 0.7, 
    min_tracking_confidence = 0.7
)

prediction_buffer = deque(maxlen=7)

cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    success,img = cap.read()
    if not success: 
        break
    
    
    img = cv2.flip(img,1) 
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        no_hand_frames = 0
        space_locked = False
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            img,
            handLms,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = []
        for lm in handLms.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        X = np.array(landmarks).reshape(1,-1)

        if is_open_hand(handLms):
            backspace_frames += 1
            if backspace_frames >= BACKSPACE_FRAMES_REQUIRED and not backspace_locked:
                if len(output_text) > 0:
                    output_text = output_text[:-1]
                backspace_locked = True
                backspace_frames = 0
                prediction_buffer.clear()
            

        else:
            backspace_frames = 0
            backspace_locked = False
            

            prediction = model.predict(X)[0]
            prediction_buffer.append(prediction)

            final_prediction = max(
                set(prediction_buffer),
                key = prediction_buffer.count
            )
        
            if final_prediction == last_prediction:
                stable_frames += 1
            else:
                stable_frames =0 
                letter_locked = False

            last_prediction = final_prediction

            if stable_frames >= STABLE_FRAMES_REQUIRED and not letter_locked:
                output_text += final_prediction 
                output_text = output_text[-MAX_CHARS:]
                letter_locked = True

    else:
        no_hand_frames += 1
        stable_frames = 0
        letter_locked = False
        last_prediction = None
        prediction_buffer.clear()
        backspace_frames = 0
        backspace_locked = False

        if no_hand_frames >= NO_HAND_FRAMES_REQUIRED and not space_locked:
            if len(output_text) > 0 and output_text[-1] != " ":
                output_text += " "
                output_text = output_text[-MAX_CHARS:]
            space_locked = True
        

    cv2.putText(
            img,
            f"Text: {output_text}", 
            (30,80), 
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255,0,0),
            2
        )

    cv2.imshow("ASL Live Prediction", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()