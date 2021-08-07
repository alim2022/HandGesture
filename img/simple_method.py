#import
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import cv2
from IPython.display import clear_output
import pickle
from sklearn.neighbors import KNeighborsClassifier

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def simple_classification(hand_landmarks):
    finger_up = [False, False, False, False, False]
    
    if(hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y
       and hand_landmarks.landmark[3].y < hand_landmarks.landmark[2].y
       and hand_landmarks.landmark[2].y < hand_landmarks.landmark[1].y
       and hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y
       and hand_landmarks.landmark[3].y < hand_landmarks.landmark[6].y 
      ):
        finger_up[0] = True
    if(hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y
       and hand_landmarks.landmark[7].y < hand_landmarks.landmark[6].y
       and hand_landmarks.landmark[6].y < hand_landmarks.landmark[5].y
       and hand_landmarks.landmark[5].y < hand_landmarks.landmark[0].y
      ):
        finger_up[1] = True
    if(hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y
       and hand_landmarks.landmark[11].y < hand_landmarks.landmark[10].y
       and hand_landmarks.landmark[10].y < hand_landmarks.landmark[9].y
       and hand_landmarks.landmark[9].y < hand_landmarks.landmark[0].y\
      ):
        finger_up[2] = True
    if(hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y
       and hand_landmarks.landmark[15].y < hand_landmarks.landmark[14].y
       and hand_landmarks.landmark[14].y < hand_landmarks.landmark[13].y
       and hand_landmarks.landmark[13].y < hand_landmarks.landmark[0].y
      ):
        finger_up[3] = True
    if(hand_landmarks.landmark[20].y < hand_landmarks.landmark[19].y 
       and hand_landmarks.landmark[19].y < hand_landmarks.landmark[18].y 
       and hand_landmarks.landmark[18].y < hand_landmarks.landmark[17].y
       and hand_landmarks.landmark[17].y < hand_landmarks.landmark[0].y
      ):
        finger_up[4] = True
    
    #print(finger_up)
    
    if(finger_up[1] and not finger_up[2] and not finger_up[3] and not finger_up[4]):
        return "one"
    elif(finger_up[1] and finger_up[2] and not finger_up[3] and not finger_up[4]):
        return "two"
    elif(finger_up[1] and finger_up[2] and finger_up[3] and not finger_up[4]):
        return "three"
    elif(finger_up[0] and not finger_up[1] and not finger_up[2] and not finger_up[3] and not finger_up[4]):
        return "thumb up"
    else:
        return "hand raise"


def show_gesture(gesture):
        img_name = gesture.replace(' ', '_') + ".png"
        img = cv2.imread(f"img/{img_name}")
        cv2.imshow("emoji", img)




if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False , max_num_hands = 1 , min_detection_confidence=0.7, min_tracking_confidence=0.5)

    history_len = 10
    gesture_history = ["none"]*history_len

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("read fail")
            continue
            
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks is not None:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = simple_classification(hand_landmarks)
        else:
            gesture = "none"
            
        gesture_history.append(gesture)
        gesture_history = gesture_history[1:]
        max_gesture = "none"
        max_cnt = 0
        for gesture_name in ["one" , "two" , "three" , "thumb up" , "thumb down" , "hand raise" , "none"]:
            cnt = gesture_history.count(gesture_name)
            if max_cnt < cnt:
                max_cnt = cnt
                max_gesture = gesture_name
        gesture = max_gesture
        
        cv2.putText(img, gesture , org=(50,50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color = (0,0,255))
        show_gesture(gesture)
        
        cv2.imshow("test" , img)
        if cv2.waitKey(5)==27:
            break
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cap.release()
