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

class HandGesture:
    def __init__(self):
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        
        with open("new_training_dataset.pkl","rb") as f:
            dataset = pickle.load(f)
        training_inputs = []
        training_labels = []
        for data in dataset:
            angle_list, true_class = data
            training_inputs.append(angle_list)
            training_labels.append(true_class)
        self.classifier.fit(training_inputs, training_labels)
    
    def get_prediction(self, hand_landmarks):
        adjacency_list = [
            [0,1,2],[1,2,3],[2,3,4],[0,5,6],[5,6,7],[6,7,8],
            [0,9,10],[9,10,11],[10,11,12],[0,13,14],[13,14,15],
            [14,15,16],[0,17,18],[17,18,19],[18,19,20]
        ]

        angle_list=[]

        for adjacency in adjacency_list :
            idx0, idx1, idx2 = adjacency
            point0 = np.array([hand_landmarks.landmark[idx0].x, hand_landmarks.landmark[idx0].y, hand_landmarks.landmark[idx0].z])
            point1 = np.array([hand_landmarks.landmark[idx1].x, hand_landmarks.landmark[idx1].y, hand_landmarks.landmark[idx1].z])
            point2 = np.array([hand_landmarks.landmark[idx2].x, hand_landmarks.landmark[idx2].y, hand_landmarks.landmark[idx2].z])

            vector0 = point0 - point1
            vector1 = point2 - point1
            inner_product_result = np.dot(vector0,vector1)
            cos_theta = inner_product_result/(np.linalg.norm(vector0) * np.linalg.norm(vector1))
            theta = np.arccos(cos_theta)
            angle_list.append(theta)
            
        #predict_class = self.classifier.predict([angle_list])[0]
        predict_probs = self.classifier.predict_proba([angle_list])[0]
        
        predict_class = np.argmax(predict_probs)
        predict_prob = max(predict_probs)
        
        return predict_class, predict_prob
    
    def is_thumb_up(self, hand_landmarks):
        thumb_mcp_idx = 2
        thumb_tip_idx = 4
        thumb_mcp_y = hand_landmarks.landmark[thumb_mcp_idx].y
        thumb_tip_y = hand_landmarks.landmark[thumb_tip_idx].y
        finger_up = thumb_tip_y < thumb_mcp_y
        return finger_up
    
    def get_gesture(self, hand_landmarks):
        prediction, predict_prob = self.get_prediction(hand_landmarks)
        
        if predict_prob < 0.7:
            return "none"
        elif prediction == 0:
            return "one"
        elif prediction == 1:
            return "two"
        elif prediction == 2:
            return "three"
        elif prediction == 3:
            if self.is_thumb_up(hand_landmarks):
                return "thumb up"
            else:
                return "thumb down"
        elif prediction == 4:
            return "hand raise"

    def show_gesture(self, gesture):
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
            gesture = hand_gesture.get_gesture(hand_landmarks)
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
        hand_gesture.show_gesture(gesture)
        
        cv2.imshow("test" , img)
        if cv2.waitKey(5)==27:
            break
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cap.release()
