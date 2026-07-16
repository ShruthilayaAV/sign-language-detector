import pickle

import cv2
import mediapipe as mp
import numpy as np

def preprocess_image(img):
    # 1. Histogram Equalization (Y channel)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)

    # 2. CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)

    # 3. Gaussian Blur
    img = cv2.GaussianBlur(img, (5,5), 0)

    # 4. Median Blur
    img = cv2.medianBlur(img, 5)

    # 5. Bilateral Filter
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # 6. Brightness Adjustment
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)  # increase brightness slightly

    # 7. Contrast Adjustment
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=0)  # increase contrast slightly

    # 8. Gamma Correction
    gamma = 1.2
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, look_up_table)

    # 9. Sharpening Filter
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, 0.9, edges_colored, 0.15, 0)

    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    # 10. Resize to 300x300 (required by your dataset code)
    img = cv2.resize(img, (500, 500))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, 0.9, edges_colored, 0.1, 0)
    
    # 2. Mild color saturation boost
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.1)  # 10% boost
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

ref_img = cv2.imread("ref.jpg")
ref_img = cv2.resize(ref_img, (500, 500))

labels_dict = {0: 'A', 1: 'B', 2: 'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame=preprocess_image(frame)
    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    H, W, _ = frame.shape

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x1, y1 = int(min(x_coords)*W)-5, int(min(y_coords)*H)-5
                x2, y2 = int(max(x_coords)*W)+5, int(max(y_coords)*H)+5

            # Highlight hand region safely
                if x1 < x2 and y1 < y2 and x1 >=0 and y1 >=0:
                    hand_region = frame[y1:y2, x1:x2]
                    if hand_region.size > 0:
                        hand_region = cv2.convertScaleAbs(hand_region, alpha=1.1, beta=20)
                        frame[y1:y2, x1:x2] = hand_region

            # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Prepare input for prediction
                data_aux = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_coords))
                    data_aux.append(y - min(y_coords))


        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    combined = np.hstack((frame, ref_img))
    cv2.imshow('frame with reference', combined)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()   