import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
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
    img = cv2.convertScaleAbs(img, alpha=1.0, beta=20)  # increase brightness slightly

    # 7. Contrast Adjustment
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=0)  # increase contrast slightly

    # 8. Gamma Correction
    gamma = 1.2
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, look_up_table)

    # 9. Sharpening Filter
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    # 10. Resize to 300x300 (required by your dataset code)
    img = cv2.resize(img, (300, 300))

    return img


print(pickle)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR =  r"data"

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img=preprocess_image(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            if(len(data_aux)==42):
                data.append(data_aux)
                labels.append(dir_)


save_dir = r"C:\Users\shrut\Downloads\sign-language-detector-python-master\sign-language-detector-python-master"
os.makedirs(save_dir, exist_ok=True)  # ensure folder exists

save_path = os.path.join(save_dir, "data.pickle")

print(f"Total samples: {len(data)}. Saving now...")
with open(save_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset saved successfully at {save_path}")