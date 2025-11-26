import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

mp_hands = mp.solutions.hands  # ✅ فقط دست‌ها
mp_drawing = mp.solutions.drawing_utils

path_data = r"C:\Users\Hami H\Desktop\project\.asl_alphabet_train"
labels = []

class Model:
    def __init__(self):
        self.model = None

    def extract_hand_feature_vector(self, results):
        """
        ورودی:
            results: خروجی mediapipe Hands (hand_landmarks, handedness)
        خروجی:
            feature_vector: آرایه numpy با شکل (134,)
                            شامل مختصات نرمال‌شده و زوایای بین انگشتان هر دو دست
        """

        def normalize_hand(hand_landmarks, hand_label):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # --- 1. انتقال مبدأ به مچ ---
            wrist = coords[0].copy()
            coords -= wrist

            # --- 2. ساخت محورهای کف دست ---
            index_mcp = coords[5]
            pinky_mcp = coords[17]
            palm_normal = np.cross(index_mcp, pinky_mcp)
            palm_normal /= np.linalg.norm(palm_normal) + 1e-9

            x_axis = index_mcp / (np.linalg.norm(index_mcp) + 1e-9)
            y_axis = np.cross(palm_normal, x_axis)
            y_axis /= np.linalg.norm(y_axis) + 1e-9
            z_axis = np.cross(x_axis, y_axis)
            z_axis /= np.linalg.norm(z_axis) + 1e-9

            R = np.vstack([x_axis, y_axis, z_axis]).T
            coords = coords @ R

            # --- 3. نرمال‌سازی اندازه ---
            scale = np.linalg.norm(coords[9])  # wrist تا middle_mcp
            coords /= (scale + 1e-9)

            # --- 4. آینه‌سازی برای دست چپ ---
            if hand_label.lower() == "left":
                coords[:, 0] *= -1

            return coords


        def compute_finger_angles(coords):
            """محاسبه زوایای بین انگشتان اصلی"""
            ids = [4, 8, 12, 16, 20]  # انتهای انگشتان
            vecs = [coords[i] - coords[0] for i in ids]
            angles = []
            for i in range(len(vecs) - 1):
                v1, v2 = vecs[i], vecs[i + 1]
                dot = np.dot(v1, v2)
                norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
                cos_theta = np.clip(dot / norm, -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cos_theta)))
            return np.array(angles, dtype=float)

        # مقادیر پیش‌فرض برای زمانی که دست دیده نشود
        left_coords = np.zeros((21, 3))
        right_coords = np.zeros((21, 3))
        left_angles = np.zeros(4)
        right_angles = np.zeros(4)

        if results and getattr(results, "multi_hand_landmarks", None):
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # --- گرفتن label دست ---
                hand_label = results.multi_handedness[idx].classification[0].label  # 'Left' یا 'Right'

                # --- نرمال‌سازی ---
                coords = normalize_hand(hand_landmarks, hand_label)

                # --- محاسبه زاویه‌ها ---
                angles = compute_finger_angles(coords)

                # --- جایگذاری در خروجی ---
                if hand_label == "Left":
                    left_coords, left_angles = coords, angles
                else:
                    right_coords, right_angles = coords, angles

        # تخت‌سازی برای مدل
        feature_vector = np.concatenate([
            left_coords.flatten(), left_angles,
            right_coords.flatten(), right_angles
        ])

        return feature_vector
    
    def train(self, dataX_seq, dataY_seq, timesteps=30, features=126):
        self.model = Sequential([
            LSTM(256, return_sequences=True, activation='relu', input_shape=(timesteps, features)),
            LSTM(128, return_sequences=False, activation='relu'),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(len(labels), activation='softmax')
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        self.model.fit(dataX_seq, dataY_seq, epochs=30, batch_size=32, verbose=1)
        self.model.save("hand_model_optimized.h5")
        # self.model = load_model("hand_model_optimized.h5")
        return
    def predict(self, input_seq):
        prediction = self.model.predict(input_seq, verbose=0)
        pred_class = np.argmax(prediction)
        return labels[pred_class]


# -------------------------------------------------------------
# 🎯 آموزش مدل
# -------------------------------------------------------------
ASL = Model()
def data_to_array(path_data):
        global labels
        all_data_X, all_data_Y = [], []

        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
            for folder in os.listdir(path_data):
                folder_path = os.path.join(path_data, folder)
                if not os.path.isdir(folder_path):
                    continue
                if folder not in ['del', 'space', 'nothing']:
                    if folder not in labels:
                        labels.append(folder)
                    for filename in os.listdir(folder_path):
                        if int(filename[1:-4]) <= 100:
                            file_path = os.path.join(folder_path, filename)
                            image = cv2.imread(file_path)
                            if image is None:
                                continue
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            results = hands.process(image_rgb)
                            features = ASL.extract_hand_feature_vector(results)
                            all_data_X.append(features)
                            all_data_Y.append(labels.index(folder))

        return np.array(all_data_X), np.array(all_data_Y)

dataX, dataY = data_to_array(path_data)
print(labels)

timesteps = 30
features = 126

# ساخت توالی‌ها
sequences, labels_seq = [], []
for i in range(len(dataX) - timesteps + 1):
    sequences.append(dataX[i:i+timesteps])
    labels_seq.append(dataY[i+timesteps-1])

dataX_seq = np.array(sequences)
dataY_seq = np.array(labels_seq)
ASL.train(dataX_seq, dataY_seq)


# -------------------------------------------------------------
# 🎥 اجرای زنده با وب‌کم (فقط دست‌ها)
# -------------------------------------------------------------
cap = cv2.VideoCapture(0)
sequence_buffer = []
cooldown_frames = 5  # تعداد فریم‌هایی که نمایش داده نمی‌شود
frame_counter = 0
last_label = 'B'

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        features = ASL.extract_hand_keypoints(results)
        sequence_buffer.append(features)
        if len(sequence_buffer) > 30:
            sequence_buffer.pop(0)

        if len(sequence_buffer) == 30:
            input_seq = np.expand_dims(sequence_buffer, axis=0)
            pred_class = ASL.predict(input_seq)
            current_label = pred_class

            if current_label != last_label:
                frame_counter = cooldown_frames
                last_label = current_label
            if frame_counter > 0:
                frame_counter -= 1
                display_label = None  # نمایش نده
            else:
                display_label = current_label
            cv2.putText(frame, f'{display_label}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # رسم دست‌ها
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
                )

        cv2.imshow('ASL Detection (Hands Only)', frame)

        if cv2.getWindowProperty("ASL Detection (Hands Only)", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()