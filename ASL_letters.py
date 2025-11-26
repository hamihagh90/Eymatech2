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

# -------------------------------------------------------------
# 🖐 استخراج ویژگی‌های دو دست (نرمال‌سازی‌شده)
# -------------------------------------------------------------
def extract_hand_keypoints(results):
    def process_hand(hand_landmarks):
        if not hand_landmarks:
            return np.zeros(21 * 3)
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        wrist = coords[0]
        coords -= wrist
        base_length = np.linalg.norm(coords[9])
        if base_length > 0:
            coords /= base_length
        coords[:, 2] *= 0.3
        return coords.flatten()

    left_hand = process_hand(results.multi_hand_landmarks[0]) if results.multi_hand_landmarks else np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)

    # اگر دو دست شناسایی شدند
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        right_hand = process_hand(results.multi_hand_landmarks[1])

    features = np.concatenate([left_hand, right_hand])
    mean = np.mean(features)
    std = np.std(features) if np.std(features) != 0 else 1e-6
    normalized_features = (features - mean) / std
    return normalized_features


# -------------------------------------------------------------
# 📦 تبدیل داده‌ها به آرایه
# -------------------------------------------------------------
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
                        features = extract_hand_keypoints(results)
                        all_data_X.append(features)
                        all_data_Y.append(labels.index(folder))

    return np.array(all_data_X), np.array(all_data_Y)


# -------------------------------------------------------------
# 🎯 آموزش مدل
# -------------------------------------------------------------
# dataX, dataY = data_to_array(path_data)
# print(labels)

# timesteps = 30
# features = 126

# # ساخت توالی‌ها
# sequences, labels_seq = [], []
# for i in range(len(dataX) - timesteps + 1):
#     sequences.append(dataX[i:i+timesteps])
#     labels_seq.append(dataY[i+timesteps-1])

# dataX_seq = np.array(sequences)
# dataY_seq = np.array(labels_seq)

# model = Sequential([
#     LSTM(256, return_sequences=True, activation='relu', input_shape=(timesteps, features)),
#     LSTM(128, return_sequences=False, activation='relu'),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dense(len(labels), activation='softmax')
# ])

# model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(dataX_seq, dataY_seq, epochs=50, batch_size=32, verbose=1)
# model.save("hand_model_optimized.h5")
model = load_model("hand_model_optimized.h5")


# -------------------------------------------------------------
# 🎥 اجرای زنده با وب‌کم (فقط دست‌ها)
# -------------------------------------------------------------
cap = cv2.VideoCapture(0)
sequence_buffer = []
cooldown_frames = 5  # تعداد فریم‌هایی که نمایش داده نمی‌شود
frame_counter = 0
last_label = 'B'

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        features = extract_hand_keypoints(results)
        sequence_buffer.append(features)
        if len(sequence_buffer) > 30:
            sequence_buffer.pop(0)

        if len(sequence_buffer) == 30:
            input_seq = np.expand_dims(sequence_buffer, axis=0)
            prediction = model.predict(input_seq, verbose=0)
            pred_class = np.argmax(prediction)
            current_label = labels[pred_class]

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