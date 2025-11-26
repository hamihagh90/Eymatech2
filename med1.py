import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
path_data = r"C:\Users\Hami H\Desktop\project\train"
labels = []

def extract_hand_keypoints(results):
    """
    این تابع نتایج مدل holistic را گرفته،
    نقاط دست راست و چپ را استخراج کرده،
    نرمال‌سازی کرده و در قالب یک آرایه numpy برمی‌گرداند.
    """

    # اگر دست راست پیدا شد:
    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()
    else:
        # اگر دستی شناسایی نشد، با صفر پر شود
        right_hand = np.zeros(21 * 3)

    # اگر دست چپ پیدا شد:
    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()
    else:
        left_hand = np.zeros(21 * 3)

    # ادغام دو دست در یک بردار (در مجموع 126 عدد)
    features = np.concatenate([left_hand, right_hand])

    # --- نرمال‌سازی ---
    # چون mediapipe مختصات را بین 0 تا 1 بر اساس تصویر می‌دهد،
    # می‌توانیم فقط مطمئن شویم میانگین صفر و واریانس یک دارند.
    mean = np.mean(features)
    std = np.std(features) if np.std(features) != 0 else 1e-6
    normalized_features = (features - mean) / std

    return normalized_features

def data_to_array(path_data):
    global labels
    """
    مسیر پوشه‌ی داده‌ها را گرفته،
    تمام تصاویر را پردازش کرده و
    آرایه‌ای از ویژگی‌های نرمال‌شده برمی‌گرداند.
    """
    all_data_X = []  # برای ذخیره ویژگی‌ها
    all_data_Y = []

    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        for folder in os.listdir(path_data):
            folder_path = os.path.join(path_data, folder)
            if not os.path.isdir(folder_path):
                continue
            if folder not in labels:
                labels.append(folder)

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                # خواندن تصویر
                image = cv2.imread(file_path)
                if image is None:
                    continue  # اگر فایل تصویر نبود، رد کن

                # RGB تبدیل به
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # پردازش توسط mediapipe
                results = holistic.process(image_rgb)

                # استخراج ویژگی‌ها
                features = extract_hand_keypoints(results)
                all_data_X.append(features)
                all_data_Y.append(labels.index(folder))

    return np.array(all_data_X), np.array(all_data_Y)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR-CONVERSION BGR-to-RGB
    image.flags.writeable = False                  # Convert image to not-writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Convert image to writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR-COVERSION RGB-to-BGR
    return image, results
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
dataX, dataY = data_to_array(path_data)
print(labels)

timesteps = 30
features = 126  # یا 63 اگر فقط یک دست

sequences = []
labels_seq = []

for i in range(len(dataX) - timesteps + 1):
    sequences.append(dataX[i:i+timesteps])
    labels_seq.append(dataY[i+timesteps-1])  # برچسب آخر توالی

dataX_seq = np.array(sequences)
dataY_seq = np.array(labels_seq)
print(dataX_seq.shape, dataY_seq.shape)  # باید (num_seq, 30, 126) و (num_seq,) باشد

model = Sequential([
    LSTM(128, return_sequences=True, activation='relu', input_shape=(timesteps, features)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    dataX_seq, dataY_seq,
    epochs=150,
    batch_size=32,
    verbose=1
)

model.save("hand_model_optimized.h5")

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # دست چپ
        
        sequence_buffer = []
        
        features = extract_hand_keypoints(results)
        sequence_buffer.append(features)

        if len(sequence_buffer) > 30:
            sequence_buffer.pop(0)

        if len(sequence_buffer) == 30:
            input_seq = np.expand_dims(sequence_buffer, axis=0)
            prediction = model.predict(input_seq)
            pred_class = np.argmax(prediction)
            # 🔹 نمایش نتیجه روی تصویر
            cv2.putText(image, f'{labels[pred_class]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.getWindowProperty("OpenCV Feed", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
draw_landmarks(frame, results)
cv2.imshow("result", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()