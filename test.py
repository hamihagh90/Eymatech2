import cv2
import mediapipe as mp
import numpy as np

def extract_hand_feature_vector(results):
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


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        new_array = extract_hand_feature_vector(results)
        print(new_array.shape)  # باید (134,) باشد

        # رسم دست‌ها
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
                )

        cv2.imshow("Hand", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # خروج با ESC
            break

cap.release()
cv2.destroyAllWindows()
