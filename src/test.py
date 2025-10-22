import cv2
import mediapipe as mp
#from model import model, predict_image_from_directory

#==================================================================#

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

#==================================================================#

def process_image(frame, target_size=(64, 64)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    normalized = resized / 255.0
    return normalized

#==================================================================#

def process_image_mediapipe(frame, target_size=(64, 64)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)
    rgb_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)
    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    if not results.multi_hand_landmarks:
        print("Aucune main détectée dans l'image.")
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    h, w, _ = frame.shape
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

    if not x_coords or not y_coords:
        print("Erreur : impossible de récupérer les coordonnées des landmarks.")
        return None

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    if x_min >= x_max or y_min >= y_max:
        print("Erreur : les coordonnées de la main sont invalides.")
        return None

    hand_width = x_max - x_min
    hand_height = y_max - y_min

    if hand_width <= 0 or hand_height <= 0:
        print("Erreur : la taille de la main détectée est nulle ou négative.")
        return None

    margin_x = int(0.20 * hand_width)
    margin_y = int(0.20 * hand_height)

    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(w, x_max + margin_x)
    y_max = min(h, y_max + margin_y)

    hand_roi = frame[y_min:y_max, x_min:x_max]  # Utiliser l'image originale pour la ROI
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    normalized = resized / 255.0
    return normalized

#==================================================================#

def save_processed_image(input_image_path, output_image_path, target_size=(64, 64)):
    # Charger l'image
    frame = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Erreur : impossible de charger l'image {input_image_path}")
        return False

    processed_img = process_image_mediapipe(frame, target_size)
    if processed_img is None:
        print("Erreur : le traitement a échoué.")
        return False

    processed_img_uint8 = (processed_img * 255).astype('uint8')

    cv2.imwrite(output_image_path, processed_img_uint8)
    print(f"Image traitée enregistrée sous : {output_image_path}")
    return True

#==================================================================#

#predict_image_from_directory(model, '../resources/dataset/archive/fingers/test')
save_processed_image(
    input_image_path="../resources/dataset/archive/fingers/train/3/0adc8f12-81fe-4818-aa4c-1269e06a4133_3R.png",
    output_image_path="../resources/test/image_traitee.jpg"
)
