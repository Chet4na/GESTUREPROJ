import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Connections between landmarks to draw lines
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index
    (0,9),(9,10),(10,11),(11,12),   # Middle
    (0,13),(13,14),(14,15),(15,16), # Ring
    (0,17),(17,18),(18,19),(19,20), # Pinky
    (5,9),(9,13),(13,17)            # Palm
]

def draw_landmarks(image, hand_landmarks_list):
    h, w, _ = image.shape
    for landmarks in hand_landmarks_list:

        # Draw lines
        for start, end in HAND_CONNECTIONS:
            x1 = int(landmarks[start].x * w)
            y1 = int(landmarks[start].y * h)
            x2 = int(landmarks[end].x * w)
            y2 = int(landmarks[end].y * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw dots
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=r'C:\Users\Chetana\Desktop\coding\GestureProj\hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, img = cap.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            draw_landmarks(img, results.hand_landmarks)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()