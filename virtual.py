import cv2
import mediapipe as mp
import pyautogui
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pyautogui.FAILSAFE = False

screen_w, screen_h = pyautogui.size()

# SMOOTHING
prev_x = 0
prev_y = 0
smoothening = 8

# TIMERS
last_click_time = 0
last_scroll_time = 0
last_screenshot_time = 0

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=r'C:\Users\Chetana\Desktop\coding\GestureProj\hand_landmarker.task'
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, img = cap.read()

        if not success:
            break

        img = cv2.flip(img, 1)

        h, w, _ = img.shape

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imgRGB
        )

        results = landmarker.detect(mp_image)

        if results.hand_landmarks:

            hand = results.hand_landmarks[0]

            index_finger = hand[8]
            thumb = hand[4]
            middle_finger = hand[12]
            pinky = hand[20]

            x = int(index_finger.x * w)
            y = int(index_finger.y * h)

            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)

            middle_x = int(middle_finger.x * w)
            middle_y = int(middle_finger.y * h)

            pinky_x = int(pinky.x * w)
            pinky_y = int(pinky.y * h)

            # CURSOR MOVEMENT
            screen_x = int(index_finger.x * screen_w)
            screen_y = int(index_finger.y * screen_h)

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)

            prev_x = curr_x
            prev_y = curr_y

            # DRAW POINTS
            cv2.circle(img, (x, y), 15, (255, 0, 255), -1)
            cv2.circle(img, (thumb_x, thumb_y), 15, (0, 255, 0), -1)
            cv2.circle(img, (middle_x, middle_y), 15, (255, 255, 0), -1)
            cv2.circle(img, (pinky_x, pinky_y), 15, (0, 255, 255), -1)

            current_time = time.time()

            # CLICK
            click_distance = (
                (x - thumb_x) ** 2 +
                (y - thumb_y) ** 2
            ) ** 0.5

            if (
                click_distance < 40 and
                current_time - last_click_time > 0.5
            ):

                pyautogui.click()

                cv2.putText(
                    img,
                    "CLICK",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                last_click_time = current_time

            # SCROLL UP
            if middle_y < y - 60:

                if current_time - last_scroll_time > 0.8:

                    pyautogui.scroll(120)

                    cv2.putText(
                        img,
                        "SCROLL UP",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3
                    )

                    last_scroll_time = current_time

            # SCROLL DOWN
            elif middle_y > y + 60:

                if current_time - last_scroll_time > 0.8:

                    pyautogui.scroll(-120)

                    cv2.putText(
                        img,
                        "SCROLL DOWN",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3
                    )

                    last_scroll_time = current_time

            # SCREENSHOT
            screenshot_distance = (
                (thumb_x - pinky_x) ** 2 +
                (thumb_y - pinky_y) ** 2
            ) ** 0.5

            if (
                screenshot_distance < 40 and
                current_time - last_screenshot_time > 1
            ):

                screenshot = pyautogui.screenshot()

                screenshot.save("gesture_screenshot.png")

                cv2.putText(
                    img,
                    "SCREENSHOT",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3
                )

                last_screenshot_time = current_time

        cv2.imshow("GestureFlow AI", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()