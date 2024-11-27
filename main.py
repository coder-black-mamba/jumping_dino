import cv2
import mediapipe as mp
import pyautogui
import time
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read frame
    success, image = cap.read()
    if not success:
        continue

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect pose
    results = pose.process(image_rgb)

    # Draw pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )


    cv2.line(image, (0, 500), (image.shape[1], 500), (0, 0, 255), 5)  # Green thick line


    # Display the image
    # if(results.pose_landmarks):
        # print(results.pose_landmarks.landmark[23])

    # print(results.pose_landmarks[0])
    # print(results.pose_landmarks[23],"\n", results.pose_landmarks[24])
    if(results.pose_landmarks):
        cv2.circle(image, (int(results.pose_landmarks.landmark[15].x * image.shape[1]), int(results.pose_landmarks.landmark[15].y * image.shape[0])), 15, (255, 0, 255), -1)
        cv2.circle(image, (int(results.pose_landmarks.landmark[23].x * image.shape[1]), int(results.pose_landmarks.landmark[23].y * image.shape[0])), 15, (255, 255, 255), -1)
        cv2.circle(image, (int(results.pose_landmarks.landmark[24].x * image.shape[1]), int(results.pose_landmarks.landmark[24].y * image.shape[0])), 15, (255, 255, 255), -1)
        target_pos_x_23=results.pose_landmarks.landmark[23].x * image.shape[0]
        target_pos_y_23=results.pose_landmarks.landmark[23].y * image.shape[1]
        target_pos_right_y=results.pose_landmarks.landmark[15].y * image.shape[1]

        target_pos_x_24=results.pose_landmarks.landmark[24].x * image.shape[0]
        traget_pos_y_24=results.pose_landmarks.landmark[24].y * image.shape[1]

        half_frame = image.shape[1] / 2;
        # if target_pos_y_23 < half frame then press space
        # draw a line on the screen where y==700

        if target_pos_y_23<900:
            print("reached")
            pyautogui.press('space')  # Press and release space
            cv2.line(image, (0, 500), (image.shape[1], 500), (0, 255, 0), 5)  # Green thick line

        # print(half_frame,target_pos_y_23)
        cv2.circle(image, (int(target_pos_x_23), int(target_pos_y_23)), 15, (255, 255, 255), -1)
    cv2.imshow('Jumping_Dino', image)

    # Break the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
pose.close()
