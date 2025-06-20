import threading
import cv2
import mediapipe as mp
from logger import logger
from utils import (
    draw_landmarks,
    get_all_angles,
    get_important_landmarks,
    speak,
    verify_angles,
    write_to_file,
    FILE_NAME,
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
thread = threading.Thread(target=speak, daemon=True)
cap = cv2.VideoCapture(str(r"C:\Users\manoj\Downloads\handstandlong.mp4"))


def main():
    with mp_pose.Pose(
        min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1
    ) as pose:
        thread.start()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(frame, 1)

            # Recolor image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
            except:
                pass

            # Draw landmarks and angles
            modified_landmarks = get_important_landmarks(landmarks)
            angles = get_all_angles(modified_landmarks)
            draw_landmarks(image, angles, modified_landmarks)

            # Verify angles
            res, point = verify_angles(angles, lower_angle=150, upper_angle=180)
            if not res:
                write_to_file(file_name=FILE_NAME, content=point.replace("_", " "))
            # print(res, point, angles[point] if not res else None)
            logger.debug(
                f"Result: {res}, Point: {point}, Angle: {angles[point] if not res else None}"
            )

            # Render detections
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            cv2.imshow("Handstand Checker", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                write_to_file(file_name=FILE_NAME, content="")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
