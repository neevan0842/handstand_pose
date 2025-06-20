from time import sleep
from logger import logger
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3

mp_pose = mp.solutions.pose
FILE_NAME = "file.txt"


def get_angle(a: list[float], b: list[float], c: list[float]) -> float:
    """Calculates the angle (in degrees) between three points.

    Args:
        a (list[float]): The first point as [x, y].
        b (list[float]): The middle point as [x, y].
        c (list[float]): The last point as [x, y].

    Returns:
        float: The angle in degrees between the three points.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_all_angles(modifiedlandmarks: dict[str, list[float]]) -> dict[str, float]:
    """Computes angles for key joints using landmark coordinates.

    Args:
        modifiedlandmarks (dict[str, list[float]]): Dictionary mapping joint names to [x, y] coordinates.

    Returns:
        dict[str, float]: Dictionary mapping joint names to their calculated angles in degrees.
    """
    angles = {
        "LEFT_ELBOW": get_angle(
            modifiedlandmarks["LEFT_SHOULDER"],
            modifiedlandmarks["LEFT_ELBOW"],
            modifiedlandmarks["LEFT_WRIST"],
        ),
        "RIGHT_ELBOW": get_angle(
            modifiedlandmarks["RIGHT_SHOULDER"],
            modifiedlandmarks["RIGHT_ELBOW"],
            modifiedlandmarks["RIGHT_WRIST"],
        ),
        "LEFT_SHOULDER": get_angle(
            modifiedlandmarks["LEFT_HIP"],
            modifiedlandmarks["LEFT_SHOULDER"],
            modifiedlandmarks["LEFT_ELBOW"],
        ),
        "RIGHT_SHOULDER": get_angle(
            modifiedlandmarks["RIGHT_HIP"],
            modifiedlandmarks["RIGHT_SHOULDER"],
            modifiedlandmarks["RIGHT_ELBOW"],
        ),
        "LEFT_HIP": get_angle(
            modifiedlandmarks["LEFT_SHOULDER"],
            modifiedlandmarks["LEFT_HIP"],
            modifiedlandmarks["LEFT_KNEE"],
        ),
        "RIGHT_HIP": get_angle(
            modifiedlandmarks["RIGHT_SHOULDER"],
            modifiedlandmarks["RIGHT_HIP"],
            modifiedlandmarks["RIGHT_KNEE"],
        ),
        "LEFT_KNEE": get_angle(
            modifiedlandmarks["LEFT_HIP"],
            modifiedlandmarks["LEFT_KNEE"],
            modifiedlandmarks["LEFT_ANKLE"],
        ),
        "RIGHT_KNEE": get_angle(
            modifiedlandmarks["RIGHT_HIP"],
            modifiedlandmarks["RIGHT_KNEE"],
            modifiedlandmarks["RIGHT_ANKLE"],
        ),
    }
    return angles


def verify_angles(
    angles: dict[str, float], lower_angle: int = 150, upper_angle: int = 180
) -> tuple[bool, str | None]:
    """Verifies if all angles are within the specified range.

    Args:
        angles (dict[str, float]): Dictionary of joint angles.
        lower_angle (int, optional): Lower bound for angle verification. Defaults to 150.
        upper_angle (int, optional): Upper bound for angle verification. Defaults to 180.

    Returns:
        tuple[bool, str | None]: (True, None) if all angles are within range, otherwise (False, joint name).
    """
    for point in angles:
        if not (lower_angle <= angles[point] <= upper_angle):
            return False, point
    return True, None


def draw_landmarks(
    image, angles: dict[str, float], landmarks: dict[str, list[float]]
) -> None:
    """Draws the angle values on the image at the corresponding landmark positions.

    Args:
        image: The image on which to draw.
        angles (dict[str, float]): Dictionary of joint angles.
        landmarks (dict[str, list[float]]): Dictionary of joint landmark coordinates.
    """
    for point in angles.keys():
        cv2.putText(
            image,
            f"{angles[point]:.2f}",
            tuple(
                np.multiply(landmarks[point], [image.shape[1], image.shape[0]]).astype(
                    int
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def speak():
    """Continuously reads content from a file and uses text-to-speech to speak it aloud.

    Reads from FILE_NAME, speaks the content if present, and then clears the file. Runs in an infinite loop with a delay.
    """
    tts = pyttsx3.init()
    tts.setProperty("rate", 150)
    sleep(3)  # wait for user to get into position
    while True:
        content = read_from_file(file_name=FILE_NAME)
        if content:
            if tts._inLoop:
                tts.endLoop()
            logger.info(f"Speaking: {content}")
            tts.say(content)
            tts.runAndWait()
            write_to_file(file_name=FILE_NAME, content="")
        sleep(2)


def write_to_file(file_name: str = FILE_NAME, content: str = "") -> None:
    """Writes the specified content to a file.

    Args:
        file_name (str, optional): The name of the file to write to. Defaults to FILE_NAME.
        content (str, optional): The content to write. Defaults to an empty string.
    """
    with open(file_name, "w") as file:
        file.write(content)
        logger.debug(f"Written to file {file_name}: {content}")


def read_from_file(file_name: str = FILE_NAME) -> str:
    """Reads and returns the content of a file.

    Args:
        file_name (str, optional): The name of the file to read from. Defaults to FILE_NAME.

    Returns:
        str: The content of the file, or an empty string if the file does not exist.
    """
    try:
        with open(file_name, "r") as file:
            return file.read()
    except FileNotFoundError:
        return ""


def get_important_landmarks(landmarks) -> dict[str, list[float]]:
    """Extracts the coordinates of important body landmarks from the landmarks object.

    Args:
        landmarks: The landmarks object containing pose landmark data.

    Returns:
        dict[str, list[float]]: Dictionary mapping important joint names to their [x, y] coordinates.
    """
    important_points = [
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
    ]
    return {
        point: [
            landmarks[mp_pose.PoseLandmark[point].value].x,
            landmarks[mp_pose.PoseLandmark[point].value].y,
        ]
        for point in important_points
    }
