import os
import cv2
import numpy as np
import pyautogui

from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
)
from eyetrax.cli import parse_common_args
from eyetrax.filters import KalmanSmoother, KDESmoother, NoSmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, iter_frames


def run_mouse_control():
    # Enable fail-safe: moving mouse to a corner will throw an exception and stop the script
    pyautogui.FAILSAFE = True

    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    confidence_level = args.confidence

    gaze_estimator = GazeEstimator(model_name=args.model)

    # Model loading or calibration
    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[mouse] Loaded gaze model from {args.model_file}")
    else:
        print("[mouse] Starting calibration...")
        if calibration_method == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=camera_index)
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    # Check if model is fitted
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(gaze_estimator.model.scaler)
    except NotFittedError:
        print("[Error] Model not fitted. Calibration failed or was skipped. Exiting.")
        return

    screen_width, screen_height = get_screen_size()
    print(f"[mouse] Screen size detected: {screen_width}x{screen_height}")

    # Setup filter
    if filter_method == "kalman":
        kalman = make_kalman()
        smoother = KalmanSmoother(kalman)
        smoother.tune(gaze_estimator, camera_index=camera_index)
    elif filter_method == "kde":
        kalman = None
        smoother = KDESmoother(screen_width, screen_height, confidence=confidence_level)
    else:
        kalman = None
        smoother = NoSmoother()

    print("=========================================================")
    print("Eye Tracking Mouse Control Started")
    print(" - Press 'Esc' to quit.")
    print(" - Move mouse to any screen corner to trigger FAILSAFE exit.")
    print("=========================================================")

    # Small window to capture keyboard input
    window_name = "EyeTrax Mouse Control (Esc to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 400, 100)

    try:
        with camera(camera_index) as cap:
            for frame in iter_frames(cap):
                features, blink_detected = gaze_estimator.extract_features(frame)

                if features is not None and not blink_detected:
                    gaze_point = gaze_estimator.predict(np.array([features]))[0]
                    x, y = map(int, gaze_point)

                    # Apply smoothing
                    x_pred, y_pred = smoother.step(x, y)

                    if x_pred is not None and y_pred is not None:
                        target_x = int(x_pred)
                        target_y = int(y_pred)

                        # Ensure coordinates are within screen bounds
                        target_x = max(0, min(screen_width - 1, target_x))
                        target_y = max(0, min(screen_height - 1, target_y))

                        # Move mouse
                        # set _pause to False for faster movement
                        pyautogui.moveTo(target_x, target_y, _pause=False)

                # Update the small window
                status_img = np.zeros((100, 400, 3), dtype=np.uint8)
                cv2.putText(
                    status_img,
                    "Running...",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                if blink_detected:
                    cv2.putText(
                        status_img,
                        "Blink",
                        (200, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow(window_name, status_img)

                if cv2.waitKey(1) == 27:  # Esc to quit
                    break

    except pyautogui.FailSafeException:
        print("\n[mouse] FailSafe triggered from mouse corner. Exiting.")
    except KeyboardInterrupt:
        print("\n[mouse] KeyboardInterrupt. Exiting.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_mouse_control()
