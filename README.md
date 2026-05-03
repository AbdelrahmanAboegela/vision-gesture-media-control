# Vision Gesture Media Control

Vision-based HCI system for authorized media and presentation control. The app uses a webcam to detect faces, recognizes registered users, unlocks gesture control only for the active authorized controller, and sends keyboard shortcuts to presentation, local video, or YouTube windows.

## Run

```powershell
python -m pip install -r requirements.txt
python main.py
```

## Project Layout

```text
.
|-- main.py                     # Thin launcher
|-- requirements.txt
|-- README.md
|-- src/
|   `-- vision_gesture_control/
|       |-- __init__.py
|       `-- app.py              # Application code
|-- config/
|   |-- gesture_config.json     # Runtime settings and bindings
|   |-- custom_gestures.example.json
|   `-- custom_gestures.json    # Generated locally, ignored by git
|-- data/
|   `-- db/                     # Private face samples, ignored by git
`-- models/                     # Downloaded model files, ignored by git
```

The app downloads these models automatically if they are missing:

| File | Source |
|---|---|
| `models/face_detection_yunet.onnx` | [OpenCV YuNet](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) |
| `models/face_recognition_sface_2021dec.onnx` | [OpenCV SFace](https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx) |
| `models/gesture_recognizer.task` | [MediaPipe Gesture Recognizer](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task) |

These files are intentionally excluded from git because they are downloaded runtime assets.

## Keyboard Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `a` | Register a new authorized person |
| `c` | Capture a custom gesture template |
| `d` | Delete a registered person |
| `m` | Cycle mode: presentation -> video -> youtube |
| `r` | Reload the face database without restart |
| `t` | Toggle dry-run mode |

## Built-In Gestures

Hold `Open_Palm` briefly to arm controls. Mode switching with `ILoveYou` is allowed without arming.

| Mode | Gesture | Action |
|---|---|---|
| Global | `ILoveYou` | Cycle mode |
| Presentation | `Thumb_Up` | Next slide |
| Presentation | `Thumb_Down` | Previous slide |
| Presentation | `Victory` | Start slideshow |
| Presentation | `Closed_Fist` | Exit slideshow |
| Video | `Open_Palm` | Play/Pause |
| Video | `Pointing_Up` | Mute/Unmute |
| Video | `Thumb_Up` | Volume up |
| Video | `Thumb_Down` | Volume down |
| Video | `Victory` | Speed up |
| Video | `Closed_Fist` | Speed down |
| Video | `Swipe_Right` | Seek forward |
| Video | `Swipe_Left` | Seek backward |
| YouTube | `Open_Palm` | Play/Pause |
| YouTube | `Pointing_Up` | Mute/Unmute |
| YouTube | `Thumb_Up` | Volume up |
| YouTube | `Thumb_Down` | Volume down |
| YouTube | `Victory` | Speed up |
| YouTube | `Closed_Fist` | Speed down |
| YouTube | `Swipe_Right` | Seek forward |
| YouTube | `Swipe_Left` | Seek backward |

Because the camera view is mirrored, swipe in the direction shown on the camera window: right for forward, left for backward.

## Custom Gestures

Press `c`, name the gesture, then assign one of the listed actions. The app captures 5 hand-landmark templates and saves them in `config/custom_gestures.json`. That file is intentionally ignored by git because it may contain personal biometric gesture samples. Custom gestures override built-in gestures only when the live hand shape is close enough to the stored template.

## Configuration

Edit `config/gesture_config.json` to change bindings, cooldowns, arming, target windows, or dry-run behavior.

Important settings:

- `external_controls.dry_run`: shows the detected action without pressing keys.
- `external_controls.focus_before_action`: tries to focus the target app before pressing keys.
- `external_controls.require_target_window`: if true, actions are blocked unless the target window is found.
- `external_controls.target_windows`: window-title keywords for PowerPoint, local video, and YouTube.
- `gestures.arming`: controls the hold-to-arm gesture and armed timeout.
- `ui.show_legend` and `ui.show_performance`: enable or disable on-screen overlays.

## Recognition And Authorization

Registered people are folders under `data/db/<person_name>/` containing sample face images. The `data/db/` contents are intentionally ignored by git. Deleting a folder removes that person after pressing `r` or restarting the app.

Only one active controller is allowed:

- The controller must be an authorized recognized face.
- If an unknown face is closer/larger than the authorized user, controls lock.
- If two authorized users are similarly prominent, controls lock as ambiguous.
- If the active controller disappears, controls remain valid briefly, then lock.
- Hand gestures are accepted only when the detected hand is closest to the active controller.

## Architecture

- OpenCV YuNet detects faces.
- OpenCV SFace creates cached face embeddings from the `db` folder.
- DeepFace emotion analysis runs in a background worker so it does not block the camera loop.
- MediaPipe Gesture Recognizer detects built-in hand gestures.
- Custom gesture matching compares normalized hand landmarks against saved templates.
- PyAutoGUI sends keyboard shortcuts to external apps.

## Optimization Notes

The original version ran `DeepFace.find(...)` and `DeepFace.analyze(...)` inside the display loop for every detected face. That made the camera feed depend on heavy model calls.

The current version is smoother because:

- Identity samples are embedded once into an SFace cache.
- Live identity checks happen only when face tracks become stable or need re-identification.
- Emotion analysis is asynchronous.
- Gesture detection uses MediaPipe live-stream mode.
- The UI loop only draws the latest available results.

## Demo Checklist

1. Register at least one person with `a`.
2. Open PowerPoint, a local video player, or YouTube.
3. Use `m` to select the correct mode.
4. Hold `Open_Palm` to arm commands.
5. Try the mode-specific gestures.
6. Press `t` for dry-run mode when testing gesture recognition without controlling apps.
7. Press `r` after changing the `db` folder manually.
