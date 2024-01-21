#!/usr/bin/env python3
import base64
import json
import signal
import time

import cv2
import numpy as np
from fastapi import Response
from nicegui import Client, app, core, run, ui
from ultralytics import YOLO

from message import send_message

NIKO = "+17788141068"
# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

class_to_label = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}
label_to_class = {v: k for k, v in class_to_label.items()}

interested_classes_labels = ["bottle", "cell phone"]
interested_classes = [label_to_class[label] for label in interested_classes_labels]
print(interested_classes)


video_capture = cv2.VideoCapture(0)


def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode(".jpg", frame)
    return imencode_image.tobytes()


frame_cnt = 0


@app.get("/video/frame")
# Thanks to FastAPI's `app.get`` it is easy to create a web route which always provides the latest image from OpenCV.
async def grab_video_frame() -> Response:
    _, frame = await run.io_bound(video_capture.read)

    jpeg = await run.cpu_bound(convert, frame)
    frame_cnt += 1
    return Response(content=jpeg, media_type="image/jpeg")


# For non-flickering image updates an interactive image is much better than `ui.image()`.
video_image = ui.interactive_image().classes("w-full h-full")
# A timer constantly updates the source of the image.
# Because data from same paths are cached by the browser,
# we must force an update by adding the current timestamp to the source.
ui.timer(
    interval=0.1, callback=lambda: video_image.set_source(f"/video/frame?{time.time()}")
)


async def scan(n):
    scanned_objects = []
    while video_capture.isOpened():
        success, frame = await run.io_bound(video_capture.read)

        if success:
            results = model(frame, classes=interested_classes, verbose=False)

            scanned_objects.append(results[0].tojson())
            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

        if frames == n:
            break

    video_capture.release()
    cv2.destroyAllWindows()

    objects = {}
    for r in scanned_objects:
        r_dict = json.loads(r)
        for obj in r_dict:
            if obj["name"] not in objects:
                objects[obj["name"]] = obj

    return objects


async def disconnect() -> None:
    """Disconnect all clients from current running server."""
    for client_id in Client.instances:
        await core.sio.disconnect(client_id)


def handle_sigint(signum, frame) -> None:
    # `disconnect` is async, so it must be called from the event loop; we use `ui.timer` to do so.
    ui.timer(0.1, disconnect, once=True)
    # Delay the default handler to allow the disconnect to complete.
    ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)


async def cleanup() -> None:
    # This prevents ugly stack traces when auto-reloading on code change,
    # because otherwise disconnected clients try to reconnect to the newly started server.
    await disconnect()
    # Release the webcam hardware so it can be used by other applications again.
    video_capture.release()


app.on_shutdown(cleanup)
# We also need to disconnect clients when the app is stopped with Ctrl+C,
# because otherwise they will keep requesting images which lead to unfinished subprocesses blocking the shutdown.
signal.signal(signal.SIGINT, handle_sigint)

ui.run()
