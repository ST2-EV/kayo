import asyncio
import json
import signal
import time
from pprint import pprint

import cv2
import numpy as np
import requests
from fastapi import Response
from nicegui import Client, app, core, run, ui
from ultralytics import YOLO

from message import send_message

NIKO = "+17788141068"

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
scanned_processed_objects = {}
ui.dark_mode().enable()
video_capture = cv2.VideoCapture(0)
frame_cnt = 0
scanned_objects = []
sus_list = {}


def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode(".jpg", frame)
    return imencode_image.tobytes()


@app.get("/video/home")
async def grab_video_frame_1() -> Response:
    global video_capture
    while video_capture.isOpened():
        success, frame = await run.io_bound(video_capture.read)
        if success:
            results = model(frame, classes=interested_classes, verbose=False)

            annotated_frame = results[0].plot()

            jpeg = await run.cpu_bound(convert, annotated_frame)
            return Response(content=jpeg, media_type="image/jpeg")
        else:
            raise RuntimeError("Could not read frame from video capture device.")


@app.get("/video/scan")
async def grab_video_frame_2() -> Response:
    global video_capture, frame_cnt, scanned_objects
    while video_capture.isOpened():
        success, frame = await run.io_bound(video_capture.read)
        if success:
            results = model(frame, classes=interested_classes, verbose=False)
            scanned_objects.append(results[0].tojson())

            annotated_frame = results[0].plot()

            jpeg = await run.cpu_bound(convert, annotated_frame)
            return Response(content=jpeg, media_type="image/jpeg")
        else:
            raise RuntimeError("Could not read frame from video capture device.")


label = ui.label("YSG")
spinner = ui.spinner(size="xl").classes(
    "flex justify-center items-center"
)  # absolute-center
spinner.set_visibility(True)

ui.input(
    label="Mobile Number",
    placeholder="+1",
    on_change=lambda e: result.set_text("you typed: " + e.value),
    validation={"Input too long": lambda value: len(value) < 20},
)

video_image_1 = ui.interactive_image().classes("w-full h-full")
ui_timer_1 = ui.timer(
    interval=0.1,
    callback=lambda: (video_image_1.set_source(f"/video/home?{time.time()}"),),
)

scan_button = ui.button(
    "Scan(10s) and keep and eye",
    on_click=lambda: scan_button_clicked(scan_button, video_image_1, ui_timer_1),
)


def alert_user(item):
    # send_message(f"Somebody stole your {item}", NIKO)
    print(f"Somebody stole your {item}")


@app.get("/video/sentry")
async def sentry():
    print("WUBBALUBBADUBDUB")
    global video_capture, sus_list, scanned_processed_objects
    while video_capture.isOpened():
        success, frame = await run.io_bound(video_capture.read)
        if success:
            results = model(frame, classes=interested_classes, verbose=False)
            objs_in_frame = json.loads(results[0].tojson())
            objects_in_frame = set([obj["name"] for obj in objs_in_frame])
            scanned_objs_string = set(scanned_processed_objects.keys())
            sub = scanned_objs_string - objects_in_frame
            if sub:
                for item in sub:
                    if item not in sus_list:
                        sus_list[item] = 1
                    else:
                        sus_list[item] += 1

            for obj in objs_in_frame:
                if obj["name"] in sus_list:
                    sus_list[obj["name"]] -= 1
                    if sus_list[obj["name"]] <= 0:
                        del sus_list[obj["name"]]

            if sus_list:
                print(sus_list)
                del_sus_list = []
                for item in sus_list:
                    if sus_list[item] >= 25:
                        alert_user(item)
                        del_sus_list.append(item)
                        del scanned_processed_objects[item]

                for item in del_sus_list:
                    del sus_list[item]

        jpeg = await run.cpu_bound(convert, frame)
        return Response(content=jpeg, media_type="image/jpeg")


def scan_button_clicked(scan_button, video_image_1, ui_timer_1):
    global frame_cnt, scanned_objects
    scan_button.delete()
    video_image_1.delete()
    ui_timer_1.delete()
    spinner.set_visibility(False)
    spinner.delete()

    frame_cnt = 0
    scanned_objects = []
    video_image_2 = ui.interactive_image().classes("w-full h-full")
    ui_timer_2 = ui.timer(
        interval=0.1,
        callback=lambda: video_image_2.set_source(f"/video/scan?{time.time()}"),
    )

    def post_scan(video_image_2, ui_timer_2):
        global scanned_processed_objects
        video_image_2.delete()
        ui_timer_2.delete()

        for r in scanned_objects:
            r_dict = json.loads(r)
            for obj in r_dict:
                if obj["name"] not in scanned_processed_objects:
                    scanned_processed_objects[obj["name"]] = obj

        if len(scanned_processed_objects) > 0:
            ui.label("Beware")
            video_image_3 = ui.interactive_image().classes("w-full h-full")
            video_image_3.set_source(f"static/eye.gif")
            ui.label("This Area is Monitored")

            video_image_4 = ui.interactive_image().classes("w-full h-full")
            ui.timer(
                interval=0.1,
                callback=lambda: video_image_4.set_source(
                    f"/video/sentry?{time.time()}"
                ),
            )
            video_image_4.set_visibility(False)
        else:
            ui.label("No belongings were detected in the space")

    ui.timer(
        interval=5,
        callback=lambda: post_scan(video_image_2, ui_timer_2),
        once=True,
    )


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
