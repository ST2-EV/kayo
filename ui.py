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
BOUNDARY_BUFFER = 150

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

interested_classes_labels = [
    "bottle",
    "cell phone",
    "backpack",
    "mouse",
    "keyboard",
    "book",
    "pizza",
]
interested_classes = [label_to_class[label] for label in interested_classes_labels]
print(interested_classes)
scanned_processed_objects = {}
ui.dark_mode().enable()
ui.add_head_html(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat&display=swap');
</style>
"""
)
video_capture = cv2.VideoCapture(0)
frame_cnt = 0
scanned_objects = []
sus_bucket = {}
mobile_number_value = ""
password_value = ""

del_sus_bucket = []


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


label = (
    ui.label("KAY/O")
    .classes("text-4xl font-bold text-center")
    .style("margin:0 auto;font-family: 'Montserrat', sans-serif;")
)
spinner = ui.spinner(size="xl").classes("absolute-center")  # absolute-center
spinner.set_visibility(True)

video_image_1 = ui.interactive_image().classes("w-full h-full")
ui_timer_1 = ui.timer(
    interval=0.1,
    callback=lambda: (video_image_1.set_source(f"/video/home?{time.time()}"),),
)

with ui.row().classes("w-9/12").style(
    "margin:0 auto;align-items: center;font-family: 'Montserrat', sans-serif;"
):
    mobile_number = ui.input(
        label="Mobile Number",
        placeholder="+1xxxxxxxxxx",
        validation={"Input too long": lambda value: len(value) < 13},
    )
    space_1 = ui.space()
    password = ui.input(
        label="Password",
        placeholder="4 digit number",
        validation={"Input too long": lambda value: len(value) < 5},
    )
    space_2 = ui.space()
    scan_button = ui.button(
        "Scan(3s) and KAY/O",
        on_click=lambda: scan_button_clicked(scan_button, video_image_1, ui_timer_1),
    ).style(
        "background-color: #cccccc !important;color: black !important;font-weight: bold !important;"
    )


def alert_user(item):
    send_message(f"Somebody stole your {item}", mobile_number_value)


@app.get("/video/sentry")
async def sentry():
    global video_capture, sus_bucket, scanned_processed_objects, del_sus_bucket
    try:
        requests.get("http://127.0.0.1:8000/heartbeat/")
    except:
        pass
    while video_capture.isOpened():
        success, frame = await run.io_bound(video_capture.read)
        if success:
            results = model(frame, classes=interested_classes, verbose=False)
            objs_in_frame = json.loads(results[0].tojson())

            good_list = []
            for scanned_key, scanned in scanned_processed_objects.items():
                for obj in objs_in_frame:
                    if (
                        (
                            abs(scanned["box"]["x1"] - obj["box"]["x1"])
                            <= BOUNDARY_BUFFER
                        )
                        and (
                            abs(scanned["box"]["x2"] - obj["box"]["x2"])
                            <= BOUNDARY_BUFFER
                        )
                        and (
                            abs(scanned["box"]["y1"] - obj["box"]["y1"])
                            <= BOUNDARY_BUFFER
                        )
                        and (
                            abs(scanned["box"]["y2"] - obj["box"]["y2"])
                            <= BOUNDARY_BUFFER
                        )
                    ):
                        good_list.append(scanned_key)

            sub = set(scanned_processed_objects.keys()) - set(good_list)
            if sub:
                for item in sub:
                    if item not in del_sus_bucket:
                        if item not in sus_bucket:
                            sus_bucket[item] = 1
                        else:
                            sus_bucket[item] += 1

            # if sub:
            #     for item in sub:
            #         if item not in sus_bucket:
            #             sus_bucket[item] = 1
            #         else:
            #             sus_bucket[item] += 1
            # del_sus_bucket = []
            for obj in objs_in_frame:
                for sus_key, sus_value in sus_bucket.items():
                    if sus_key not in del_sus_bucket:
                        if (
                            (
                                abs(
                                    scanned_processed_objects[sus_key]["box"]["x1"]
                                    - obj["box"]["x1"]
                                )
                                <= BOUNDARY_BUFFER
                            )
                            and (
                                abs(
                                    scanned_processed_objects[sus_key]["box"]["x2"]
                                    - obj["box"]["x2"]
                                )
                                <= BOUNDARY_BUFFER
                            )
                            and (
                                abs(
                                    scanned_processed_objects[sus_key]["box"]["y1"]
                                    - obj["box"]["y1"]
                                )
                                <= BOUNDARY_BUFFER
                            )
                            and (
                                abs(
                                    scanned_processed_objects[sus_key]["box"]["y2"]
                                    - obj["box"]["y2"]
                                )
                                <= BOUNDARY_BUFFER
                            )
                        ):
                            sus_bucket[sus_key] -= 2
                            # pprint(sus_bucket)
                            if sus_bucket[sus_key] <= 0:
                                sus_bucket[sus_key] = 0
                                if sus_key not in del_sus_bucket:
                                    del_sus_bucket.append(sus_key)
                        else:
                            sus_bucket[sus_key] += 1

            del_sus_bucket = []
            for sus_key, sus_value in sus_bucket.items():
                if sus_value >= 25:
                    print(sus_key, "is stolen")
                    alert_user(scanned_processed_objects[sus_key]["name"])
                    del scanned_processed_objects[sus_key]
                    # sus_bucket[sus_key] = 0
                    del_sus_bucket.append(sus_key)

            for key in del_sus_bucket:
                if key in sus_bucket:
                    del sus_bucket[key]

        jpeg = await run.cpu_bound(convert, frame)
        return Response(content=jpeg, media_type="image/jpeg")


def scan_button_clicked(scan_button, video_image_1, ui_timer_1):
    global frame_cnt, scanned_objects, mobile_number_value, password_value
    scan_button.delete()
    video_image_1.delete()
    ui_timer_1.delete()
    mobile_number_value = mobile_number.value
    mobile_number.delete()
    password_value = password.value
    password.delete()
    spinner.set_visibility(False)
    spinner.delete()

    frame_cnt = 0
    scanned_objects = []
    video_image_2 = ui.interactive_image().classes("w-full h-full")
    ui_timer_2 = ui.timer(
        interval=0.1,
        callback=lambda: video_image_2.set_source(f"/video/scan?{time.time()}"),
    )

    # Checks based on boundary
    def check_if_different_item(obj):
        global scanned_processed_objects, BOUNDARY_BUFFER
        for scanned_kikey, scanned_obj in scanned_processed_objects.items():
            # print(scanned_obj)
            if scanned_obj["name"] == obj["name"]:
                if (
                    (
                        abs(scanned_obj["box"]["x1"] - obj["box"]["x1"])
                        <= BOUNDARY_BUFFER
                    )
                    and (
                        abs(scanned_obj["box"]["x2"] - obj["box"]["x2"])
                        <= BOUNDARY_BUFFER
                    )
                    and (
                        abs(scanned_obj["box"]["y1"] - obj["box"]["y1"])
                        <= BOUNDARY_BUFFER
                    )
                    and (
                        abs(scanned_obj["box"]["y2"] - obj["box"]["y2"])
                        <= BOUNDARY_BUFFER
                    )
                ):
                    return False, scanned_kikey

        return True, ""

    def post_scan(video_image_2, ui_timer_2):
        global scanned_processed_objects
        video_image_2.delete()
        ui_timer_2.delete()
        space_1.delete()
        space_2.delete()

        try:
            requests.get("http://127.0.0.1:8000/initiate/")
        except:
            pass

        # for r in scanned_objects:
        #     r_dict = json.loads(r)
        #     for obj in r_dict:
        #         if obj["name"] not in scanned_processed_objects:
        #             scanned_processed_objects[obj["name"]] = obj

        scanned_processed_objects_cnt = {}
        for r in scanned_objects:
            r_dict = json.loads(r)
            for obj in r_dict:
                check_diff, kikey = check_if_different_item(obj)
                if check_diff or len(scanned_processed_objects) == 0:
                    scanned_processed_objects[
                        f"{obj['name']}_{obj['box']['x1']}_{obj['box']['x2']}_{obj['box']['y1']}_{obj['box']['y2']}"
                    ] = obj
                    scanned_processed_objects_cnt[
                        f"{obj['name']}_{obj['box']['x1']}_{obj['box']['x2']}_{obj['box']['y1']}_{obj['box']['y2']}"
                    ] = 1
                else:
                    if kikey not in scanned_processed_objects_cnt:
                        scanned_processed_objects_cnt[kikey] = 1
                    else:
                        scanned_processed_objects_cnt[kikey] += 1

        # for key, i in scanned_processed_objects.items():
        #     print(key, i["box"]["x1"], i["box"]["y1"])
        # print("==")
        # for key, i in sus_bucket.items():
        #     print(key, i)

        for key, i in scanned_processed_objects_cnt.items():
            if i <= 25:
                if key in scanned_processed_objects:
                    del scanned_processed_objects[key]

        pprint(scanned_processed_objects)
        pprint(scanned_processed_objects_cnt)

        if len(scanned_processed_objects) > 0:
            with ui.column().classes("w-full").style(
                "margin:0 auto;font-family: 'Montserrat', sans-serif;"
            ):
                beware_label = (
                    ui.label("Beware")
                    .classes("text-4xl font-bold text-center")
                    .style("margin:0 auto;font-family: 'Montserrat', sans-serif;")
                )
                video_image_3 = ui.interactive_image().classes("w-full h-full")
                video_image_3.set_source(f"static/eye.gif")
                area_monitored_label = (
                    ui.label("This Area is Monitored")
                    .classes("text-4xl font-bold text-center")
                    .style("margin:0 auto;font-family: 'Montserrat', sans-serif;")
                )
                password_check = ui.input(
                    label="Password",
                    validation={"Input too long": lambda value: len(value) < 5},
                ).style("margin:0 auto;")
                pass_button = (
                    ui.button(
                        "Disarm",
                        on_click=lambda: disarm(
                            password_check.value, password_check, pass_button
                        ),
                    )
                    .style("margin:0 auto;")
                    .style(
                        "background-color: #cccccc !important;color: black !important;"
                    )
                )

            video_image_4 = ui.interactive_image().classes("w-full h-full")
            timer_4 = ui.timer(
                interval=0.1,
                callback=lambda: video_image_4.set_source(
                    f"/video/sentry?{time.time()}"
                ),
            )
            video_image_4.set_visibility(False)

            def disarm(val, password_check, pass_button):
                if val == password_value:
                    video_image_4.delete()
                    timer_4.delete()
                    video_image_3.delete()
                    beware_label.delete()
                    area_monitored_label.delete()
                    password_check.delete()
                    pass_button.delete()
                    try:
                        requests.get("http://127.0.0.1:8000/finalize/")
                    except:
                        pass
                    ui.label("The eye has been disarmed.").classes(
                        "text-3xl font-bold text-center"
                    ).style("margin:0 auto;font-family: 'Montserrat', sans-serif;")

        else:
            ui.label("No belongings were detected in the space").classes(
                "text-3xl font-bold text-center"
            ).classes("text-3xl font-bold text-center").style("margin:0 auto;").style(
                "margin:0 auto;font-family: 'Montserrat', sans-serif;"
            )

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
