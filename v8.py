import json

import cv2
from ultralytics import YOLO

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


def scan(n):
    cap = cv2.VideoCapture(0)
    frames = 0
    scanned_objects = []
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame, classes=interested_classes, verbose=False)

            scanned_objects.append(results[0].tojson())
            frames += 1
            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

        if frames == n:
            break

    cap.release()
    cv2.destroyAllWindows()

    objects = {}
    for r in scanned_objects:
        r_dict = json.loads(r)
        for obj in r_dict:
            if obj["name"] not in objects:
                objects[obj["name"]] = obj

    return objects


def alert_user(item):
    print(f"Somebody stole your {item}")


def alert_sus_list(sus_list):
    print(sus_list)
    for item in sus_list:
        if sus_list[item] >= 25:
            alert_user(item)


def sentry(scanned_objects):
    cap = cv2.VideoCapture(0)
    sus_list = {}
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model(frame, classes=interested_classes, verbose=False)

            objs_in_frame = json.loads(results[0].tojson())

            objects_in_frame = set([obj["name"] for obj in objs_in_frame])
            scanned_objs_string = set(scanned_objects.keys())

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
                alert_sus_list(sus_list)

            annotated_frame = results[0].plot()

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    scanned_objects = scan(50)
    sentry(scanned_objects)
