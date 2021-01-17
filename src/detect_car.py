

import time
import cv2

from frame import Frame
from utils import fix_rect


CAR_CASCADE = None


def detect_car(frame: Frame) -> dict:
    global CAR_CASCADE
    ts = time.time()
    cfg = frame.cfg.car_detect
    if CAR_CASCADE is None:
        CAR_CASCADE = cv2.CascadeClassifier(
            cfg.cascade_file
        )
    cas = CAR_CASCADE.detectMultiScale(
        frame.img_gray, cfg.scale_factor, cfg.min_neighbours
    )
    # has_cas = len(cas)
    count = 0
    max_area = 0
    zones = []
    area = 0
    # if has_cas > 0:
    for zone in cas:
        zone = fix_rect(zone, frame.cfg)
        # print(zone)
        x, y, w, h = zone
        area = w * h
        if area >= cfg.min_area and (cfg.max_area == 0 or area <= cfg.max_area):
            if max_area < area:
                max_area = area
            zones.append(zone)
            count += 1
    print('Car detected:', count, len(cas), area)
    ret = {
        'car_detected': count,
        'car_area': max_area,
        'car_zones': zones,
        'car_ts': time.time() - ts
    }

    img_path = frame.cfg.car_detect.save_image_path
    if count > 0 and img_path:
        file_name = (
            f'car-{frame.frame}-{count}-{max_area}.jpg'
        )
        cv2.imwrite(img_path + '/' + file_name, frame.image)
        ret['car_file'] = file_name

    return ret
