

import time
import cv2

from frame import Frame
from utils import fix_rect


FACE_CASCADE = None


def detect_face(frame: Frame) -> dict:
    global FACE_CASCADE
    ts = time.time()
    cfg = frame.cfg.face_detect
    if FACE_CASCADE is None:
        FACE_CASCADE = cv2.CascadeClassifier(
            cfg.cascade_file
        )
    cas = FACE_CASCADE.detectMultiScale(
        frame.img_gray, cfg.scale_factor, cfg.min_neighbours
    )
    count = len(cas)
    max_area = 0
    zones = []
    if count > 0:
        for zone in cas:
            zone = fix_rect(zone, frame.cfg)
            x, y, w, h = zone
            area = w * h
            if max_area < area:
                max_area = area
            if cfg.draw_rect:
                cv2.rectangle(
                    frame.image,
                    (x, y), (x + w, y + h),
                    cfg.draw_rect_color,
                    cfg.draw_rect_thickness
                )
            zones.append(zone)

    ret = {
        'face_detected': count,
        'face_area': max_area,
        'face_zones': zones,
        'face_ts': time.time() - ts
    }

    img_path = frame.cfg.face_detect.save_image_path
    if count > 0 and img_path:
        file_name = (
            f'face-{frame.frame}-{count}-{max_area}.jpg'
        )
        cv2.imwrite(img_path + '/' + file_name, frame.image)
        ret['face_file'] = file_name

    return ret
