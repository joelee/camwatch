

import time
import cv2
import dlib

from frame import Frame
from utils import resize_img

detector = dlib.get_frontal_face_detector()


def detect_face(frame: Frame) -> dict:
    ts = time.time()
    cfg = frame.cfg.face_detect

    img, scale = resize_img(
        frame.img_gray if cfg.convert_gray else frame.image,
        cfg.resize_img_pixel
    )

    dets = detector(img, cfg.upsample)

    count = len(dets)
    max_area = 0
    zones = []
    for d in dets:
        l, t, r, b = d.left(), d.top(), d.right(), d.bottom()
        left = int(l / scale)
        top = int(t / scale)
        width = int((r - l) / scale)
        height = int((b - t) / scale)
        area = width * height
        if max_area < area:
            max_area = area
        zones.append(
            (left, top, width, height)
        )
        if cfg.draw_rect:
            print(
                'Rect:', (l, t, r, b), scale, ':',
                (left, top, width, height),
                area, max_area
            )
            cv2.rectangle(
                frame.image,
                (left, top), (left + width, top + height),
                cfg.draw_rect_color,
                cfg.draw_rect_thickness
            )

    ret = {
        'face_detected': count,
        'face_area': max_area,
        'face_zones': zones,
        'face_ts': time.time() - ts,
        'face_img': frame.image
    }

    img_path = cfg.save_image_path
    if count > 0 and img_path:
        file_name = (
            f'face-{frame.index}-{count}-{max_area}.jpg'
        )
        cv2.imwrite(img_path + '/' + file_name, frame.image)
        ret['face_file'] = file_name

    return ret