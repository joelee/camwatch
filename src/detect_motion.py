
import time
import cv2

from frame import Frame
from utils import fix_rect


def detect_motion(frame: Frame) -> dict:
    ts = time.time()
    cfg = frame.cfg.detector
    delta = cv2.absdiff(frame.base_img, frame.img_blur)
    if delta is None:
        return {
            'motion': False,
            'motion_area': 0,
            'motion_zones': [],
            'motion_count': 0,
            'motion_ts': time.time() - ts
        }
    threshold = cv2.threshold(
        delta,
        cfg.threshold.value,
        cfg.threshold.max_value,
        cv2.THRESH_BINARY
    )[1]
    (contours, _) = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    t_area = 0
    zones = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < cfg.min_area:
            continue
        t_area += area
        zone = fix_rect(cv2.boundingRect(contour), frame.cfg)
        if cfg.draw_rect:
            x, y, w, h = zone
            cv2.rectangle(
                frame.image,
                (x, y), (x + w, y + h),
                cfg.draw_rect_color,
                cfg.draw_rect_thickness
            )
        zones.append(zone)

    count = len(zones)
    ret = {
        'motion': t_area > 0,
        'motion_area': t_area,
        'motion_zones': zones,
        'motion_count': len(zones),
        'motion_ts': time.time() - ts
    }

    img_path = frame.cfg.detector.save_image_path
    if t_area > 0 and img_path:
        file_name = (
            f'motion-{frame.frame}-{count}-{t_area}.jpg'
        )
        cv2.imwrite(img_path + '/' + file_name, frame.image)
        ret['motion_file'] = file_name

    return ret
