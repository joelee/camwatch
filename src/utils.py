
import json
import cv2
import numpy as np


def fix_rect(rect, cfg):
    """
    Use to correct the x, y from the image crop during gray and blur
    """
    if offset := cfg.capture.area_rect:
        ox, oy, ow, oh = offset
        return rect[0] + ox, rect[1] + oy, rect[2], rect[3]
    else:
        return rect

def resize_img(img, pix, inter = cv2.INTER_AREA):
    (h, w) = img.shape[:2]

    landscape = h < w

    # if pix is less than 1 consider is a scale
    if pix < 1.0:
        if landscape:
            pix = int(h * pix)
        else:
            pix = int(w * pix)
    elif pix == 1:
        return img, 1.0

    if landscape:
        if pix == h:
            return img, 1.0
        scale = pix / float(h)
        dim = (int(w * scale), pix)
    else:
        if pix == w:
            return img, 1.0
        scale = pix / float(w)
        dim = (pix, int(h * scale))

    return cv2.resize(img, dim, interpolation = inter), scale


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

