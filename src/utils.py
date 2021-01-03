
import json
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

