"""
Configuration Class
"""

import os
import json
import yaml
import numpy as np


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, CfgObject):
            return obj.serialize()
        return json.JSONEncoder.default(self, obj)

class ConfiguratorException(Exception):
    pass


class Configurator:
    CHANNEL_DEFAULT = {
        # Name
        'name': None,
        # Set to False to disable
        'enabled': True,
        # Motion Capture configuration
        'capture': {
            # CV2 Capture Source
            'source': None,
            # Frame per second
            'fps': 5,
            # seconds to record before motion detected
            'pre_motion': 5,
            # seconds to record after motion detected
            'post_motion': 5,
            # max video length per file in seconds
            'video_len': 300,
            # num of frames to skip detection when recording starts
            'rec_hold': 60,
            # Run time in seconds before quitting (0 = run forever)
            'uptime': 0,
            # Image width
            'width': 1280,
            # Image height
            'height': 720,
            # Polygon for detection area (None = whole image)
            'areas': None,
            # Draw detection area (for debugging)
            'draw_areas': False,
            # Show frame on OpenCV Window (for debugging)
            'show_frame': False
        },
        # Motion Detection configuration
        'detector': {
            # Minimum Area in pixel to trigger detection
            'min_area': 3000,
            # Cool down in second before start detection again
            'cool_down': 5,
            # Number of frames ago to compare motion detection
            'frames_ago': 10,
            # Enabled Key Frame capture
            'key_frame_enabled': True,
            # Block of frames for key frame selection (in second)
            # Frame with the largest motion are in the block will be
            #  choose as key frame [ first_block, subsequent_blocks ]
            'key_frame_block': [ 1.0, 10.0 ],
            # Draw rectangle on motion
            'draw_rect': False,
            # rectangle colour array [blue, green, red]
            # [0, 255, 255] => '#00FFFF' (Yellow)
            'draw_rect_color': [0, 255, 255],
            # Rectangle line thickness
            'draw_rect_thickness': 2,
            # Path to save motion image
            'save_image_path': None,
            # OpenCV Gaussion Blur settings
            'gaussian_blur': {
                'kernel_size': [45, 45],
                'sigmax': 0
            },
            # OpenCV Threshold settings
            'threshold': {
                # threshold value
                'value': 35,
                # maximum value to use with the THRESH_BINARY
                # thresholding types
                'max_value': 255
            }
        },
        # Face Detection configuration
        'face_detect': {
            'enabled': True,
            # Maximum faces to detect in a motion session
            # No face detection after to reduce workload
            'max_face': 1,
            # Cascade classifier file
            'cascade_file': 'cascades/haarcascade_frontalface_alt.xml',
            # Parameter specifying how much the image size is reduced at each image scale.
            'scale_factor': 1.5,
            # Parameter specifying how many neighbors each candidate rectangle should have to retain it.
            'min_neighbours': 2,
            # Draw rectangle on face
            'draw_rect': False,
            # rectangle colour array [blue, green, red]
            # [255, 0, 0] => '#0000FF' (Blue)
            'draw_rect_color': [255, 0, 0],
            # Rectangle line thickness
            'draw_rect_thickness': 2,
            # Path to save face detected image
            'save_image_path': None,
        },
        # Car Detection configuration
        'car_detect': {
            'enabled': False,
            # Maximum cars to detect in a motion session
            # No face detection after to reduce workload
            'max_car': 1,
            # Cascade classifier file
            'cascade_file': 'cascades/cars.xml',
            # Parameter specifying how much the image size is reduced at each image scale.
            'scale_factor': 1.5,
            # Parameter specifying how many neighbors each candidate rectangle should have to retain it.
            'min_neighbours': 2,
            # Draw rectangle on face
            'draw_rect': False,
            # rectangle colour array [blue, green, red]
            # [255, 0, 0] => '#0000FF' (Green)
            'draw_rect_color': [0, 255, 0],
            # Rectangle line thickness
            'draw_rect_thickness': 2,
            # Path to save face detected image
            'save_image_path': None,
        },
        # Video Writer configuration
        'writer': {
            # Path to write Video files. None/null = disable
            'location': None,
            # motion, always, disabled
            'record_on': 'motion',
            # Frame per second
            'fps': 5,
            # Video filename format
            'file_format': '{name}/%Y-%m-%d/{name}-%j-%H%M%S-%f-{enc}.mp4',
            # Key Frame filename format
            'kf_file_format': '{name}/%Y-%m-%d/{name}-%j-%H%M%S-%f.jpg',
            # Video encoding/codec - fourcc code
            # e.g.: mp4v, DIVX, XVID, MJPG, X264, WMV1, WMV2
            'file_encoding': 'mp4v',
            # Image width (0 = source width)
            'width': 0,
            # Image height (0 = source height)
            'height': 0
        }
    }

    SERVICE_DEFAULT = {
        # MQTT
        'mqtt': {
            # Hostname or IP Address of MQTT Server
            # MQTT is disabled if null
            'host': None,
            # Port of MQTT Server
            'port': 1883,
            'keepalive': 120,
            # App Name used for Topic prefix
            'app': 'cam_detector',
            # Seconds for Motion event to be reset
            'motion_reset': 60,
            # Seconds for Face detected event to be reset
            'face_reset': 60,
            # Seconds for Car detected event to be reset
            'car_reset': 60
        },
        # Face recognition settings
        'face_recognition': {
            # Enable face recognition
            'enabled': False,
            # Path to store Face recognition files
            'location': None,
            # Retrain face encodings automatically everytime at restart.
            # Default is Manual Retraining
            'retrain_onstart': False,
            # Use the large (SLOWER) model for face recognition
            'use_large_model': False,
            # How many times to re-sample the face when calculating encoding.
            # Higher is more accurate, but slower (i.e. 100 is 100x slower)
            'num_jitters': 1,
            # How much distance between faces to consider it a match.Lower is
            # more strict. 0.6 is typical best performance.
            'tolerance': 0.6,
            # How many times to upsample the image looking for faces.Higher
            # numbers find smaller faces.
            'n_upsample': 1,
            # Use a more accurate deep-learning model using GPU/CUDA (if available)
            'cuda_model': False,
            # Scale the original image to speed-up recognition
            'scale': 0.75,
            # Minimum score for encoding matching (0.1 to 1.0)
            'min_score': 0.75
        },
        # Number Plate Recognition (in development)
        'car_plate_recognition': {
            'enabled': False
        }
    }

    def __init__(self, cfg=None):
        if cfg is None:
            home_path = os.path.expanduser('~')
            app_path = os.path.realpath(os.path.dirname(__file__) + '/..')
            paths = [
                home_path, home_path + '/.camwatch', '/etc',
                app_path + '/config', app_path, os.getcwd()
            ]
            cfg_path = os.getenv('CV_CONFIG_PATH')
            if cfg_path is not None:
                paths.insert(0, cfg_path)
            for path in paths:
                if os.path.isdir(path):
                    for f in [
                        'camwatch.yaml', 'camwatch.json',
                        '.camwatch.yaml', '.camwatch.json'
                    ]:
                        file = os.path.join(path, f)
                        if os.path.isfile(file):
                            cfg = self._load_from_file(file)
                            break
            if cfg is None:
                raise ConfiguratorException('Configuration file not found.')

        if not isinstance(cfg, dict):
            raise ConfiguratorException('Invalid Configuration type')

        self._pointer = 0
        self._index = {}
        self._channels = []
        channel_default = self.CHANNEL_DEFAULT
        service_default = self.SERVICE_DEFAULT
        pointer = 0

        if 'default' in cfg:
            channel_default = Configurator.__merge_dicts(
                channel_default, cfg['default']
            )

        if 'services' in cfg:
            self._services = Configurator.__merge_dicts(
                service_default, cfg['services']
            )

        if 'channels' not in cfg:
            raise ConfiguratorException('No channel was specified in configuration')

        for name in cfg['channels']:
            row = Configurator.__merge_dicts(channel_default, cfg['channels'][name])
            row['name'] = name
            self._index[name] = pointer
            self._channels.append(self._validate(row))
            pointer += 1
        self.load()

    def load(self):
        if self._pointer < len(self._channels):
            rtn = self._channels[self._pointer]
            self._curr = CfgObject(rtn)
        else:
            self._curr = None
        return self

    def to_json(self, **kwargs):
        return self._curr.to_json(**kwargs)

    def channel(self, name):
        if name in self._index:
            rtn = self._channels[self._index[name]]
            return CfgObject(rtn)
        else:
            return None

    @property
    def services(self):
        return CfgObject(self._services)

    @property
    def base_path(self):
        return os.path.realpath(
            os.path.dirname(__file__) + '/..'
        )

    def get(self, key=None):
        if key is None:
            return self._curr
        if key in self._curr:
            rtn = self._curr[key]
            if isinstance(rtn, dict):
                return CfgObject(rtn)
            return rtn
        return None

    def __getattr__(self, item):
        return self.get[item]

    def __getitem__(self, item):
        return self.get[item]

    def reset(self):
        self._pointer = 0
        self.load()
        return self

    def next(self):
        if self._pointer <= len(self._channels):
            self._pointer += 1
            self.load()
        else:
            self._curr = None

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self._pointer < len(self._channels):
            rtn = self._curr
            self._pointer += 1
            self.load()
            return rtn
        else:
            raise StopIteration

    def __repr__(self):
        return 'Configurator(' + self.__str__() + ')'

    def __str__(self):
        return json.dumps(self.serialize(), cls = JsonEncoder, indent=2)

    def serialize(self):
        return {
            'channel_index': self._index,
            'channels': self._channels,
            'service': self._services
        }

    def _validate(self, row: dict) -> dict:
        if v := row['detector']['gaussian_blur']['kernel_size']:
            row['detector']['gaussian_blur']['kernel_size'] = tuple(v)
        if areas := row['capture']['areas']:
            if not isinstance(areas[0][0], list):
                areas = [areas]
                row['capture']['areas'] = areas
            contours = []
            x, y, w, h = (100000, 100000, 0, 0)
            for area in areas:
                contours.append(np.array(area))
                for pt in area:
                    if x > pt[0]: x = pt[0]
                    if y > pt[1]: y = pt[1]
                    if w < pt[0]: w = pt[0]
                    if h < pt[1]: h = pt[1]
            row['capture']['area_rect'] = (x, y, w, h)
            row['capture']['area_contours'] = contours
        else:
            row['capture']['area_rect'] = None
            row['capture']['area_contours'] = None
        if 'mqtt' not in row:
            row['mqtt'] = self.services['mqtt']
        else:

            row['mqtt'] = Configurator.__merge_dicts(
                self.services['mqtt'], row['mqtt']
            )
        return row

    @staticmethod
    def _load_from_file(file_path: str):
        result = None
        print('Loading configuration from file:', file_path)
        with open(file_path, 'r') as stream:
            file_ext = file_path[-5:].lower()
            if file_ext == '.yaml':
                result = yaml.safe_load(stream)
            elif file_ext == '.json':
                result = json.load(stream)
        return result

    @staticmethod
    def __merge_dicts(a, b):
        for key, val in a.items():
            if isinstance(val, dict):
                b_node = b.setdefault(key, {})
                if b_node is None:
                    b[key] = val
                else:
                    Configurator.__merge_dicts(val, b_node)
            else:
                if key not in b:
                    b[key] = val
        return b


class CfgObject:
    def __init__(self, cfg):
        if not isinstance(cfg, dict):
            cfg = {'value': cfg}
        self._cfg = cfg

    def get(self, key=None):
        if key is None:
            return self._cfg
        if key in self._cfg:
            rtn = self._cfg[key]
            if isinstance(rtn, dict):
                return CfgObject(rtn)
            return rtn
        return None

    def to_json(self, **kwargs):
        return json.dumps(self.serialize(), cls = JsonEncoder, **kwargs)

    def serialize(self):
        return self._cfg

    def __getattr__(self, key):
        rtn = self.get(key)
        if rtn is not None:
            return rtn

    def __getitem__(self, key):
        rtn = self.get(key)
        if rtn is not None:
            return rtn

    def __repr__(self):
        return 'CfgObject(' + self.__str__() + ')'

    def __str__(self):
        return json.dumps(self._cfg, cls = JsonEncoder, indent=2)


CV_CONFIG = Configurator()
