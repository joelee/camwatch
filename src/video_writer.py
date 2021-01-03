"""
CvUtils: Video Writer
"""

import time
import os
from datetime import datetime
import threading
import cv2


class VideoWriter:
    def __init__(self, config):
        cfg = config.writer
        self._base_path = cfg.location
        self._name = config.name or 'CvVideo'
        if cfg.file_encoding:
            self._encoding = cfg.file_encoding
        elif cfg.file_format[-4:] == '.avi':
            self._encoding = 'xvid'
        else:
            self._encoding = 'mp4v'
        self._fourcc = cv2.VideoWriter_fourcc(*self._encoding)
        self._fps = int(cfg.fps) or 15
        if cfg.width and cfg.height:
            size = (int(cfg.width), int(cfg.height))
        else:
            size = None
        self._size = size
        self._template = cfg.file_format\
            .replace('{name}', self._name)\
            .replace('{enc}', self._encoding)
        self._kf_template = cfg.kf_file_format\
            .replace('{name}', self._name)
        self._video_len = config.capture.video_len or 0
        self._filename = None
        self._session = None

        self._writer = None
        self._started = None
        self._frame_num = 0

    @property
    def active(self) -> bool:
        return self._writer is not None

    @property
    def elapse_time(self):
        if self._started is None:
            return None
        return time.time() - self._started

    @property
    def session_id(self):
        if not self._session:
            return None
        return self._session

    @property
    def frame_num(self):
        return self._frame_num

    @property
    def session_frame_id(self):
        if not self._session:
            return None
        return self._session + '?' + str(self._frame_num)

    @property
    def filename(self):
        if self._filename is None:
            self._new_filename()
        return self._filename

    @property
    def kf_filename(self):
        ts = datetime.now()
        file = os.path.join(
            self._base_path, ts.strftime(self._kf_template)
        )
        os.makedirs(os.path.dirname(file), exist_ok=True)
        return file

    def next_file(self):
        if self._writer:
            self._writer.release()
        self._writer = None
        self._new_filename()
        self._started = time.time()
        return self

    def _new_filename(self):
        self._session = datetime.now().strftime(self._template)
        self._filename = os.path.join(
            self._base_path, self._session
        )
        self._frame_num = 0
        os.makedirs(os.path.dirname(self._filename), exist_ok=True)
        print('Writer: new file: ' + self._filename)

    def write_key_frame(self, img):
        cv2.imwrite(self.kf_filename, img)
        return self

    def write(self, img):
        if self._started is None:
            self._started = time.time()
        elif self._video_len and self.elapse_time > self._video_len:
            self.next_file()
        if self._writer is None:
            self._writer = cv2.VideoWriter(
                self.filename, self._fourcc,
                self._fps, self._get_size(img)
            )
        self._writer.write(img)
        self._frame_num += 1
        return self

    def close(self):
        if self._writer is None:
            return self
        print('Writer - stopped:', self._filename)
        self._writer.release()
        self._writer = None
        self._started = None
        self._frame_num = 0
        self._filename = None
        self._session = None
        return self

    def _get_size(self, img):
        if self._size is None:
            self._size = (img.shape[1], img.shape[0])
        return self._size
