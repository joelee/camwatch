"""
Capture Class
"""

import sys
import time
import signal
import cv2
from collections import deque

from frame import Frame, KeyFrame
from tasks import execute_task, execute_threading
from mqtt_helper import MqttHelper
from video_writer import VideoWriter

from config import CV_CONFIG


class CaptureException(Exception):
    pass


class Counter:
    motion: int = 0
    face: int = 0
    car: int = 0

    def reset(self):
        self.motion = 0
        self.face = 0
        self.car = 0


class Capture:
    completed: bool = False
    last_motion_ts = 0
    c_post_motion = 0
    counter = Counter()
    frame = None
    _key_frame = KeyFrame()

    def __init__(self, channel: str):
        self.name = channel
        self.start_ts = time.time()
        cfg = CV_CONFIG.channel(channel)
        if cfg is None:
            raise CaptureException(
                f'Channel "{channel}" not found in configuration'
            )
        if cfg.capture.source is None:
            raise CaptureException(
                f'Channel "{channel}" does not have any capture source'
            )

        self._q_prev_blur = deque(maxlen=cfg.detector.frames_ago)
        self._pre_cap = deque(maxlen=cfg.capture.fps * cfg.capture.pre_motion)
        self._f_post_motion = cfg.capture.fps * cfg.capture.post_motion
        self._f_t_wait = int(1000 / cfg.capture.fps)
        self._uptime = cfg.capture.uptime

        self._cap = cv2.VideoCapture(cfg.capture.source)
        if cfg.capture.width:
            self._cap.set(3, cfg.capture.width)
        if cfg.capture.height:
            self._cap.set(4, cfg.capture.height)

        if cfg.writer.location:
            self._writer = VideoWriter(cfg)
        else:
            self._writer = None

        self.cfg = cfg

        self.mqtt = MqttHelper(cfg)
        self._key_frame.reset()

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    @property
    def frame_id(self):
        """
        Recording Session Frame Id
        to identify recording file and Frame number
        """
        if self._writer is None:
            return None
        return self._writer.session_frame_id

    @property
    def key_frame(self):
        if self._key_frame.empty():
            return self.frame
        if self._key_frame.face.frame is not None:
            return self._key_frame.face.frame
        if self._key_frame.car.frame is not None:
            return self._key_frame.car.frame
        return self._key_frame.motion.frame

    def start(self):
        self.completed = False
        self.mqtt.connect()
        cfg = self.cfg
        while not self.completed:
            _, img = self._cap.read()
            if img is None:
                print('End of capture stream.')
                self.completed = True
                break
            self.frame = Frame(img, self.cfg)

            self._task_prep_image()

            if cfg.capture.show_frame:
                cv2.imshow(self.name, img)

            if self._uptime and self._uptime < time.time() - self.start_ts:
                print('Uptime reached. Stopping...')
                self.completed = True

            ch = cv2.waitKey(self._f_t_wait)
            if ch == 27:
                print('Escape key pressed. Stopping...')
                self.completed = True

        if self._writer:
            self._writer.close()
        cv2.destroyAllWindows()
        self.mqtt.disconnect()
        print('Capture stream has been terminated.')
        print(f'Elapsed time: {time.time() - self.start_ts}')
        sys.exit(0)

    def _task_prep_image(self):
        ret = execute_task('prep', self.frame)

        self._q_prev_blur.append(ret.img_blur)
        if len(self._q_prev_blur) > 1:
            self._task_detect_motion()

    def _task_detect_motion(self):
        self.frame.set('base_img', self._q_prev_blur[0])
        if self.cfg.detector.save_image_path:
            self.frame.set(
                'save_motion_img_path',
                self.cfg.detector.save_image_path
            )
        ret = execute_task('motion', self.frame)
        log_state = None
        if ret.motion:
            self.counter.motion += 1
            if self.counter.motion == 1:
                log_state = 'STARTED'
                if self.mqtt.enabled:
                    self.mqtt.publish_event(
                        'motion_start', ret.image,
                        self.frame_id, self.cfg.mqtt.motion_reset
                    )
            elif self.c_post_motion != self._f_post_motion:
                log_state = 'RESUME'
            self.c_post_motion = self._f_post_motion
            self.last_motion_ts = time.time()
            self._key_frame.motion.frame = self.frame
            if self._key_frame.motion.area < ret.motion_area:
                self._key_frame.motion.frame = ret.frame
                self._key_frame.motion.area = ret.motion_area

        elif self.c_post_motion > 0:
            if self.c_post_motion == self._f_post_motion:
                log_state = 'IDLE'
            self.c_post_motion -= 1
            if self.c_post_motion == 0:
                self._writer.close()
                self.counter.reset()
                if self.mqtt.enabled:
                    self.mqtt.publish_event(
                        'motion_completed', self._key_frame.motion.frame.image,
                        self.frame_id, self.cfg.mqtt.motion_reset
                    )
                self._key_frame.reset()
                log_state = 'STOPPED'

        if  self._writer:
            self._write_frame()

        if log_state is not None:
            print(
                f'Motion {log_state} on frame {self.frame_id or ""}: '
                f'c={self.counter.motion}, '
                f'ma={ret.motion_area}, t={ret.motion_ts}'
            )

        if ret.motion:
            self._task_detect_objects()

    def _task_detect_objects(self):
        fd_cfg = self.cfg.face_detect
        cd_cfg = self.cfg.car_detect
        tasks = []
        if fd_cfg.enabled and self.counter.face < fd_cfg.max_face:
            tasks.append('face')
        if cd_cfg.enabled and self.counter.car < cd_cfg.max_car:
            tasks.append('car')

        if not tasks:
            return

        ret = execute_threading(tasks, self.frame)
        if ret.face_detected:
            self._task_face_detected(ret)
        if ret.car_detected:
            self._task_car_detected(ret)

    def _task_face_detected(self, ret):
        self.counter.face += 1
        print(
            'Face DETECTED on frame '
            f'{self.frame_id}: c={ret.face_detected}, '
            f'ma={ret.face_area}, t={ret.face_ts}'
        )

        if self._key_frame.face.area < ret.face_area:
            self._key_frame.face.frame = ret
            self._key_frame.face.area = ret.face_area
            if self.mqtt.enabled:
                self.mqtt.publish_event(
                    'face_detected', ret.image,
                    self.frame_id, self.cfg.mqtt.face_reset
                )

    def _task_car_detected(self, ret):
        self.counter.car += 1
        print(
            'Car DETECTED on frame '
            f'{self.frame_id}: c={ret.car}, '
            f'ma={ret.car_area}, t={ret.car_ts}'
        )
        if self._key_frame.car.area < ret.car_area:
            self._key_frame.car.frame = ret
            self._key_frame.car.area = ret.car_area
            if self.mqtt.enabled:
                self.mqtt.publish_event(
                    'car_detected', ret.image,
                    self.frame_id, self.cfg.mqtt.car_reset
                )

    def _write_frame(self):
        # Always record frames
        if self.cfg.writer.record_on == 'always':
            self._writer.write(self.frame.image)
            return

        # Record frames on motion
        elif self.cfg.writer.record_on == 'motion':
            # Motion detected
            if self.frame.motion:
                # Reset post-frame counter
                self.c_post_motion = self._f_post_motion
                # Write pre-frame from buffer
                if len(self._pre_cap) > 0:
                    while self._pre_cap:
                        self._writer.write(self._pre_cap.pop())
                # Write frame
                self._writer.write(self.frame.image)

            # Motion stopped: Keep recording on
            # post-frame countdown
            elif self.c_post_motion > 0:
                self._writer.write(self.frame.image)

            # Buffer pre-capture frames
            else:
                self._pre_cap.append(self.frame.image)

    def shutdown(self, signum, frame):
        print(f'Shutting down ({signum})...')
        self.completed = True


if __name__ == "__main__":
    channel = sys.argv[1]
    if channel:
        Capture(channel).start()
    else:
        print(f'Usage: {sys.argv[1]} channel_name')
        sys.exit(1)
