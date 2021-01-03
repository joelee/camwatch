"""
MQTT Publisher helper
"""

import time
import cv2
import paho.mqtt.client as mqtt


class MqttHelper:
    def __init__(self, cfg):
        self.cfg = cfg
        self._enabled = bool(cfg.mqtt.host)
        self.prefix = cfg.mqtt.app + '/' + cfg.name + '/'
        self._client = None
        self._last_pub = {}

    @property
    def enabled(self):
        return self._enabled

    @property
    def client_id(self):
        return self.cfg.mqtt.app + '_' + self.cfg.name

    @property
    def client(self):
        if self._enabled and self._client is None:
            self._client = mqtt.Client()
        return self._client

    def connect(self):
        if self._enabled:
            self.client.connect(
                self.cfg.mqtt.host,
                port=self.cfg.mqtt.port,
                keepalive=self.cfg.mqtt.keepalive
            )
            self.client.loop_start()
            print('MQTT: Connected to ' + self.cfg.mqtt.host)
            self.publish('connected', 'yes')
        return self

    def disconnect(self):
        if self._enabled:
            print('MQTT: Disconnected')
            self.publish('connected', 'no')
            self.client.loop_stop()
            self._client.disconnect()
        return self

    def publish_event(self, event: str, snapshot, frame_id, reset=None):
        if not self._enabled:
            return False
        topic = f'{event}/snapshot'
        if reset and not self.pub_ready(topic, reset):
            return False
        ret = self.pub_image(topic, snapshot)
        if frame_id:
            self.publish(f'{event}/frame_id', frame_id)
        return ret

    def publish(self, topic, message):
        if self._enabled:
            self._last_pub[topic] = time.time()
            return self.client.publish(self.prefix + topic, message, 1)
        return None

    def pub_image(self, topic, image):
        if not self._enabled:
            return None
        img_str = cv2.imencode('.jpg', image)[1].tostring()
        self._last_pub[topic] = time.time()
        print(f'MQTT: Publish image: {self.prefix + topic}..')
        ret = self.client.publish(self.prefix + topic, img_str, 1)
        return ret

    def last_pub(self, topic):
        if topic not in self._last_pub:
            return None
        return time.time() - self._last_pub[topic]

    def pub_ready(self, topic, timeout):
        last_pub = self.last_pub(topic)
        if last_pub is None:
            return True
        return time.time() - last_pub < timeout
