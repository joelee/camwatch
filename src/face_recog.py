
import time
import os
import numpy as np
import cv2
import dlib
import face_recognition as fr
from face_trainer import FaceTrainer
from config import CV_CONFIG, ConfiguratorException

NAME_ENCODINGS = FaceTrainer().load()


class FaceRecognition:
    _cfg = None
    _mqtt = None
    def __init__(self):
        cfg = CV_CONFIG.services.face_recognition
        if not cfg.enabled:
            raise ConfiguratorException('Face recognition service is not enabled')
        if not cfg.location:
            raise ConfiguratorException('Face recognition location path is not set')
        self._cfg = cfg
        if CV_CONFIG.services.mqtt.host:
            self._mqtt = CV_CONFIG.services.mqtt
        self.set_scale(cfg.scale)
        self.set_min_score(cfg.min_score)
        self._session = None
        self.reset()

    def reset(self):
        self._img = None
        self._cv2_img = None
        self._encs = None
        self._locations = None
        self._cache_recognition = None
        self._cache_names = None

    def set_session(self, session_id: str):
        self._session = session_id
        return self

    def load_image(self, img):
        self.reset()
        self._cv2_img = img
        return self

    def load_image_file(self, filepath, flags=cv2.IMREAD_COLOR):
        return self.load_image(cv2.imread(filepath, flags))

    def load_image_str(self, str_value, flags=cv2.IMREAD_COLOR):
        img = np.fromstring(str_value, dtype=np.uint8);
        return self.load_image(cv2.imdecode(img, flags))

    @property
    def image(self):
        if self._img is None:
            if self._scale > 0 and self._scale < 1:
                frame = cv2.resize(
                    self._cv2_img, (0, 0),
                    fx=self._scale, fy=self._scale
                )
            else:
                frame = self._cv2_img
            self._img = frame[:, :, ::-1]
        return self._img

    @property
    def cv2_image(self):
        return self._cv2_img

    @property
    def encodings(self):
        if self._encs is None:
            self._encs = fr.face_encodings(
                self.image, self.locations, self._cfg.num_jitters,
                'large' if self._cfg.use_large_model else 'small'
            )
        return self._encs

    @property
    def locations(self):
        if self._locations is None:
            locs = fr.face_locations(
                self.image,
                self._cfg.n_upsample,
                'cnn' if self._cfg.cuda_model else 'hog'
            )
            if self._scale > 0 and self._scale < 1:
                s = 1 / self._scale
                rs_locs = []
                for x, y, w, h in locs:
                    x *= s; y *= s; w *= s; h *= s
                    rs_locs.append((
                        int(x), int(y), int(w), int(h)
                    ))
                locs = rs_locs
            self._locations = locs
        return self._locations

    def draw_rects(self, ):
        names = self.name_all()
        i = 0
        known_color = (0, 255, 0)
        unknown_color = (0, 0, 255)
        for loc in self.locations:
            t, r, b, l = self._apply_offset(loc)
            name = names[i]
            i += 1
            color = known_color if name else unknown_color
            cv2.rectangle(self._cv2_img, (l, t), (r, b), color, 1)
            if name:
                cv2.rectangle(
                    self._cv2_img, (l, b - 35), (r, b), color, cv2.FILLED
                )
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    self._cv2_img, name, (l + 6, b - 6),
                    font, 0.75, (18, 0, 0), 1
                )
        return self._cv2_img

    def recognise_one(self, index=0):
        if self._cache_recognition is not None:
            return self._cache_recognition[index]
        return self._compare(self.encodings[index])

    def name_one(self, index=0):
        if self._cache_names is not None:
            return self._cache_names[index]
        return self._get_name(self.encodings[index])

    def recognise_all(self):
        if self._cache_recognition is None:
            ret = []
            for enc in self.encodings:
                ret.append(self._compare(enc))
            self._cache_recognition = ret
        return self._cache_recognition

    def name_all(self):
        if self._cache_names is None:
            ret = []
            for enc in self.encodings:
                ret.append(self._get_name(enc))
            self._cache_names = ret
        return self._cache_names

    def set_scale(self, scale: float):
        if scale > 0 and scale < 1:
            self._scale = scale
        else:
            self._scale = 1.0
        return self

    def set_min_score(self, score: float):
        if score > 0 and score < 1:
            self._min_score = score
        else:
            self._min_score = 0.75
        return self

    def _compare(self, face_encoding):
        ret = {}
        for name in NAME_ENCODINGS:
            encs = NAME_ENCODINGS[name]
            comp = fr.compare_faces(encs, face_encoding, self._cfg.tolerance)
            ret[name] = comp.count(True) / len(encs)
        return ret

    def _get_name(self, face_encoding):
        ret_name = None
        val = 0
        for name in NAME_ENCODINGS:
            encs = NAME_ENCODINGS[name]
            comp = fr.compare_faces(encs, face_encoding, self._cfg.tolerance)
            c_val = comp.count(True) / len(encs)
            if val < c_val:
                ret_name = name
                val = c_val
            elif val == c_val and ret_name is not None:
                ret_name += ', ' + name

        if val < self._min_score:
            ret_name = None

        return ret_name

    def _apply_offset(self, loc: tuple, offset=(0.25, 0.45)):
        h, v = offset
        t, r, b, l = loc
        if h < 1.0: h = int((r - l) * h)
        if v < 1.0: v = int((b - t) * v)
        t -= v; r += h; b += v; l -= h
        if t < 0: t = 0
        if l < 0: l = 0
        shape = self._cv2_img.shape
        if r > shape[1]: r = shape[1]
        if b > shape[0]: b = shape[0]
        return (t, r, b, l)


if __name__ == '__main__':
    r_path = '/Users/josephlee/Dropbox/Projects/camwatch-ai/var/res'

    detector = dlib.get_frontal_face_detector()

    for name in os.listdir(r_path):
        if name[-4:] == '.jpg':
            f = r_path + '/' + name
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        start_ts = time.time()
        dets = detector(img, 1)
        end_ts = time.time() - start_ts
        print("Number of faces detected: {}".format(len(dets)))
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom()))
        print('Detection time:', end_ts)




    # for name in os.listdir(r_path):
    #     if name[-4:] == '.jpg':
    #         img_file = r_path + '/' + name
    #         print('Scanning:', name)
    #         recon = FaceRecognition().load_image_file(img_file)
    #         start_ts = time.time()
    #         # print(recon.locations)
    #         # print(recon.recognise_all())
    #         print(recon.name_all())
    #         img = recon.draw_rects()
    #         print('TS:', time.time() - start_ts)
    #         cv2.imshow('video', img)
    #         ch = cv2.waitKey()
    #         if ch == 27:
    #             print('Escape key pressed. Stopping...')
    #             break
    # cv2.destroyAllWindows()
