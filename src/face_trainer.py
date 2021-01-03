
import os
import json
import numpy as np
import face_recognition as fr

from config import CV_CONFIG, ConfiguratorException
from utils import NumpyEncoder


class FaceTrainer:
    def __init__(self):
        cfg = CV_CONFIG.services.face_recognition
        if not cfg.enabled:
            raise ConfiguratorException('Face recognition service is not enabled')
        if not cfg.location:
            raise ConfiguratorException('Face recognition location path is not set')
        self._path = cfg.location
        if self._path[0:1] != '/':
            self._path = CV_CONFIG.base_path + '/' + self._path
        self._face_encoding_file = self._path + '/face_encodings.json'
        self._cfg = cfg

    def start(self):
        print('FaceTrainer: start training on ' + self._path)
        result = {}
        for name, file in self._photo_generator():
            if name not in result:
                print(f'Training to recognise "{name}"...')
                result[name] = []
            image = fr.load_image_file(file)
            result[name].append(fr.face_encodings(
                image, None, self._cfg.num_jitters,
                'large' if self._cfg.use_large_model else 'small'
            )[0])
        with open(self._face_encoding_file, 'w') as fh:
            json.dump(result, fh, cls=NumpyEncoder)
        print('Encoding file saved')

    def load(self):
        ret = {}
        with open(self._face_encoding_file) as fh:
            data = json.load(fh)
            for name in data:
                ret[name] = []
                for enc in data[name]:
                    ret[name].append(np.asarray(enc))
        return ret

    def _photo_generator(self):
        for name in os.listdir(self._path):
            path = self._path + '/' + name
            if os.path.isdir(path):
                files = os.listdir(path)
                for file in files:
                    if file[-4:].lower() in ('.jpg', '.png'):
                        filepath = path + '/' + file
                        yield name, filepath


if __name__ == "__main__":
    FaceTrainer().start()
