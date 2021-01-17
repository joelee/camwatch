
from config import CfgObject

class Frame:
    def __init__(self,
                 image,
                 cfg: CfgObject,
                 **kwargs):
        self._task = None
        # self._frame = frame
        self._image = image
        self._cfg = cfg
        self._data = kwargs

    @property
    def task(self) -> str:
        return self._task

    # @property
    # def frame(self) -> int:
    #     return self._frame

    @property
    def image(self):
        return self._image

    @property
    def cfg(self):
        return self._cfg

    def __getattr__(self, item):
        if item == 'data':
            return self._data
        if item in self._data:
            return self._data[item]

    def get(self, item: str):
        return self._data.get(item, None)

    def set(self, item: str, value):
        self._data[item] = value
        return self

    def set_dict(self, items: dict):
        if items:
            self._data = {**self._data, **items}
        return self

    def new_task(self, task_name: str):
        self._task = task_name
        return self

    def clone(self, task_name=None):
        if task_name is None:
            task_name = self._task
        return Frame(
            self._frame,
            self._image,
            self._cfg
        ).set_dict(self._data).new_task(task_name)

    def values(self):
        rtn = {
            'name': self._name,
            'frame': self._frame,
            'image': self._image
        }
        return {**rtn, **self._data}


class FrameData:
    frame = None
    area: int = 0
    count: int = 0
    frame_num: int = 0

    def reset(self):
        self.frame = None
        self.area = 0
        self.count = 0
        self.frame_num = 0


class KeyFrame:
    motion: FrameData = FrameData()
    face: FrameData = FrameData()
    car: FrameData = FrameData()

    def reset(self):
        self.motion.reset()
        self.face.reset()
        self.car.reset()

    def empty(self):
        return (
            self.motion.frame is None and
            self.face.frame is None and
            self.car.frame is None
        )
