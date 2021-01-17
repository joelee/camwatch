
import time
from collections import deque
import numpy as np
import cv2
import threading
# import multiprocessing

from frame import Frame
from detect_motion import detect_motion
from detect_face_dlib import detect_face
from detect_car import detect_car


def prep_image(frame: Frame) -> dict:
    cfg = frame.cfg.detector
    contours = frame.cfg.capture.area_contours
    if not contours:
        img = frame.image
    else:
        # Mask detection area
        stencil = np.zeros(frame.image.shape).astype(frame.image.dtype)
        cv2.fillPoly(stencil, contours, [255, 255, 255])
        img = cv2.bitwise_and(frame.image, stencil)
        if frame.cfg.capture.area_rect:
            x, y, w, h = frame.cfg.capture.area_rect
            img = img[y:h, x:w]
        if frame.cfg.capture.draw_areas:
            colour = frame.cfg.capture.draw_area_color
            for pts in frame.cfg.capture.area_pts:
                cv2.polylines(frame.image, [pts], True, colour)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(
        img_gray,
        cfg.gaussian_blur.kernel_size,
        cfg.gaussian_blur.sigmax
    )

    return {
        'img_gray': img_gray,
        'img_blur': img_blur
    }


TASK = {
    'prep': prep_image,
    'motion': detect_motion,
    'face': detect_face,
    'car': detect_car
}


def execute_task(task: str, frame: Frame) -> Frame:
    if task not in TASK:
        raise Exception(f'Task "{task}" not found.')
    frame.new_task(task)
    ret = TASK[task](frame)
    return frame.set_dict(ret)


def execute_sync(tasks: list, frame: Frame) -> Frame:
    if len(tasks) == 1:
        return execute_task(tasks.pop(), frame)
    ts = time.time()
    for task in tasks:
        execute_task(task, frame)

    print('SingleProcess', tasks, ':', time.time() - ts)
    return frame


def execute_threading(tasks: list, frame: Frame) -> Frame:
    if len(tasks) == 1:
        return execute_task(tasks.pop(), frame)
    ts = time.time()
    ret_q = deque()

    def store_ret_q(func):
        def wrapper(*args):
            ret_q.append(func(*args))
        return wrapper

    @store_ret_q
    def exec_task(l_task: str, l_frame: Frame):
        if l_task not in TASK:
            raise Exception(f'Task "{task}" not found.')
        frame.new_task(l_task)
        ret = TASK[l_task](l_frame)
        return ret

    threads = list()
    for task in tasks:
        thread = threading.Thread(
            target=exec_task,
            args=(task, frame)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    while len(ret_q) > 0:
        frame.set_dict(ret_q.popleft())
    print('MultiThreading', tasks, ':', time.time() - ts)
    return frame


# def exec_mp_task(task: str, frame: Frame):
#     ret = {}
#     print('exec_mp_task', task)
#     if task not in TASK:
#         raise Exception(f'Task "{task}" not found.')
#     frame.new_task(task)
#     ret[task] = TASK[task](frame)
#
#
# def execute_multiprocessing(tasks: list, frame: Frame) -> Frame:
#     if len(tasks) == 1:
#         return execute_task(tasks.pop(), frame)
#     ts = time.time()
#
#     manager = multiprocessing.Manager()
#     return_dict = manager.dict()
#
#     processes = []
#     for task in tasks:
#         print('func:', exec_mp_task)
#         print('task:', task, frame, return_dict)
#         f = frame.clone(task)
#         print('clone:', f)
#         p = multiprocessing.Process(
#             target=exec_mp_task,
#             args=(task, f)
#         )
#         print('p', p)
#         processes.append(p)
#         print('processes', processes)
#         p.start()
#         print('p started', p)
#
#     for p in processes:
#         p.join()
#
#     # for task in return_dict:
#     #     frame.set_dict(return_dict[task])
#
#     print('MultiProcessing', tasks, ':', time.time() - ts)
#     return frame
