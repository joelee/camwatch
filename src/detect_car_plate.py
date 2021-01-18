
import numpy as np
import imutils
import cv2
import pytesseract
from config import CV_CONFIG, ConfiguratorException


class CarPlateDetection:
    _cfg = None
    _mqtt = None
    def __init__(self, img=None):
        cfg = CV_CONFIG.services.car_plate_recognition
        if not cfg.enabled:
            raise ConfiguratorException('Car Plate recognition service is not enabled')
        self._cfg = cfg
        self._min_ar = cfg.aspect_ratio[0]
        self._max_ar = cfg.aspect_ratio[1]
        self._ocr_opts = (
            f'-c tessedit_char_whitelist={cfg.ocr_characters} '
            f'--psm {cfg.tesseract_mode}'
        )
        if CV_CONFIG.services.mqtt.host:
            self._mqtt = CV_CONFIG.services.mqtt
        self._img = None
        self.reset()
        if img is not None:
            self.load_image(img)
            self.locate_plate()

    def reset(self):
        self.candidates = None
        self.plate_rect = None
        self.plate_img = None
        self.plate_text = None

    def load_image(self, img):
        self.reset()
        if len(img.shape) < 3:
            self._img = img
        else:
            self._img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self

    @property
    def image(self):
        return self._img

    def draw_rect(self,
                  img=None,
                  color=(0, 188, 0),
                  thickness=2,
                  show_ocr=False,
                  font_color=(255, 255, 255),
                  font_size=0.5
                  ):
        if not self.plate_rect:
            return None
        if img is None:
            img = self._img
        x, y, w, h = self.plate_rect
        cv2.rectangle(
            img,
            (x, y), (x + w, y + h),
            color,
            thickness
        )
        if show_ocr:
            cv2.rectangle(
                img,
                (x, y + h), (x + w, y + h + 20),
                color, -1
            )
            cv2.putText(
                img,
                self.plate_text,
                (x, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                font_color,
                1
            )
        return img

    def locate_plate(self):
        self.reset()
        plate_text_len = 0

        for cnt in self._detect_candidates():
            rect = cv2.boundingRect(cnt)
            (x, y, w, h) = rect
            aspect_ratio = w / float(h)
            if aspect_ratio >= self._min_ar and aspect_ratio <= self._max_ar:
                plate_img = self._img[y:y + h, x:x + w]
                ocr_out = pytesseract.image_to_string(
                    plate_img, config=self._ocr_opts
                )
                ocr_text = ocr_out.strip() if isinstance(ocr_out, str) else ''
                ocr_len = len(ocr_text)

                if ocr_text and ocr_len > 3 and ocr_len > plate_text_len:
                    self.plate_rect = rect
                    self.plate_img = plate_img
                    self.plate_text = ocr_text
                    plate_text_len = ocr_len

        return self.plate_rect

    def _detect_candidates(self):
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(
            self._img, cv2.MORPH_BLACKHAT, rect_kernel
        )

        square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(self._img, cv2.MORPH_CLOSE, square_kernel)
        light = cv2.threshold(
            light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]

        grad_x = cv2.Sobel(
        blackhat, ddepth=cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
        grad_x = grad_x.astype("uint8")
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        thresh = cv2.threshold(
            grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        candidates = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        candidates = imutils.grab_contours(candidates)
        candidates = sorted(candidates, key=cv2.contourArea, reverse=True)
        self.candidates = candidates[:self._cfg.max_candidate]
        return self.candidates
