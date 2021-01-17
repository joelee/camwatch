
import numpy as np
import imutils
import cv2
import pytesseract
from config import CV_CONFIG, ConfiguratorException


class CarPlateDetection:
    _cfg = None
    _mqtt = None
    def __init__(self):
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
        self._session = None
        self.reset()

    def reset(self):
        self._img = None
        self.candidates = None
        self.plate_contour = None
        self.plate_rect = None
        self.plate_img = None
        self.plate_ocr = None

    def set_session(self, session_id: str):
        self._session = session_id
        return self

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

    def ocr_plate(self):
        if self.plate_img is None:
            self.locate_plate()
            if self.plate_img is None:
                return None
        self._plate_ocr = pytesseract.image_to_string(
            self.plate_img,
            config=self._ocr_opts
        )
        return self.plate_ocr

    def locate_plate(self):
        plate_contour = None
        plate_img = None
        for cnt in self._detect_candidates():
            (x, y, w, h) = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            if aspect_ratio >= self._min_ar and aspect_ratio <= self._max_ar:
                plate_contour = cnt
                plate_img = self._img[y:y + h, x:x + w]
                break
        self.plate_contour = plate_contour
        self.plate_img = plate_img
        if plate_contour is not None:
            self.plate_rect = cv2.boundingRect(plate_contour)
        else:
            self.plate_rect = None
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


if __name__ == '__main__':
    img_file = '/Users/josephlee/Dropbox/Projects/camwatch-ai/var/res/car-ukreg-1.jpg'
    img = cv2.imread(img_file)

    creg = CarPlateDetection().load_image(img)
    cnp_text = creg.ocr_plate()
    x, y, w, h = cv2.boundingRect(creg.plate_contour)
    cv2.rectangle(
        img,
        (x, y), (x + w, y + h),
        (255, 0, 0),
        2
    )

    print('OCRed Number Plate:', cnp_text)
    cv2.imshow("Number Plate", img)
    cv2.waitKey(0)
