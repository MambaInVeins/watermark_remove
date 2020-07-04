# uncompyle6 version 3.7.2
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.9 (default, Apr 18 2020, 01:56:04) 
# [GCC 8.4.0]
# Embedded file name: ./autowatermarkremoval/remover.py
# Compiled at: 2019-11-22 09:39:26
# Size of source mod 2**32: 5952 bytes
import cv2, os, pathlib, logging, traceback
from . import utils
from matplotlib import pyplot as plt
import numpy as np
log = logging.getLogger(__name__)
KERNEL_SIZE = 3

class Remover(object):

    def __init__(self, watermark_path: pathlib.Path='', alpha_path: pathlib.Path='', image_path: pathlib.Path='', same_position=True):
        self.same_position = same_position
        if watermark_path.is_absolute():
            watermark_path = watermark_path.absolute()
        if alpha_path.is_absolute():
            alpha_path = alpha_path.relative_to('/')
        else:
            if image_path.is_absolute():
                image_path = image_path.relative_to('/')
            else:
                self.watermark = cv2.imread(watermark_path.absolute().__str__())
                assert self.watermark is not None, 'Can not load watermark image from %s' % watermark_path
                self.alpha = cv2.imread(alpha_path.absolute().__str__())
                assert self.alpha is not None, 'Can not load watermark alpha image from %s' % alpha_path
            self.src_image = cv2.imread(image_path.absolute().__str__())
            assert self.src_image is not None, 'Can not load source image from %s' % image_path

    def calc_image_gradient(self, img):
        """
        watermark on the same postion of each image
        estimate the watermark (grad(W) = median(grad(J)))
        Also, give the list of gradients, so that further processing can be done on it
        """
        log.info('Computing single image gradient')
        image_gradx = cv2.Sobel(img, (cv2.CV_64F), 1, 0, ksize=KERNEL_SIZE)
        image_grady = cv2.Sobel(img, (cv2.CV_64F), 0, 1, ksize=KERNEL_SIZE)
        return (
         image_gradx, image_grady)

    def get_watermark_pos_chamfer_matching(self, src_image, pattern):
        methods = [
         'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
         'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        h, w = self.watermark.shape[:-1]
        for meth in methods:
            img = src_image.copy()
            method = eval(meth)
            res = cv2.matchTemplate(img, pattern, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (
             top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img, top_left, bottom_right, 255, 2)
            print(top_left, bottom_right)
            (plt.subplot(121), plt.imshow(res, cmap='gray'))
            (plt.title('Matching Result'), plt.xticks([]), plt.yticks([]))
            (plt.subplot(122), plt.imshow(img, cmap='gray'))
            (plt.title('Detected Point'), plt.xticks([]), plt.yticks([]))
            (plt.subplot(131), plt.imshow((self.watermark), cmap='gray'))
            (plt.title('Watermark'), plt.xticks([]), plt.yticks([]))
            plt.suptitle(meth)
            plt.show()

    def _detect_watermark(self, image, gx, gy, thresh_low=200, thresh_high=220, printval=False):
        """ Compute a verbose edge map using Canny edge detector, take its magnitude.
            Assuming cropped values of gradients are given.
            Returns image, start and end coordinates
            """
        watermark = np.sqrt(np.square(gx) + np.square(gy)) / np.mean(np.sqrt(np.square(gx) + np.square(gy)))
        print(image.shape, watermark.shape)
        image_edgemap = cv2.Canny(image, thresh_low, thresh_high)
        self.save_image(image_edgemap, 'image_edgemap.jpg')
        print(utils.jpg_2_pgm('tmp/image_edgemap.jpg'))
        chamfer_dist = cv2.filter2D(image_edgemap.astype(float), -1, watermark)
        print(chamfer_dist)
        exit(0)
        log.error(watermark.shape)
        rect = watermark.shape
        index = np.unravel_index(np.argmax(chamfer_dist), image.shape)
        log.info(f"detect watermark unraval_index {index}")
        log.info(rect)
        x, y = index[0] - rect[0] / 2, index[1] - rect[1] / 2
        x = int(x)
        y = int(y)
        self.get_watermark_pos_chamfer_matching(image, watermark)
        return (
         x, y, x + rect[0], y + rect[1])

    def remove(self):
        wm_image_gradx, wm_image_grady = self.calc_image_gradient(self.watermark)
        min_x, min_y, max_x, max_y = self._detect_watermark(self.src_image, wm_image_gradx, wm_image_grady)
        top_left = (
         min_x, min_y)
        bottom_right = (max_x, max_y)
        cv2.rectangle(self.src_image, (min_x, min_y), (max_x, max_y), 255, 2)
        print(top_left, bottom_right)
        (plt.subplot(121), plt.imshow((self.src_image), cmap='gray'))
        (plt.title('Matching Result'), plt.xticks([]), plt.yticks([]))
        (plt.subplot(122), plt.imshow((self.src_image), cmap='gray'))
        (plt.title('Detected Point'), plt.xticks([]), plt.yticks([]))
        (plt.subplot(131), plt.imshow((self.watermark), cmap='gray'))
        (plt.title('Watermark'), plt.xticks([]), plt.yticks([]))
        plt.suptitle('love')
        plt.show()


def remove(watermark_path, alpha_path, image_path):
    remover = Remover(watermark_path, alpha_path, image_path)
    remover.remove()
# okay decompiling remover.pyc
