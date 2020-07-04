# uncompyle6 version 3.7.2
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.9 (default, Apr 18 2020, 01:56:04) 
# [GCC 8.4.0]
# Embedded file name: ./autowatermarkremoval/solver.py
# Compiled at: 2019-11-26 10:02:11
# Size of source mod 2**32: 19727 bytes
import pathlib, logging, os, traceback, cv2, copy
from typing import Tuple, List
from . import utils, estimatewatermark
from . import closed_form_matting
from . import reconstructwatermark
import numpy as np
from .utils import load_images, PlotImage
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)
KERNEL_SIZE = 3

def show_single_image(img):
    plt.subplot(1, 1, 1)
    plt.imshow(PlotImage(img))
    plt.draw()
    plt.pause(10)


class Solver(object):
    _image_root_path = ''
    _raw_images = None
    _normalized_images = {}
    _watermark_positions = {}

    def __init__(self, source_path: pathlib.Path='', dest_path: pathlib.Path='', same_position=True):
        self.same_position = same_position
        if source_path.is_absolute():
            source_path = source_path.relative_to('/')
        self._image_source_path = os.path.join(self._image_root_path, source_path)
        self._image_dest_path = os.path.join(self._image_root_path, dest_path)
        if not os.path.exists(self._image_source_path):
            log.error(f"image source folder {self._image_source_path} doesn't exist")
        if not os.path.exists(self._image_dest_path):
            log.warn("dest path {self._image_dest_path} doesn't exist")
        self._raw_images = utils.load_images(source_path)
        if len(self._raw_images) == 0:
            log.error(f"no image in folder {source_path}")
        max_row = 0
        max_column = 0
        for img in self._get_dict_values(self._raw_images):
            max_row = max(max_row, img.shape[0])
            max_column = max(max_column, img.shape[1])

        for path, image in self._raw_images.items():
            self._normalized_images[path] = self._normalize_image(image, (max_row, max_column))

        self._rectangle_top_left = None
        self._rectangle_bottom_right = None

    def has_decimal(self, num):
        """ if number is int , return False.
        """
        if int(num) == num:
            return False
        else:
            return True

    def _normalize_image(self, image, final_shape):
        image_shape = image.shape
        if self.has_decimal((final_shape[0] - image_shape[0]) / 2.0):
            top = int((final_shape[0] - image_shape[0]) / 2 + 0.5)
            bottom = int((final_shape[0] - image_shape[0]) / 2)
        else:
            top = bottom = (final_shape[0] - image_shape[0]) / 2.0
        if self.has_decimal((final_shape[1] - image_shape[1]) / 2.0):
            left = int((final_shape[1] - image_shape[1]) / 2 + 0.5)
            right = int((final_shape[1] - image_shape[1]) / 2)
        else:
            left = right = (final_shape[1] - image_shape[1]) / 2.0
        if not int(left) == left:
            raise AssertionError
        else:
            left = int(left)
            assert int(right) == right
            right = int(right)
            assert int(top) == top
            top = int(top)
            assert int(bottom) == bottom
        bottom = int(bottom)
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, [0, 0, 0])

    def _get_dict_values(self, d):
        return list(d.values())

    def get_mouse_position(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            log.info(f"left button press down , position =( {x}, {y} )")
            self._rectangle_top_left = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            log.info(f"left button release up, position =( {x}, {y} )")
            self._rectangle_bottom_right = (x, y)

    def _user_crop_watermark(self, image):
        cv2.imshow('current image', image)
        cv2.setMouseCallback('current image', self.get_mouse_position)
        while cv2.waitKey(20):
            if 255 == 27:
                break
            elif self._rectangle_top_left:
                if self._rectangle_bottom_right:
                    log.info(f"user selected watermark range {self._rectangle_top_left} {self._rectangle_bottom_right}")
                    cv2.rectangle(image, self._rectangle_top_left, self._rectangle_bottom_right, (0,
                                                                                                  255,
                                                                                                  0), 3)
                    cv2.imshow('current image', image)
                    cv2.waitKey(0)
                    break

        cv2.destroyAllWindows()

    def _estimate_watermark_manual(self):
        log.warning('Not implement yet.')

    def _estimate_watermark(self) -> Tuple[(float, float, np.ndarray, np.ndarray)]:
        """
        watermark on the same postion of each image
        estimate the watermark (grad(W) = median(grad(J)))
        Also, give the list of gradients, so that further processing can be done on it
        """
        log.info('Computing gradients')
        images_gradx = [cv2.Sobel(img, (cv2.CV_64F), 1, 0, ksize=KERNEL_SIZE) for path, img in self._normalized_images.items()]
        images_grady = [cv2.Sobel(img, (cv2.CV_64F), 0, 1, ksize=KERNEL_SIZE) for path, img in self._normalized_images.items()]
        log.info('Computing median gradients.')
        Wm_x = np.median((np.array(images_gradx)), axis=0)
        Wm_y = np.median((np.array(images_grady)), axis=0)
        return (Wm_x, Wm_y, images_gradx, images_grady)

    def _crop_watermark(self, gradx, grady, threshold=0.4, boundary_size=2):
        """ Crops the watermark by taking the edge map of magnitude of grad(W)
        Assumes the gradx and grady to be in 3 channels
        @param: threshold - gives the threshold param
        @param: boundary_size - boundary around cropped image
        """
        W_mod = np.sqrt(np.square(gradx) + np.square(grady))
        W_mod = utils.PlotImage(W_mod)
        W_gray = utils.image_threshold(np.average(W_mod, axis=2),
          threshold=threshold)
        x, y = np.where(W_gray == 1)
        xm, xM = np.min(x) - boundary_size - 1, np.max(x) + boundary_size + 1
        ym, yM = np.min(y) - boundary_size - 1, np.max(y) + boundary_size + 1
        return (gradx[xm:xM, ym:yM, :], grady[xm:xM, ym:yM, :])

    def _detect_watermark(self, image, gx, gy, thresh_low=100, thresh_high=220, printval=False):
        """ Compute a verbose edge map using Canny edge detector, take its magnitude.
            Assuming cropped values of gradients are given.
            Returns image, start and end coordinates
            """
        methods = [
         'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
         'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED',
         'CANNY_2D_FILTER']
        watermark = np.average((np.sqrt(np.square(gx) + np.square(gy))), axis=2)
        image_edgemap = cv2.Canny(image, thresh_low, thresh_high)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        watermark_gray = watermark.astype(np.float32)
        t_w, t_h = watermark.shape[::-1]
        results = {}
        plt.figure(1)
        for i, meth in enumerate(methods):
            if meth == 'CANNY_2D_FILTER':
                chamfer_dist = cv2.filter2D(image_edgemap.astype(float), -1, watermark)
                rect = watermark.shape
                index = np.unravel_index(np.argmax(chamfer_dist), image.shape[:-1])
                x, y = index[0] - t_w / 2, index[1] - t_h / 2
                x = int(x)
                y = int(y)
                top_left = (x, y)
                bottom_right = (top_left[0] + t_w, top_left[1] + t_h)
            else:
                res = cv2.matchTemplate(image_gray, watermark_gray, eval(meth))
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if eval(meth) in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
                bottom_right = (
                 top_left[0] + t_w, top_left[1] + t_h)
            tmp_img = image.copy()
            cv2.rectangle(tmp_img, top_left, bottom_right, 255, 2)
            ax_drawing = plt.subplot((len(methods) + 1) / 2, 2, i + 1)
            plt.sca(ax_drawing)
            plt.imshow(PlotImage(tmp_img))
            ax_drawing.set_title(meth)
            plt.draw()
            log.debug('%s %s %s' % (meth, top_left, bottom_right))
            results[meth] = list(top_left) + list(bottom_right)

        method_index = input('Please select a method result \n%s \n:' % '\n'.join(['%s. %s' % (i + 1, meth) for i, meth in enumerate(methods)]))
        try:
            return results[methods[int(method_index)]]
        except:
            return results['CANNY_2D_FILTER']

    def _possion_reconstruct(self, gradx, grady, kernel_size=KERNEL_SIZE, num_iters=100, h=0.1, boundary_image=None, boundary_zero=True):
        """Iterative algorithm for Poisson reconstruction.

        Given the gradx and grady values, find laplacian, and solve for image
            Also return the squared difference of every step.
            h = convergence rate
        """
        fxx = cv2.Sobel(gradx, (cv2.CV_64F), 1, 0, ksize=kernel_size)
        fyy = cv2.Sobel(grady, (cv2.CV_64F), 0, 1, ksize=kernel_size)
        laplacian = fxx + fyy
        m, n, p = laplacian.shape
        if boundary_zero:
            est = np.zeros(laplacian.shape)
        else:
            if not boundary_image is not None:
                raise AssertionError
            elif not boundary_image.shape == laplacian.shape:
                raise AssertionError
            est = boundary_image.copy()
        est[1:-1, 1:-1, :] = np.random.random((m - 2, n - 2, p))
        loss = []
        for i in range(num_iters):
            old_est = est.copy()
            est[1:-1, 1:-1, :] = 0.25 * (est[0:-2, 1:-1, :] + est[1:-1, 0:-2, :] + est[2:, 1:-1, :] + est[1:-1, 2:, :] - h * h * laplacian[1:-1, 1:-1, :])
            error = np.sum(np.square(est - old_est))
            loss.append(error)

        log.info('possion reconstruct finished.')
        return est

    def estimate_normalized_alpha(self, J, W_m, num_images=30, threshold=170, invert=False, adaptive=False, adaptive_threshold=21, c2=10):
        _Wm = (255 * utils.PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
        if adaptive:
            thr = cv2.adaptiveThreshold(_Wm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_threshold, c2)
        else:
            ret, thr = cv2.threshold(_Wm, threshold, 255, cv2.THRESH_BINARY)
        if invert:
            thr = 255 - thr
        thr = np.stack([thr, thr, thr], axis=2)
        num, m, n, p = J.shape
        alpha = np.zeros((num_images, m, n))
        iterpatch = 900
        log.info('Estimating normalized alpha using %d images.' % num_images)
        for idx in range(num_images):
            imgcopy = thr
            alph = closed_form_matting.closed_form_matte(J[idx], imgcopy)
            alpha[idx] = alph

        alpha = np.median(alpha, axis=0)
        return alpha

    def estimate_blend_factor(self, J, W_m, alph, threshold=2.5500000000000003):
        K, m, n, p = J.shape
        Jm = J - W_m
        gx_jm = np.zeros(J.shape)
        gy_jm = np.zeros(J.shape)
        for i in range(K):
            gx_jm[i] = cv2.Sobel(Jm[i], cv2.CV_64F, 1, 0, 3)
            gy_jm[i] = cv2.Sobel(Jm[i], cv2.CV_64F, 0, 1, 3)

        Jm_grad = np.sqrt(gx_jm ** 2 + gy_jm ** 2)
        est_Ik = alph * np.median(J, axis=0)
        gx_estIk = cv2.Sobel(est_Ik, cv2.CV_64F, 1, 0, 3)
        gy_estIk = cv2.Sobel(est_Ik, cv2.CV_64F, 0, 1, 3)
        estIk_grad = np.sqrt(gx_estIk ** 2 + gy_estIk ** 2)
        C = []
        for i in range(3):
            c_i = np.sum(Jm_grad[:, :, :, i] * estIk_grad[:, :, i]) / np.sum(np.square(estIk_grad[:, :, i])) / K
            log.debug(c_i)
            C.append(c_i)

        return (C, est_Ik)

    def sharpen(self, src_img):
        kernel = np.array([
         [
          0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        dst_img = cv2.filter2D(src_img, (-1), kernel=kernel)
        return dst_img

    def solve(self, iters=4, watermark_threshold=0.4, save_result=True):
        images_num = len(self._raw_images)
        log.debug('Sharpening normalized images...')
        self._raw_images = [self.sharpen(img) for path, img in self._normalized_images.items()]
        result_ = []
        if self.same_position:
            log.info('start estimate watermark(same position)')
            Wm_gradx, Wm_grady, images_gradx, images_grady = self._estimate_watermark()
            utils.save_train_obj(Wm_gradx, 'Wm_gradx_{shape}'.format(shape=('_'.join([str(i) for i in Wm_gradx.shape]))))
            utils.save_train_obj(Wm_grady, 'Wm_grady_{shape}'.format(shape=('_'.join([str(i) for i in Wm_grady.shape]))))
        else:
            log.error('start estimate watermark(different position), need user operation')
            raise Exception('Not support float waterark')
        Wm_gradx_cropped, Wm_grady_cropped = self._crop_watermark(Wm_gradx,
          Wm_grady, threshold=watermark_threshold)
        utils.save_train_obj(Wm_gradx_cropped, 'Wm_gradx_cropped_{shape}'.format(shape=('_'.join([str(i) for i in Wm_gradx_cropped.shape]))))
        utils.save_train_obj(Wm_grady_cropped, 'Wm_grady_cropped_{shape}'.format(shape=('_'.join([str(i) for i in Wm_grady_cropped.shape]))))
        watermark = self._possion_reconstruct(Wm_gradx_cropped, Wm_grady_cropped, KERNEL_SIZE, 100)
        utils.save_train_obj(watermark, 'watermark_{shape}'.format(shape=('_'.join([str(i) for i in watermark.shape]))))
        Wm = watermark - watermark.min()
        cropped_images_J = np.zeros((images_num,) + Wm_gradx_cropped.shape)
        log.debug('The watermar shape is %s' % str(Wm_gradx_cropped.shape))
        # 对于水印检测算法不稳定，请输入要删除的最终位置，例如：x，y（w，h是水印形状）或enter跳过。
        customer_position_raw_text = input('For the watermark detect algo is not stable, please input the final position you want to remove like: x,y (w, h is the watermark shape) or enter to skip it.')
        fixed_positions = []
        if len(customer_position_raw_text) > 0:
            fixed_positions = [int(c.strip()) for c in customer_position_raw_text.split(',')]
        for i, (path, image) in enumerate(self._normalized_images.items()):
            if len(fixed_positions) != 2:
                min_x, min_y, max_x, max_y = self._detect_watermark(image, Wm_gradx_cropped, Wm_grady_cropped)
            else:
                min_x, min_y, max_x, max_y = (
                 fixed_positions[0], fixed_positions[1],
                 fixed_positions[0] + Wm_gradx_cropped.shape[0], fixed_positions[1] + Wm_gradx_cropped.shape[1])
            result_.append({'position':(min_x, min_y, max_x, max_y),  'raw_image':image, 
             'path':path})
            show_single_image(Wm_gradx_cropped)
            try:
                cropped_images_J[i, :, :, :] = utils.get_cropped_image(image, min_x, min_y, max_x, max_y)
            except BaseException:
                traceback.print_exc()

            show_single_image(cropped_images_J[i])
            result_[i]['cropped_images_J'] = cropped_images_J[i]

        images = np.array([r['raw_image'] for r in result_])
        utils.save_train_obj(images, 'images_{shape}'.format(shape=('_'.join([str(i) for i in images.shape]))))
        utils.save_train_obj(cropped_images_J, 'cropped_images_J_{shape}'.format(shape=('_'.join([str(i) for i in cropped_images_J.shape]))))
        alph_est = self.estimate_normalized_alpha(cropped_images_J, Wm, images_num)
        utils.save_train_obj(alph_est, 'alph_est_{shape}'.format(shape=('_'.join([str(i) for i in alph_est.shape]))))
        alph = np.stack([alph_est, alph_est, alph_est], axis=2)
        utils.save_train_obj(alph, 'alph_{shape}'.format(shape=('_'.join([str(i) for i in alph.shape]))))
        C, est_Ik = self.estimate_blend_factor(cropped_images_J, Wm, alph)
        utils.save_train_obj(C, 'C_{shape}'.format(shape=('_'.join([str(i) for i in [len(C)] + list(C[0].shape)]))))
        utils.save_train_obj(est_Ik, 'est_Ik_{shape}'.format(shape=('_'.join([str(i) for i in est_Ik.shape]))))
        alpha = alph.copy()
        for i in range(3):
            alpha[:, :, i] = C[i] * alpha[:, :, i]

        Wm = Wm + alpha * est_Ik
        utils.save_train_obj(Wm, 'Wm_{shape}'.format(shape=('_'.join([str(i) for i in Wm.shape]))))
        W = Wm.copy()
        for i in range(3):
            W[:, :, i] /= C[i]

        Jt = cropped_images_J
        Wk, Ik, W, alpha1 = reconstructwatermark.solve_images(Jt,
          watermark, alpha, W, iters=iters, is_display=False)
        utils.save_train_obj(Wk, 'Wk_{shape}'.format(shape=('_'.join([str(i) for i in Wk.shape]))))
        utils.save_train_obj(Ik, 'Ik_{shape}'.format(shape=('_'.join([str(i) for i in Ik.shape]))))
        utils.save_train_obj(W, 'W_{shape}'.format(shape=('_'.join([str(i) for i in W.shape]))))
        utils.save_train_obj(alpha1, 'alpha1_{shape}'.format(shape=('_'.join([str(i) for i in alpha1.shape]))))
        for i in range(len(result_)):
            try:
                result_image = result_[i]['raw_image'].copy()
                min_x, min_y, max_x, max_y = result_[i]['position']
                result_image[min_x:max_x, min_y:max_y, :] = Ik[i]
                result_[i]['result_image'] = result_image
                if save_result:
                    utils.save_image(result_image, 'result_%s.jpg' % str(i))
            except Exception as exc:
                print(exc)

        return (
         Wk, Ik, W, alpha1, result_)


def solve(foldername, dest_foldername, iters, watermark_threshold, save_result):
    solver = Solver(foldername, dest_foldername)
    solver.solve(iters=iters,
      watermark_threshold=watermark_threshold,
      save_result=save_result)

