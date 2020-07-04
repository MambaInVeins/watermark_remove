# uncompyle6 version 3.7.2
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.9 (default, Apr 18 2020, 01:56:04) 
# [GCC 8.4.0]
# Embedded file name: ./autowatermarkremoval/utils.py
# Compiled at: 2019-11-22 09:39:26
# Size of source mod 2**32: 6920 bytes
import os, cv2, logging, pathlib, traceback, numpy as np
if os.environ.get('WM_UI_DISPLAY') == 'TRUE':
    from PIL import Image
    from matplotlib import pyplot as plt
from typing import List, Union, Tuple, Dict
log = logging.getLogger(__name__)
__all__ = [
 'load_images',
 'save_images',
 'PlotImage',
 'show_images',
 'get_cropped_image',
 'get_cropped_images']

def load_images(foldername: pathlib.Path) -> Dict[(pathlib.Path, np.ndarray)]:
    """ return all images under give foldername. (not recursive)
    return dict key is image absolute path, value is image (np.ndarray)
    """
    if not os.path.exists(foldername):
        log.warn('Folder {} does not exist.'.format(foldername))
        return {}
    else:
        images = {}
        for r, dirs, files in os.walk(foldername):
            for file in files:
                if file.startswith('.'):
                    pass
                else:
                    log.debug('imread from file {}'.format(os.path.abspath(os.path.join(r, file))))
                    absolute_path = os.path.abspath(os.path.join(r, file))
                    img = cv2.imread(absolute_path, 1)
                    if img is not None:
                        images[absolute_path] = img
                    else:
                        log.error('%s not found.' % file)

        return images


def load_image(image_path: pathlib.Path) -> np.ndarray:
    if not os.path.exists(image_path.absolute()):
        log.warning('Folder {} does not exist.'.format(image_path.absolute()))
        return
    else:
        return cv2.imread(image_path.absolute(), 1)


def save_images(images: Dict[(pathlib.Path, np.ndarray)], folderpath: pathlib.Path):
    """ save images to a give folderpath
    """
    for path, image in images:
        image_filename = os.path.basename(path)
        target_path = os.path.join(folderpath, 'watermark_removed__' + image_filename)
        cv2.imwrite(target_path, image)
        log.info(f"write image to {target_path} success")


def image_threshold(image, threshold=0.5):
    """
    Threshold the image to make all its elements greater than threshold*MAX = 1
    """
    m, M = np.min(image), np.max(image)
    im = PlotImage(image)
    im[im >= threshold] = 1
    im[im < 1] = 0
    return im


def PlotImage(image):
    """ PlotImage: Give a normalized image matrix which can be used with implot, etc.Maps to [0, 1]
    """
    im = image.astype(float)
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def show_images(images: List[np.ndarray], plot_size: Tuple[int]=(0, 0), convert_func=lambda x: x):
    """ use matplot to show given images at once
    """
    images_num = len(images)
    if plot_size == (0, 0):
        plot_size = (
         int(images_num / 2) + 1,
         (images_num + 1) / (int(images_num / 2) + 1))
    row, column = plot_size
    for index, image in enumerate(images):
        plt.subplot(row, column, index + 1)
        plt.imshow(convert_func(image))

    plt.show()
    cv2.waitKey(0)
    plt.close('all')


def get_cropped_images(images: Dict[(pathlib.Path, np.ndarray)], min_x, min_y, max_x, max_y) -> Dict[(str, np.ndarray)]:
    """ crop an array of images.
    """
    cropped_images = {path + '_cropped':get_cropped_image(image, min_x, min_y, max_x, max_y) for path, image in images.items()}
    return cropped_images


def get_cropped_image(image: np.ndarray, min_x, min_y, max_x, max_y) -> np.ndarray:
    """ Crop an image
    """
    log.debug(f"get_cropped_image min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
    image_cropped = image[min_x:max_x, min_y:max_y]
    return image_cropped


def normalize_image(image, shape):
    """
    """
    out_image = cv2.resize(image, shape)
    return out_image


def bgr2rgb(bgr):
    return bgr[..., ::-1]


def rgb2bgr(rgb):
    return rgb[...::-1]


def rbg2gbr(rgb):
    gbr = rgb[(..., [2, 0, 1])]


def chamfer_matching(src_image, pattern):
    pass


def jpg_2_pgm(img_src):
    im = Image.open(img_src)
    im = im.convert('RGB')
    im.save('%s.pgm' % img_src)
    return '%s.pgm' % img_src


def show_image(image):
    cv2.namedWindow('Result')
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(img, name):
    if not os.path.exists('result'):
        os.mkdir('result')
    try:
        log.debug('saving image {name}'.format(name=(os.path.join('result', name))))
        cv2.imwrite(os.path.join('result', name), img)
    except BaseException:
        log.error(traceback.format_exc())


def detect_watermark_position(image, watermark):
    """finding the watermark position in original image
    """
    KERNEL_SIZE = 3
    thresh_low = 200
    thresh_high = 220
    w_g_x = cv2.Sobel(watermark, (cv2.CV_64F), 1, 0, ksize=KERNEL_SIZE)
    w_g_y = cv2.Sobel(watermark, (cv2.CV_64F), 0, 1, ksize=KERNEL_SIZE)
    w_mag = np.average((np.sqrt(np.square(w_g_x) + np.square(w_g_y))), axis=2)
    image_edgemap = cv2.Canny(image, thresh_low, thresh_high)
    chamfer_dist = cv2.filter2D(image_edgemap.astype(float), -1, w_mag)
    rect = w_mag.shape
    index = np.unravel_index(np.argmax(chamfer_dist), image.shape[:-1])
    x, y = index[0] - rect[0] / 2, index[1] - rect[1] / 2
    x = int(x)
    y = int(y)
    return (
     x, y, x + rect[0], y + rect[1])


def remove_watermark(src_image, position, watermark, alpha):
    roi_image = src_image[position[0]:position[2], position[1]:position[3]]
    print('before: %s' % str(roi_image.shape))
    roi_image_result = (roi_image - watermark * alpha) / (1 - alpha)
    print('after: %s' % str(roi_image_result.shape))
    print(np.unique(alpha, return_counts=True))
    src_image[position[0]:position[2], position[1]:position[3]] = roi_image_result


def save_numpy_obj(np_obj, obj_path):
    np.save(obj_path, np_obj)


def load_numpy_obj(obj_path):
    return np.load(obj_path)


def save_train_obj(np_obj, obj_path):
    if not os.path.exists('result'):
        os.mkdir('result')
    try:
        log.debug('saving obj {name}'.format(name=(os.path.join('result', obj_path))))
        save_numpy_obj(np_obj, os.path.join('result', obj_path))
    except Exception:
        log.error(traceback.format_exc())


def load_train_obj(obj_path):
    if not os.path.exists('result'):
        os.mkdir('result')
    try:
        log.debug('saving obj {name}'.format(name=(os.path.join('result', obj_path))))
        return load_numpy_obj(os.path.join('result', obj_path))
    except Exception:
        log.error(traceback.format_exc())
# okay decompiling utils.pyc
