# uncompyle6 version 3.7.2
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.9 (default, Apr 18 2020, 01:56:04) 
# [GCC 8.4.0]
# Embedded file name: ./autowatermarkremoval/reconstructwatermark.py
# Compiled at: 2019-11-22 09:39:26
# Size of source mod 2**32: 11845 bytes
import os, scipy, numpy as np, cv2, logging, time, matplotlib.pyplot as plt
from scipy.sparse import *
from scipy.sparse import linalg
from .estimatewatermark import *
from .closed_form_matting import *
from .utils import load_images, PlotImage, show_image
log = logging.getLogger(__name__)

def get_cropped_images(foldername, num_images, start, end, shape):
    """
    This is the part where we get all the images, extract their parts, and then add it to our matrix
    """
    images_cropped = np.zeros((num_images,) + shape)
    imgs = load_images(foldername)
    for index, img in enumerate(imgs):
        images_cropped[index, :, :, :] = img[
         start[0]:start[0] + end[0], start[1]:start[1] + end[1], :]

    return (
     images_cropped, imgs)


def _get_ysobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    return [
     (
      i - 1, j, k, -2), (i - 1, j - 1, k, -1), (i - 1, j + 1, k, -1),
     (
      i + 1, j, k, 2), (i + 1, j - 1, k, 1), (i + 1, j + 1, k, 1)]


def _get_xsobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    return [
     (
      i, j - 1, k, -2), (i - 1, j - 1, k, -1), (i - 1, j + 1, k, -1),
     (
      i, j + 1, k, 2), (i + 1, j - 1, k, 1), (i + 1, j + 1, k, 1)]


def _filter_list_item(coord, shape):
    i, j, k, v = coord
    m, n, p = shape
    if i >= 0:
        if i < m:
            if j >= 0:
                if j < n:
                    return True


def _change_to_ravel_index(li, shape):
    li = filter(lambda x: _filter_list_item(x, shape), li)
    i, j, k, v = zip(*li)
    return zip(np.ravel_multi_index((i, j, k), shape), v)


def get_ySobel_matrix(m, n, p):
    size = m * n * p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda x: _get_ysobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape), ijk_nbrs)
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    return coo_matrix((vals, (i, j)), shape=(size, size))


def get_xSobel_matrix(m, n, p):
    size = m * n * p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda x: _get_xsobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape), ijk_nbrs)
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    return coo_matrix((vals, (i, j)), shape=(size, size))


def estimate_normalized_alpha(J, W_m, num_images=30, threshold=170, invert=False, adaptive=False, adaptive_threshold=21, c2=10):
    _Wm = (255 * PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
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
    log.debug('Estimating normalized alpha using %d images.' % num_images)
    for idx in range(num_images):
        imgcopy = thr
        alph = closed_form_matte(J[idx], imgcopy)
        alpha[idx] = alph

    alpha = np.median(alpha, axis=0)
    return alpha


def estimate_blend_factor(J, W_m, alph, threshold=2.5500000000000003):
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


def Func_Phi(X, epsilon=0.001):
    return np.sqrt(X + epsilon ** 2)


def Func_Phi_deriv(X, epsilon=0.001):
    return 0.5 / Func_Phi(X, epsilon)


def show_single_image(img):
    plt.subplot(1, 1, 1)
    plt.imshow(PlotImage(img))
    plt.draw()
    plt.pause(10)


def solve_watermark_and_original_image(i, Wki, Iki, Jki, alpha, Wm_gx, Wm_gy, W, alpha_gx, alpha_gy, sobelx, sobely, alpha_diag, alpha_bar_diag, lambda_i, cx, cy, gamma, beta, W_m, lambda_w, lambda_a):
    """
        Wk[i] ---> Wki
        Ik[i] ---> Iki
        J[i] ---> Jki
    """
    log.debug('Sovling new W and new watermark-free image of {img_index}'.format(img_index=(str(i))))
    Wkx = cv2.Sobel(Wki, cv2.CV_64F, 1, 0, 3)
    Wky = cv2.Sobel(Wki, cv2.CV_64F, 0, 1, 3)
    Ikx = cv2.Sobel(Iki, cv2.CV_64F, 1, 0, 3)
    Iky = cv2.Sobel(Iki, cv2.CV_64F, 0, 1, 3)
    alphaWk = alpha * Wki
    alphaWk_gx = cv2.Sobel(alphaWk, cv2.CV_64F, 1, 0, 3)
    alphaWk_gy = cv2.Sobel(alphaWk, cv2.CV_64F, 0, 1, 3)
    log.debug('start...')
    phi_data = diags(Func_Phi_deriv(np.square(alpha * Wki + (1 - alpha) * Iki - Jki).reshape(-1)))
    phi_f = diags(Func_Phi_deriv(((Wm_gx - alphaWk_gx) ** 2 + (Wm_gy - alphaWk_gy) ** 2).reshape(-1)))
    phi_aux = diags(Func_Phi_deriv(np.square(Wki - W).reshape(-1)))
    phi_rI = diags(Func_Phi_deriv(np.abs(alpha_gx) * Ikx ** 2 + np.abs(alpha_gy) * Iky ** 2).reshape(-1))
    phi_rW = diags(Func_Phi_deriv(np.abs(alpha_gx) * Wkx ** 2 + np.abs(alpha_gy) * Wky ** 2).reshape(-1))
    L_i = sobelx.T.dot(cx * phi_rI).dot(sobelx) + sobely.T.dot(cy * phi_rI).dot(sobely)
    L_w = sobelx.T.dot(cx * phi_rW).dot(sobelx) + sobely.T.dot(cy * phi_rW).dot(sobely)
    L_f = sobelx.T.dot(phi_f).dot(sobelx) + sobely.T.dot(phi_f).dot(sobely)
    A_f = alpha_diag.T.dot(L_f).dot(alpha_diag) + gamma * phi_aux
    bW = alpha_diag.dot(phi_data).dot(Jki.reshape(-1)) + beta * L_f.dot(W_m.reshape(-1)) + gamma * phi_aux.dot(W.reshape(-1))
    bI = alpha_bar_diag.dot(phi_data).dot(Jki.reshape(-1))
    A = vstack([hstack([alpha_diag ** 2 * phi_data + lambda_w * L_w + beta * A_f, alpha_diag * alpha_bar_diag * phi_data]),
     hstack([alpha_diag * alpha_bar_diag * phi_data, alpha_bar_diag ** 2 * phi_data + lambda_i * L_i])]).tocsr()
    b = np.hstack([bW, bI])
    x = linalg.spsolve(A, b)
    return x


def solve_images(J, W_m, alpha, W_init, gamma=1, beta=1, lambda_w=0.005, lambda_i=1, lambda_a=0.01, iters=4, is_display=False):
    """
    Master solver, follows the algorithm given in the supplementary.
    W_init: Initial value of W
    Step 1: Image Watermark decomposition
    """
    K, m, n, p = J.shape
    size = m * n * p
    sobelx = get_xSobel_matrix(m, n, p)
    sobely = get_ySobel_matrix(m, n, p)
    Ik = np.zeros(J.shape)
    Wk = np.zeros(J.shape)
    for i in range(K):
        Ik[i] = J[i] - W_m
        Wk[i] = W_init.copy()

    W = W_init.copy()
    for _ in range(iters):
        log.info('------------------------------------')
        log.debug('Iteration: %d' % _)
        log.debug('Step 1: calculating the sobel of each direction gradient')
        alpha_gx = cv2.Sobel(alpha, cv2.CV_64F, 1, 0, 3)
        alpha_gy = cv2.Sobel(alpha, cv2.CV_64F, 0, 1, 3)
        Wm_gx = cv2.Sobel(W_m, cv2.CV_64F, 1, 0, 3)
        Wm_gy = cv2.Sobel(W_m, cv2.CV_64F, 0, 1, 3)
        cx = diags(np.abs(alpha_gx).reshape(-1))
        cy = diags(np.abs(alpha_gy).reshape(-1))
        alpha_diag = diags(alpha.reshape(-1))
        alpha_bar_diag = diags((1 - alpha).reshape(-1))
        for i in range(K):
            x = solve_watermark_and_original_image(i, Wk[i], Ik[i], J[i], alpha, Wm_gx, Wm_gy, W, alpha_gx, alpha_gy, sobelx, sobely, alpha_diag, alpha_bar_diag, lambda_i, cx, cy, gamma, beta, W_m, lambda_w, lambda_a)
            Wk[i] = x[:size].reshape(m, n, p)
            Ik[i] = x[size:].reshape(m, n, p)
            log.debug('Done...')

        log.debug('Step 2: get a general watermark.')
        W = np.median(Wk, axis=0)
        log.debug('Step 3: calculating the alpha of the watermark')
        W_diag = diags(W.reshape(-1))
        start_t = time.time()
        for i in range(K):
            alphaWk = alpha * Wk[i]
            alphaWk_gx = cv2.Sobel(alphaWk, cv2.CV_64F, 1, 0, 3)
            alphaWk_gy = cv2.Sobel(alphaWk, cv2.CV_64F, 0, 1, 3)
            phi_f = diags(Func_Phi_deriv(((Wm_gx - alphaWk_gx) ** 2 + (Wm_gy - alphaWk_gy) ** 2).reshape(-1)))
            phi_kA = diags((Func_Phi_deriv((alpha * Wk[i] + (1 - alpha) * Ik[i] - J[i]) ** 2) * (W - Ik[i]) ** 2).reshape(-1))
            phi_kB = (Func_Phi_deriv((alpha * Wk[i] + (1 - alpha) * Ik[i] - J[i]) ** 2) * (W - Ik[i]) * (J[i] - Ik[i])).reshape(-1)
            phi_alpha = diags(Func_Phi_deriv(alpha_gx ** 2 + alpha_gy ** 2).reshape(-1))
            L_alpha = sobelx.T.dot(phi_alpha.dot(sobelx)) + sobely.T.dot(phi_alpha.dot(sobely))
            L_f = sobelx.T.dot(phi_f).dot(sobelx) + sobely.T.dot(phi_f).dot(sobely)
            A_tilde_f = W_diag.T.dot(L_f).dot(W_diag)
            if i == 0:
                A1 = phi_kA + lambda_a * L_alpha + beta * A_tilde_f
                b1 = phi_kB + beta * W_diag.dot(L_f).dot(W_m.reshape(-1))
            else:
                A1 += phi_kA + lambda_a * L_alpha + beta * A_tilde_f
                b1 += phi_kB + beta * W_diag.T.dot(L_f).dot(W_m.reshape(-1))

        alpha = linalg.spsolve(A1, b1).reshape(m, n, p)
        end_t = time.time()
        log.debug('end using time {u_time}...'.format(u_time=(end_t - start_t)))

    return (Wk, Ik, W, alpha)


def changeContrastImage(J, I):
    cJ1 = J[0, 0, :]
    cJ2 = J[-1, -1, :]
    cI1 = I[0, 0, :]
    cI2 = I[-1, -1, :]
    I_m = cJ1 + (I - cI1) / (cI2 - cI1) * (cJ2 - cJ1)
    return I_m
# okay decompiling reconstructwatermark.pyc
