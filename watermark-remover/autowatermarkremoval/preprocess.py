# uncompyle6 version 3.7.2
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.9 (default, Apr 18 2020, 01:56:04) 
# [GCC 8.4.0]
# Embedded file name: ./autowatermarkremoval/preprocess.py
# Compiled at: 2019-11-22 09:39:26
# Size of source mod 2**32: 1090 bytes
import sys, os, cv2, numpy as np

def preprocess(foldername, size=500, suffix='_processed'):
    dest_folder = foldername + suffix
    processed = os.path.abspath(dest_folder)
    if os.path.exists(processed):
        print('Directory %s already exists.' % processed)
        return
    os.mkdir(dest_folder)
    for root, dirs, files in os.walk(foldername):
        for file in files:
            path = os.sep.join([os.path.abspath(root), file])
            img = cv2.imread(path)
            if img is not None:
                m, n, p = img.shape
                m_t, n_t = (size - m) / 2, (size - n) / 2
                final_img = np.pad(img,
                  ((m_t, size - m - m_t), (n_t, size - n - n_t), (0, 0)), mode='constant')
                cv2.imwrite(os.sep.join([dest_folder, file]), final_img)
                print('Saved to : %s' % file)
                print(final_img.shape)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Format : %s <foldername>' % sys.argv[0])
    else:
        preprocess(sys.argv[1])
# okay decompiling preprocess.pyc
