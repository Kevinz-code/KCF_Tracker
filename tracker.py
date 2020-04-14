from __future__ import division
import numpy as np
import cv2
import time
from HOG import HOG

'''
Important:
FFT takes nums in even number 
'''

def fft(area):
    # forward
    return cv2.dft(np.float32(area), flags=cv2.DFT_COMPLEX_OUTPUT)


def ifft(area):
    # backward and remember the scale
    return cv2.dft(np.float32(area), flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))


def real(x):
    # real value of a complex
    return x[:, :, 0]


def im(x):
    # not real value
    return x[:, :, 1]


def complex_multi_2d(x1, x2):
    res = np.zeros(x1.shape, x1.dtype)

    res[:, :, 0] = x1[:, :, 0] * x2[:, :, 0] - x1[:, :, 1] * x2[:, :, 1]
    res[:, :, 1] = x1[:, :, 0] * x2[:, :, 1] + x1[:, :, 1] * x2[:, :, 0]
    return res


def complex_division(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1.0 / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2 + 0.0000001)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] - a[:, :, 0] * b[:, :, 1]) * divisor
    return res


def get_border_roi(x1, y1, x2, y2, frame, i=1):
    h, w = frame.shape
    d_x1, d_x2, d_y1, d_y2 = [0] * 4
    # in_roi
    if x1 < 0:
        # x2 -= x1
        d_x1 = -x1
        x1 = 0
    if y1 < 0:
        # y2 -= y1
        d_y1 = -y1
        y1 = 0
    if x2 > w:
        # x1 -= w
        d_x2 = x2 - w
        x2 = w
    if y2 > h:
        # y1 -= h
        d_y2 = y2 - h
        y2 = h
    in_roi = frame[y1:y2 + 1, x1:x2 + 1]
    final_roi = in_roi

    if [d_x1, d_x2, d_y1, d_y2] != [0, 0, 0, 0]:
        # print(x1, y1, x2, y2)
        # print(d_x1, d_x2, d_y1, d_y2)
        bordertype = cv2.BORDER_WRAP
        final_roi = cv2.copyMakeBorder(in_roi, d_y1, d_y2, d_x1, d_x2, bordertype)
        # cv2.imshow("border", final_roi)
        # cv2.waitKey(200)

    return final_roi


class Tracker(object):
    def __init__(self, sigma, lamda, hog=False, multiscale=False,
                 pad=2, adapt=0.075, scale=0.8, scale_thresh=1,
                 TargetGaussianBand=40.0, gamma=2.0):

        # important parameters
        self.sigma = sigma
        self.lamda = lamda
        self.s = TargetGaussianBand
        self.gamma = gamma
        self.region = [0, 0, 0, 0]
        self.region_size = [0, 0]
        self.fixed_size = [0, 0]
        self.frame = 0
        self._train_roi = 0.0

        # other essentials
        self.hann = 0.0
        self.hog = hog
        self.multiscale = multiscale

        self.complex_alpha = 0.0
        self.pad = pad
        self.scale = scale
        self.scale_thresh = scale_thresh
        self.adapt = adapt

    def set_first_frame(self, frame, region):
        if self.hog:
            self.fixed_size = [32, 225]
        else:
            self.fixed_size = [64, 64]

        # set cur_frame
        self.frame = frame
        self.region = region
        self.region_size = [self.region[3]-self.region[1] + 1, self.region[2]-self.region[0] + 1]

        self.create_hanning()
        self._train_roi, nothing= self.get_featuremap(frame, region[0], region[1], region[2], region[3], scale=1.0)

    def get_featuremap(self, frame, new_x1, new_y1, new_x2, new_y2, scale):
        """Important slice operation"""
        # pad_scale_roi = frame[max(new_y1, 0):new_y2+1, max(new_x1,0):new_x2 + 1]
        pad_scale_roi = get_border_roi(new_x1, new_y1, new_x2, new_y2, frame)

        if scale != 1.0:
            half_w = (new_x2 - new_x1 + 1) / 2 * (scale - 1)
            half_h = (new_y2 - new_y1 + 1) / 2 * (scale - 1)

            new_x1 = int(np.ceil(new_x1 - half_w))
            new_y1 = int(np.ceil(new_y1 - half_h))
            new_x2 = int(np.floor(new_x2 + half_w))
            new_y2 = int(np.floor(new_y2 + half_h))

            """Important slice operation"""
            # pad_scale_roi = frame[max(new_y1, 0):new_y2+1, max(new_x1, 0):new_x2+1]  # frame follow H x W sequenc
            pad_scale_roi = get_border_roi(new_x1, new_y1, new_x2, new_y2, frame)


        if self.pad != 0:
            half_w = (new_x2 - new_x1 + 1) / 2 * (self.pad - 1)
            half_h = (new_y2 - new_y1 + 1) / 2 * (self.pad - 1)

            tmp_x1 = int(np.ceil(new_x1 - half_w))
            tmp_y1 = int(np.ceil(new_y1 - half_h))
            tmp_x2 = int(np.floor(new_x2 + half_w))
            tmp_y2 = int(np.floor(new_y2 + half_h))

            """Important slice operation"""
            # pad_scale_roi = frame[max(tmp_y1, 0):tmp_y2+1, max(tmp_x1, 0):tmp_x2+1]  # frame fol
            pad_scale_roi = get_border_roi(tmp_x1, tmp_y1, tmp_x2, tmp_y2, frame, i=2)

        fix_roi = cv2.resize(pad_scale_roi, dsize=(self.fixed_size[1], self.fixed_size[1]))  # dsize follow W x H (64x64)
        fix_roi = np.asarray(fix_roi, dtype=np.float)  # np.array
        fix_roi = fix_roi / 255.0 - 0.5          # normalize
        fix_roi = np.power(fix_roi, self.gamma)  # gamma correct
        fix_roi = fix_roi * self.hann            # Hanning filter

        if self.hog:
            fix_roi = fix_roi + 0.5
            hog = HOG(window=fix_roi, cell_size=4, bin_size=8, gamma=1.0)
            self.adapt = 0.1
            hog.init_mag_angle()
            fix_roi = hog.get_window_grad()

        return fix_roi, [new_x1, new_y1, new_x2, new_y2]

    def create_hanning(self):
        N = self.fixed_size[0]
        hann2t, hann1t = np.ogrid[0:N, 0:N]

        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (N - 1)))
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (N - 1)))
        hann2d = hann2t * hann1t

        self.hann = hann2d
        self.hann = self.hann.astype(np.float32)

    def create_target(self, len_y, len_x):
        half_y = (len_y - 0) / 2
        half_x = (len_x - 0) / 2

        # calculate the bandwidth
        target_sigma = np.sqrt(len_y*len_x) / self.s  # float array
        bandwidth = (-2.0) * (target_sigma**2)

        # generate grid
        y_vector, x_vector = np.ogrid[0:len_y, 0:len_x]

        # do not forget the sequence
        y_vector = (y_vector - half_y)**2
        x_vector = (x_vector - half_x)**2
        target = np.exp((y_vector + x_vector) / bandwidth)

        return target

    def kernel_correlation(self, x1, x2):
        # the only place that use multi-channels
        c = cv2.mulSpectrums(fft(x2), fft(x1), flags=0, conjB=True)
        c = ifft(c)
        c = real(c)  # c 2-dim

        # normalize the kernel function
        d = (np.sum(x1*x1) + np.sum(x2*x2) - 2*c) / (self.fixed_size[0]*self.fixed_size[1])
        d = d * (d > 0)

        k = np.exp(-d/(self.sigma*self.sigma))
        return c

    def train(self, x, y):
        k = self.kernel_correlation(x, x)
        complex_alpha = complex_division(fft(y), (fft(k) + self.lamda))
        return complex_alpha

    def test(self, complex_alpha, x, z):
        k = self.kernel_correlation(x, z)
        response = ifft(complex_multi_2d(fft(k),complex_alpha,))
        response = real(response)

        return response

    def get_peak(self, response):
        pos = np.argmax(response) + 1
        pos_y = int(pos/self.fixed_size[1]) + 1  # row
        pos_x = pos % self.fixed_size[1]  # line
        peak = response[pos_y - 1, pos_x - 1]

        return pos_y, pos_x, peak

    def refresh(self, cur_frame):
        # Decode axis and set img_size
        x1 = self.region[0]
        y1 = self.region[1]
        x2 = self.region[2]
        y2 = self.region[3]

        target_y = self.create_target(len_y=self.fixed_size[0], len_x=self.fixed_size[1])
        # training get alpha
        train_roi_x, nothing = self.get_featuremap(self.frame, x1, y1, x2, y2, scale=1.0)
        self._train_roi = (1 - self.adapt) * self._train_roi + self.adapt * train_roi_x
        complex_alpha = self.train(x=self._train_roi, y=target_y)
        self.complex_alpha = (1 - self.adapt) * self.complex_alpha + self.adapt * complex_alpha

        # Detect
        # Beginning
        scale_weight_list = [1.0, 0.86, 0.9]
        scale_list = [1.0, 1.0/self.scale, self.scale]
        test_roi_z = [0, 0, 0]
        response = [0, 0, 0]
        pos_y = [0, 0, 0]
        pos_x = [0, 0, 0]
        peak = [0, 0, 0]
        new_axis = [0, 0, 0]

        test_roi_z[0], new_axis[0] = self.get_featuremap(cur_frame, x1, y1, x2, y2, scale=1.0)
        response[0] = self.test(self.complex_alpha, x=self._train_roi, z=test_roi_z[0])
        pos_y[0], pos_x[0], peak[0] = self.get_peak(response[0])
        idx = 0

        if self.multiscale:
            test_roi_z[1], new_axis[1] = self.get_featuremap(cur_frame, x1, y1, x2, y2, scale=1.0 / self.scale)
            response[1] = self.test(self.complex_alpha, x=self._train_roi, z=test_roi_z[1])
            pos_y[1], pos_x[1], peak[1] = self.get_peak(response[1])

            test_roi_z[2], new_axis[2] = self.get_featuremap(cur_frame, x1, y1, x2, y2, scale=self.scale)
            response[2] = self.test(self.complex_alpha, x=self._train_roi, z=test_roi_z[2])
            pos_y[2], pos_x[2], peak[2] = self.get_peak(response[2])

            idx = int(np.argmax(np.array(peak) * np.array(scale_weight_list)))

            # conditional choose new scale test results
            horizontal_shift = abs(pos_x[idx] - pos_x[0])
            vertical_shift = abs(pos_y[idx] - pos_y[0])
            if abs(horizontal_shift - vertical_shift) > self.scale_thresh:
                idx = 0
            print(idx)

        scale_list = np.array([1.0, 0.90, 0.90]) * np.array(scale_list)  ## smooth update scale

        final_pos_y = pos_y[idx]
        final_pos_x = pos_x[idx]
        final_axis = new_axis[idx]

        center_y = int(self.fixed_size[0] / 2) + 0.5  # center shift
        center_x = int(self.fixed_size[1] / 2) + 0.5  # center shift
        # center_y = self.fixed_size[0] / 2  # center shift
        # center_x = self.fixed_size[1] / 2  # center shift

        delta_y = (final_pos_y - 0.5 - center_y) / self.fixed_size[0] * self.region_size[0] * scale_list[idx] * self.pad
        delta_x = (final_pos_x - 0.5 - center_x) / self.fixed_size[1] * self.region_size[1] * scale_list[idx] * self.pad

        # update if delta > 0.5 due to int() property
        # restrained within the self.frame
        final_axis[1] += delta_y
        final_axis[3] += delta_y
        final_axis[0] += delta_x
        final_axis[2] += delta_x
        self.region = list(map(int, [np.floor(item) for item in final_axis]))

        self.frame = cur_frame

        return self.region




