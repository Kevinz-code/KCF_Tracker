from __future__ import division
import cv2
import numpy as np
import time


class HOG(object):

    def __init__(self, window, cell_size, bin_size, gamma):
        super(HOG, self).__init__()
        self.window = window
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.unit_angle = 180. / self.bin_size

        self.gamma = gamma

    def gamma_correct(self):
        # to eliminate  sunshine or darkness
        # very important
        self.window = self.window / 255 - 0.5
        self.window = np.power(self.window, self.gamma)

    def init_mag_angle(self):
        h, w = self.window.shape
        grad_y = cv2.Sobel(self.window, cv2.CV_64F, dx=0, dy=1, ksize=5)
        grad_x = cv2.Sobel(self.window, cv2.CV_64F, dx=1, dy=0, ksize=5)

        self.grad_magnitude = abs(cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, gamma=0.0))
        #self.grad_magnitude = cv2.magnitude(grad_x, grad_y)
        self.grad_angle = cv2.phase(grad_x, grad_y, angleInDegrees=True) % 180
        self.cell_grad_vector = np.zeros((int(h/self.cell_size), int(w/self.cell_size), int(self.bin_size)))

    def calculate_cellgrad(self, cell_mag, cell_angle):
        axis = [0] * self.bin_size
        hc, wc = cell_angle.shape
        for i in range(hc):
            for j in range(wc):
                tmp_mag = cell_mag[i, j]
                tmp_angle = cell_angle[i, j]
                # get index
                min_index = (tmp_angle / self.unit_angle) % 8
                max_index = (min_index + 1) % 8
                mod = tmp_angle % self.unit_angle

                axis[int(min_index)] += (tmp_mag * (1 - mod / self.unit_angle))
                axis[int(max_index)] += (tmp_mag * (mod / self.unit_angle))

        return axis

    def get_cell_grad(self):
        new_h, new_w, bins = self.cell_grad_vector.shape  # 16 16 8
        for i in range(new_h):
            for j in range(new_w):
                # slice from the global magnitude and angle
                cell_mag = self.grad_magnitude[i*self.cell_size:(i+1)*self.cell_size,
                               j*self.cell_size:(j+1)*self.cell_size]
                cell_angle = self.grad_angle[i*self.cell_size:(i+1)*self.cell_size,
                                 j*self.cell_size:(j+1)*self.cell_size]

                # calculate the grad_vector per cell
                self.cell_grad_vector[i,j] = self.calculate_cellgrad(cell_mag, cell_angle)

    def get_window_grad(self):
        self.get_cell_grad()
        new_h, new_w, bins = self.cell_grad_vector.shape
        window_grad_vector = []  # window grad
        for i in range(new_h - 1):
            for j in range(new_w - 1):
                block_grad_vector = []
                block_grad_vector.extend(self.cell_grad_vector[i, j])
                block_grad_vector.extend(self.cell_grad_vector[i + 1, j + 1])
                block_grad_vector.extend(self.cell_grad_vector[i , j + 1])
                block_grad_vector.extend(self.cell_grad_vector[i + 1, j ])

                # Normalize block vector
                func = lambda l: np.sqrt(sum(item ** 2 for item in l))
                mag = func(block_grad_vector)
                if mag != 1:
                    block_grad_vector = [item / mag for item in block_grad_vector]

                # extend window vector
                window_grad_vector.append(block_grad_vector)

        return np.array(window_grad_vector).reshape((self.bin_size*4, (new_h-1) * (new_w-1)))








