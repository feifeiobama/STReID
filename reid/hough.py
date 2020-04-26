import skimage.transform as st
import numpy as np

class Hough(object):
    def __init__(self, cam_num, grid_len, diag_len, start_time, freq=25, short_cut=False):
        self.cam_num = cam_num
        self.grid_len = grid_len
        self.diag_len = diag_len
        self.start_time = start_time
        self.freq = freq
        self.short_cut = short_cut

        self.accumulators = []
        for i in range(cam_num):
            accumulator_row = []
            for j in range(i+1):
                accumulator_row.append(np.zeros((diag_len, diag_len)))
            self.accumulators.append(accumulator_row)

        self.deltas = []

    def update(self, c1, c2, t1, t2, val=1):
        c1r, c2r = (c1, c2) if c1 > c2 else (c2, c1)
        t1r, t2r = ((t1 - self.start_time) / self.freq, (t2 - self.start_time) / self.freq)
        t1r, t2r = (t1r, t2r) if c1 > c2 else (t2r, t1r)
        accumulator = self.accumulators[c1r][c2r]
        accumulator[int(t1r / self.grid_len)][int(t2r / self.grid_len)] += val

    def _calc_delta(self):
        for i in range(self.cam_num):
            delta_row = []
            for j in range(i+1):
                h, theta, d = st.hough_line(self.accumulators[i][j])
                peaks = list(zip(*st.hough_line_peaks(h, theta, d, num_peaks=10)))
                filted_peaks = list(filter(lambda peak: np.abs(peak[1] + np.pi / 4) < 0.05, peaks))
                delta_row.append(list(map(lambda peak: peak[2] * self.grid_len, filted_peaks)))
            self.deltas.append(delta_row)
        return self.deltas

    def get_delta(self, c1, c2):
        if len(self.deltas) == 0:
            self._calc_delta()
        reverse_flag = not (c1 > c2)
        c1r, c2r = (c2, c1) if reverse_flag else (c1, c2)
        deltas = self.deltas[c1r][c2r]
        return list(map(lambda delta: -delta if reverse_flag else delta, deltas))

    def on_peak(self, c1, c2, t1, t2, width=60):
        if self.short_cut: return True
        delta_list = self.get_delta(c1, c2)
        test_list = [np.abs((t1 - t2) / self.freq + delta * np.sqrt(2)) < width for delta in delta_list]
        return any(test_list)
