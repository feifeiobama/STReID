from sklearn.mixture import GaussianMixture
import numpy as np
import random


def normalize(x, mu, var, thres=1):
    offset = abs(x - mu) + thres
    return 1 / (np.sqrt(2 * np.pi * var)) * np.exp(- offset ** 2 / (2 * var))


class GMM(object):
    def __init__(self, num, weights, means, covariances):
        self.num = num
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.n = len(weights)

    def val(self, x):
        y = 0
        for i in range(self.n):
            y += self.weights[i] * normalize(x, self.means[i], self.covariances[i])
        return y * self.num
    
    def gate(self, x, alpha=0.05):
        for i in range(self.n):
            if self.weights[i] < alpha: continue
            if (x - self.means[i]) ** 2 < 9 * self.covariances[i]: return True
        return False
    
    def toGaussianMixture(self):
        g = GaussianMixture(self.n)
        g.fit(np.random.rand(2 * self.n).reshape((-1, 1)))
        g.weights_ = np.array(self.weights)
        g.means_ = np.array(self.means)[:, np.newaxis]
        g.covariances_ = np.array(self.covariances)[:, np.newaxis, np.newaxis]
        return g


def div_uni(g1, g2, denominator):
    # g1 / (g2 * denominator)
    if g2.num == 0: return GMM(0, [], [], [])
    w_s, mu_s, var_s = [], [], []
    for i in range(g1.n):
        w1, w2 = g1.weights[i], g2.weights[0]
        mu1, mu2 = g1.means[i], g2.means[0]
        var1, var2 = g1.covariances[i], g2.covariances[0]
        if var1 >= var2: continue
        var = var1 * var2 / (var2 - var1)
        w_s.append((w1 / w2) / (var1 / var2) ** 0.25 * np.exp((mu1 - mu2) ** 2 / (var2 - var1)) * 2 * np.pi * var ** 0.25)
        mu_s.append((mu1 * var2 - mu2 * var1) / (var2 - var1))
        var_s.append(var)
    w_sum = sum(w_s)
    num = w_sum * g1.num / (g2.num * denominator)
    w_s = [wi / w_sum for wi in w_s]
    return GMM(num, w_s, mu_s, var_s)


class Distribution(object):
    def __init__(self, cam_num, n=1):
        self.cam_num = cam_num

        self.deltas, self.fits = [], []
        for i in range(cam_num):
            delta_row, fit_row = [], []
            for j in range(i + 1):
                delta_row.append([])
            self.deltas.append(delta_row)
            self.fits.append(fit_row)

        self.n = n

    def update(self, c1, c2, t1, t2):
        c1r, c2r = (c1, c2) if c1 > c2 else (c2, c1)
        t1, t2 = (t1, t2) if c1 > c2 else (t2, t1)
        delta = t1 - t2 if c1 != c2 or random.random() < 0.5 else t2 - t1
        self.deltas[c1r][c2r].append(delta)

    def estimate(self):
        g = GaussianMixture(n_components=self.n)
        for i in range(self.cam_num):
            for j in range(i + 1):
                total = len(self.deltas[i][j])
                try:
                    g.fit(np.array(self.deltas[i][j]).reshape(-1, 1))
                    gmm = GMM(total, g.weights_, g.means_[:, 0], g.covariances_[:, 0, 0])
                except:
                    gmm = GMM(0, [], [], [])
                finally:
                    self.fits[i].append(gmm)
                    
    def in_delta(self, c1, c2, t1, t2):
        c1r, c2r = (c1, c2) if c1 > c2 else (c2, c1)
        t1, t2 = (t1, t2) if c1 > c2 else (t2, t1)
        return (t1 - t2) in self.deltas[c1r][c2r]
                    
    def in_peak(self, c1, c2, t1, t2, alpha=0.1):
        c1r, c2r = (c1, c2) if c1 > c2 else (c2, c1)
        t1, t2 = (t1, t2) if c1 > c2 else (t2, t1)
        return self.fits[c1r][c2r].gate(t1 - t2, alpha)
    
    def val(self, c1, c2, t1, t2):
        c1r, c2r = (c1, c2) if c1 > c2 else (c2, c1)
        t1, t2 = (t1, t2) if c1 > c2 else (t2, t1)
        return self.fits[c1r][c2r].val(t1 - t2)


def ln(x, thres=1e-6):
    return np.log(x + thres)

class ST_Model(object):
    def __init__(self, cam_num, n=5):
        self.cam_num = cam_num
        self.n = n
        self.probs = None
        self.same_rate = 0

    def fit(self, dataset, sample_num=10000000):
        # n(ST, same) / (n(ST) * p(same))
        same = Distribution(self.cam_num, self.n)
        total = Distribution(self.cam_num, 1)

        cluster_by_pid, cam_count = {}, [0] * self.cam_num
        for (_, pid, cam, timestamp) in dataset:
            if pid not in cluster_by_pid:
                cluster_by_pid[pid] = []
            cluster_by_pid[pid].append((cam, timestamp))
            cam_count[cam] += 1

        same_num = 0
        for cluster in cluster_by_pid.values():
            l = len(cluster)
            for i in range(l):
                for j in range(i):
                    same.update(cluster[i][0], cluster[j][0], cluster[i][1], cluster[j][1])
            same_num += l * (l - 1) / 2
        same.estimate()

        l = len(dataset)
        for _ in range(sample_num):
            i = random.randint(0, l - 1)
            j = random.randint(0, l - 1)
            if i == j: continue
            total.update(dataset[i][2], dataset[j][2], dataset[i][3], dataset[j][3])
        total.estimate()

        for i in range(self.cam_num):
            for j in range(i):
                total.fits[i][j].num = cam_count[i] * cam_count[j]
            total.fits[i][i].num = cam_count[i] * (cam_count[i] - 1) / 2
        total_img = sum(cam_count)
        total_num = total_img * (total_img - 1) / 2

        self.same_rate = same_num / total_num

        self.probs = []
        for i in range(self.cam_num):
            prob_row = []
            for j in range(i + 1):
                prob = div_uni(same.fits[i][j], total.fits[i][j], self.same_rate)
                prob_row.append(prob)
            self.probs.append(prob_row)
            
        return same, total

    def apply(self, distmat, query, gallery):
        # d = d + d_mean * ln(p) / ln(same_rate)
        d_mean, d_var = distmat.mean(), distmat.var()
#         print('d_mean = %f, d_var = %f, same_rate = %e' % (d_mean, d_var, self.same_rate))
        for i, (_, _, c1, t1) in enumerate(query):
            for j, (_, _, c2, t2) in enumerate(gallery):
                c1r, c2r = (c1, c2) if c1 > c2 else (c2, c1)
                t1, t2 = (t1, t2) if c1 > c2 else (t2, t1)
                p = self.probs[c1r][c2r].val(t1 - t2)
                distmat[i][j] += d_mean * ln(p) / ln(self.same_rate)
        return distmat

    def in_peak(self, c1, c2, t1, t2, alpha=0.1):
        c1r, c2r = (c1, c2) if c1 > c2 else (c2, c1)
        t1, t2 = (t1, t2) if c1 > c2 else (t2, t1)
        return self.probs[c1r][c2r].gate(t1 - t2, alpha)
    
    def val(self, c1, c2, t1, t2):
        c1r, c2r = (c1, c2) if c1 > c2 else (c2, c1)
        t1, t2 = (t1, t2) if c1 > c2 else (t2, t1)
        return self.probs[c1r][c2r].val(t1 - t2)
    
    def on_peak(self, c1, c2, t1, t2):
        return self.in_peak(c1, c2, t1, t2, 0.2)


class ST_Model_KNN(ST_Model):
    def __init__(self, cam_num, ranking, n=5):
        super(ST_Model_KNN, self).__init__(cam_num, n)
        self.ranking = ranking

    def fit(self, dataset, g, b, sample_num=10000000):
        # n(ST, same) / (n(ST) * p(same))
        same = Distribution(self.cam_num, self.n)
        total = Distribution(self.cam_num, 1)

        cam_count = [0] * self.cam_num
        for (_, _, cam, _) in dataset:
            cam_count[cam] += 1

        same_num = 0
        for i in range(len(dataset)):
            for j_ in range(b):
                j = self.ranking[i][j_]
                if i > j and g(i, j):
                    same.update(dataset[i][2], dataset[j][2], dataset[i][3], dataset[j][3])
                    same_num += 1
        same.estimate()

        l = len(dataset)
        for _ in range(sample_num):
            i = random.randint(0, l - 1)
            j = random.randint(0, l - 1)
            if i == j: continue
            total.update(dataset[i][2], dataset[j][2], dataset[i][3], dataset[j][3])
        total.estimate()

        for i in range(self.cam_num):
            for j in range(i):
                total.fits[i][j].num = cam_count[i] * cam_count[j]
            total.fits[i][i].num = cam_count[i] * (cam_count[i] - 1) / 2
        total_img = sum(cam_count)
        total_num = total_img * (total_img - 1) / 2

        self.same_rate = same_num / total_num

        self.probs = []
        for i in range(self.cam_num):
            prob_row = []
            for j in range(i + 1):
                prob = div_uni(same.fits[i][j], total.fits[i][j], self.same_rate)
                prob_row.append(prob)
            self.probs.append(prob_row)
            
        return same, total