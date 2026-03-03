from .base import UnsupervisedDriftDetector

import numpy as np
import math
from queue import deque
from pyhht.emd import EMD
from scipy.signal import argrelmax, argrelmin


class Entropy:
    def __init__(self):
        pass

    def cal_entropy(self):
        pass


class ApEn(Entropy):
    def __init__(self, u, m, r):
        self.u = u  # 时间序列
        self.r = r  # 相似度比较的阅知
        self.m = m  # 子序列长度
        self.N = len(u)  # 时间序列长度

    def _max_dist(self, xi, xj):
        return np.max([np.abs(a - b) for a, b in zip(xi, xj)])

    def _phi(self, m):
        X = []
        for i in range(self.N - m + 1):
            Xi = []
            for j in range(i, i + m):
                Xi.append(self.u[j])
            X.append(Xi)
        C = []
        for Xi in X:
            num = 0
            for Xj in X:
                if self._max_dist(Xi, Xj) <= self.r:
                    num += 1
            C.append(num / (self.N - m + 1))
        phi = np.sum(np.log(C)) / (self.N - m + 1)
        return phi

    def cal_entropy(self):
        return np.abs(self._phi(self.m + 1) - self._phi(self.m))


class SampEn(Entropy):
    def __init__(self, u, m, r):
        self.u = u
        self.r = r
        self.m = m
        self.N = len(u)

    def _max_dist(self, xi, xj):
        return np.max([np.abs(a - b) for a, b in zip(xi, xj)])

    def _phi(self, m):
        X = []
        for i in range(self.N - m + 1):
            Xi = []
            for j in range(i, i + m):
                Xi.append(self.u[j])
            X.append(Xi)
        B = []
        for Xi in X:
            num = 0
            for Xj in X:
                if self._max_dist(Xi, Xj) <= self.r:
                    num += 1
            B.append((num - 1) / (self.N - m))
        phi = np.sum(B) / (self.N - m + 1)
        return phi

    def cal_entropy(self):
        return -np.log(self._phi(self.m + 1) / self._phi(self.m))


class FsEn(Entropy):
    def __init__(self, u, m, r):
        self.u = u
        self.r = r
        self.m = m
        self.N = len(u)

    def _max_dist(self, xi, xj):
        return np.max([np.abs(a) - np.abs(b) for a, b in zip(xi, xj)])

    def _phi(self, m):
        X = []  # 重构的子序列
        for i in range(self.N - m + 1):
            Xi = []
            for j in range(i, i + m):
                Xi.append(self.u[j])
            Xi = Xi - np.mean(Xi)
            X.append(Xi)
        dij = []  # 满足相似度阈值条件的个数与总的统计数目之间的比值
        for Xi in X:
            di = []
            for Xj in X:
                di.append(self._max_dist(Xi, Xj))
            dij.append(di)
        A = []
        for i in range(self.N - m + 1):
            Ai = []
            for j in range(self.N - m + 1):
                if i == j:
                    continue
                Ai.append(np.exp(-np.log(2) * np.square(dij[i][j] / self.r)))
            A.append(Ai)
        C = []
        for Ai in A:
            C.append(np.sum(Ai) / (self.N - m))
        phi = np.sum(C) / (self.N - m + 1)
        return phi

    def cal_entropy(self):
        return np.log(self._phi(self.m)) - np.log(self._phi(self.m + 1))


class PeEn(Entropy):
    def __init__(self, u, m, l):
        self.u = u
        self.m = m
        self.l = l
        self.N = len(u)

    def cal_entropy(self, std=True):
        '''

        Parameters
        ----------
        std 是否标准化输出

        Returns
        -------

        '''
        X = []
        for i in range(self.N - self.m + 1):
            Xi = []
            for j in range(i, i + (self.m - 1 + 1) * self.l, self.l):
                Xi.append(self.u[j])
            X.append(Xi)

        J = []
        for Xi in X:
            dict = {}
            for i in range(len(Xi)):
                dict[i + 1] = Xi[i]
            sorted_dict = sorted(dict.items(), key=lambda x: x[1])
            j = []
            for x, y in sorted_dict:
                j.append(x)
            J.append(j)

        J_dict = {}
        for j in J:
            temp = map(str, j)
            temp = ''.join(temp)
            if temp not in J_dict.keys():
                J_dict[temp] = 1
            else:
                J_dict[temp] += 1

        H = []
        for key in J_dict.keys():
            count = J_dict[key]
            p = count / len(J)
            H.append(-p * np.log(p))

        Hp = np.sum(H)
        if std:
            return Hp / np.log(math.factorial(self.m))
        else:
            return Hp


class WeightedPeEn(Entropy):
    def __init__(self, u, m, l):
        self.u = u
        self.m = m
        self.l = l
        self.N = len(u)

    def cal_entropy(self, std=True):
        '''

        Parameters
        ----------
        std 是否标准化输出

        Returns
        -------

        '''
        X = []  # 重构的子序列
        for i in range(self.N - self.m + 1):
            Xi = []
            for j in range(i, i + (self.m - 1 + 1) * self.l, self.l):
                Xi.append(self.u[j])
            X.append(Xi)

        J = []
        for Xi in X:
            dict = {}
            var = np.var(Xi)
            for i in range(len(Xi)):
                dict[i + 1] = Xi[i]
            sorted_dict = sorted(dict.items(), key=lambda x: x[1])
            j = []
            dict_var = {}
            for x, y in sorted_dict:
                j.append(x)
            key = map(str, j)
            key = ''.join(key)
            dict_var[key] = var
            J.append(dict_var)

        J_dict = {}
        for j in J:
            for key in j:
                if key not in J_dict:
                    value = j[key]
                    J_dict[key] = [value]
                else:
                    value = j[key]
                    J_dict[key].append(value)

        weighted_sum = 0
        for key in J_dict.keys():
            weighted_sum += np.sum(J_dict[key])

        H = []
        for key in J_dict.keys():
            w_sum = np.sum(J_dict[key])
            p = w_sum / weighted_sum
            H.append(-p * np.log(p))

        Hp = np.sum(H)
        if std:
            return Hp / np.log(math.factorial(self.m))
        else:
            return Hp


class IncrEn(Entropy):
    def __init__(self, u, m, r):
        self.u = u
        self.m = m
        self.r = r
        self.N = len(u)

    def sgn(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def cal_q(self, x, r, std):
        n_std = int(x / std)
        if n_std >= r:
            return r
        else:
            return n_std

    def cal_entropy(self):
        X = []
        diff_u = np.diff(self.u)
        for i in range(self.N - self.m):
            Xi = []
            for j in range(i, i + self.m):
                Xi.append(diff_u[j])
            X.append(Xi)

        # 重构向量对应的模型向量 example:[[(s1,q1),(s2,q2)], [(s1,q1),(s2,q2)]] 其中[(s1,q1),(s2,q2)]代表一阶差分序列中每个子序列对应的模式向量
        X_w = []
        std = np.std(np.abs(diff_u))  # 窗口内一阶差的标准差

        for i in range(self.N - self.m):
            u = X[i]
            u_w = []
            for j in range(self.m):
                s = self.sgn(u[j])
                q = self.cal_q(np.abs(u[j]), self.r, std)
                u_w.append((s, q))
            X_w.append(u_w)

        key_list = []  # 所有的w组合 [1111, 1110, -12-13]
        for w in X_w:
            key = []
            for wi in w:
                for a in wi:
                    key.append(a)
            key_list.append(''.join(map(str, key)))

        J_dict = {}
        for key in key_list:
            if key not in J_dict.keys():
                J_dict[key] = 1
            else:
                J_dict[key] += 1

        H = []
        for key in J_dict.keys():
            count = J_dict[key]
            p = count / len(key_list)
            H.append(-p * np.log(p))

        en = np.sum(H)
        return en


# EMD with extrema extension
class EMDBE():

    def __init__(self, series):
        self.series = series

    def addBound(self):

        series = self.series
        n = 8
        max_index = argrelmax(series)[0]
        min_index = argrelmin(series)[0]
        max_data = [series[i] for i in max_index]
        min_data = [series[i] for i in min_index]

        # Initialize variables
        lefted_series = list(series)  # Default to the original series
        lefted_num = 0
        righted_series = list(series)  # Default to the original series
        righted_num = 0

        # lefted_num = 0
        if max_index[0] < min_index[0]:
            #########################################################
            # if series[0] > min_data[0]:
            # if series[0][0] > min_data[0][0]:
            if (series[0] > min_data[0]).any():
                # if (series[0] > min_data[0]).all():
                extra_data = []
                for i in range(max_index[0] + 1, n + 1 + 1):
                    extra_data.append(series[i])
                lefted_series = []
                for i in range(len(extra_data)):
                    index = len(extra_data) - 1 - i
                    lefted_series.append(extra_data[index])
                for i in range(max_index[0], len(series)):
                    lefted_series.append(series[i])
                if len(lefted_series) - len(series) > 0:
                    lefted_num = len(lefted_series) - len(series)

            #########################################################
            # elif series[0] <= min_data[0]:
            # elif series[0][0] <= min_data[0][0]:
            elif (series[0] <= min_data[0]).any():
                # elif (series[0] <= min_data[0]).all():

                lefted_series = []
                extra_data = []
                for i in range(1, n + 1):
                    extra_data.append(series[i])

                for i in range(len(extra_data)):
                    index = len(extra_data) - 1 - i
                    lefted_series.append(extra_data[index])
                for i in range(len(series)):
                    lefted_series.append(series[i])
                lefted_num = n

        elif max_index[0] > min_index[0]:
            #########################################################
            # if series[0] < max_data[0]:
            # if series[0][0] < min_data[0][0]:
            if (series[0] < min_data[0]).any():
                # if (series[0] < min_data[0]).all():
                extra_data = []
                for i in range(min_index[0] + 1, n + 1 + 1):
                    # ex_index = 2 * max_index[0] - i
                    extra_data.append(series[i])
                lefted_series = []
                for i in range(len(extra_data)):
                    index = len(extra_data) - 1 - i
                    lefted_series.append(extra_data[index])
                for i in range(min_index[0], len(series)):
                    lefted_series.append(series[i])
                if len(lefted_series) - len(series) > 0:
                    lefted_num = len(lefted_series) - len(series)

            #########################################################
            # elif series[0] >= max_data[0]:
            # elif series[0][0] >= min_data[0][0]:
            elif (series[0] >= min_data[0]).any():
                # elif (series[0] >= min_data[0]).all():
                lefted_series = []
                extra_data = []
                for i in range(1, n + 1):
                    extra_data.append(series[i])

                for i in range(len(extra_data)):
                    index = len(extra_data) - 1 - i
                    lefted_series.append(extra_data[index])
                for i in range(len(series)):
                    lefted_series.append(series[i])

                lefted_num = n

        series = np.array(lefted_series)
        max_index = argrelmax(series)[0]
        min_index = argrelmin(series)[0]
        max_data = [series[i] for i in max_index]
        min_data = [series[i] for i in min_index]

        # ---------------------from right---------------------
        righted_num = 0
        if max_index[-1] > min_index[-1]:
            #########################################################
            # if series[-1] > min_data[-1]:
            # if series[-1][-1] > min_data[-1][-1]:
            if (series[-1] > min_data[-1]).any():
                # if (series[-1] > min_data[-1]).all():
                extra_data = []
                for i in range(n):
                    extra_data.append(series[max_index[-1] - 1 - i])
                righted_series = []

                for i in range(max_index[-1] + 1):
                    righted_series.append(series[i])
                for i in range(len(extra_data)):
                    righted_series.append(extra_data[i])
                if len(righted_series) - len(series) > 0:
                    righted_num = len(righted_series) - len(series)

            #########################################################
            # elif series[-1] <= min_data[-1]:
            # elif series[-1][-1] <= min_data[-1][-1]:
            elif (series[-1] <= min_data[-1]).any():
                # elif (series[-1] <= min_data[-1]).all():

                extra_data = []
                for i in range(n):
                    extra_data.append(series[-1 - 1 - i])
                righted_series = []

                for i in range(len(series)):
                    righted_series.append(series[i])
                for i in range(len(extra_data)):
                    righted_series.append(extra_data[i])

                righted_num = n

        elif max_index[-1] < min_index[-1]:

            #########################################################
            # if series[-1] < max_data[-1]:
            # if series[-1][-1] < min_data[-1][-1]:
            if (series[-1] < min_data[-1]).any():
                # if (series[-1] < min_data[-1]).all():
                extra_data = []
                for i in range(n):
                    extra_data.append(series[min_index[-1] - 1 - i])

                righted_series = []

                for i in range(min_index[-1] + 1):
                    righted_series.append(series[i])

                for i in range(len(extra_data)):
                    righted_series.append(extra_data[i])
                if len(righted_series) - len(series) > 0:
                    righted_num = len(righted_series) - len(series)

            #########################################################
            # elif series[-1] >= max_data[-1]:
            # elif series[-1][-1] >= min_data[-1][-1]:
            elif (series[-1] >= min_data[-1]).any():
                # elif (series[-1] >= min_data[-1]).all():

                extra_data = []

                for i in range(n):
                    extra_data.append(series[-1 - 1 - i])

                righted_series = []

                for i in range(len(series)):
                    righted_series.append(series[i])

                for i in range(len(extra_data)):
                    righted_series.append(extra_data[i])

                righted_num = n

        else:
            righted_series = np.array(lefted_series)

        return righted_series, lefted_num, righted_num

    def decompose(self):
        series, lefted_num, righted_num = self.addBound()
        series = np.array(series)
        emd = EMD(series.ravel())

        imfs = emd.decompose()
        length = len(imfs[0])

        imfs = imfs[:, lefted_num:length - righted_num]

        return imfs


# the ETFE class to extract features
class ETFE(UnsupervisedDriftDetector):
    def __init__(self, window_len=100, entropy_type='', threshold=0,
                 startup=1500, H=150, seed: int = None,
                 recent_samples_size: int = 500):
        super().__init__(seed=seed, recent_samples_size=recent_samples_size)

        self.window_len = window_len
        self.window_data = deque(maxlen=self.window_len)
        self.count = 0
        self.entropy_list = []
        self.average_entropy_list = []
        self.entropy_type = entropy_type
        self.fig, self.ax = None, None  # Initialize figure and axes
        self.glr_stats = []  # Stores interim GLR statistics
        self.threshold = threshold  # Control limit for GLR test, adjust based on experimentation

        self.startup = startup  # Startup phase
        self.H = H  # Sliding window size
        #self.data_window = []  # Sliding window for recent observations
        self.W = 0  # Sum of observations
        self.P = 0  # Sum of squared deviations
        self.count = 0  # Counter for observations

    def cal_imfs(self, series):

        decomposer = EMDBE(series)
        imfs = decomposer.decompose()

        return imfs

    def cal_entropy(self, series):
        n_series = np.shape(series)[0]
        entropys = []
        for i in range(n_series):
            u = series[i]
            m = 2
            r = 0.2 * np.std(u)
            if self.entropy_type == '':
                en = PeEn(u, 3, 1)
            elif self.entropy_type == 'PeEn':
                en = PeEn(u, 4, 1)
            elif self.entropy_type == 'WPeEn':
                en = WeightedPeEn(u, 4, 1)
            elif self.entropy_type == 'FsEn':
                en = FsEn(u, m, r)
            elif self.entropy_type == 'IncrEn':
                en = IncrEn(u, m, 2)
            elif self.entropy_type == 'ApEn':
                en = ApEn(u, m, r)
            elif self.entropy_type == 'SampEn':
                en = SampEn(u, m, r)

            entropy = en.cal_entropy()
            entropys.append(entropy)
        return entropys

    def calculate_glr(self, entropy_window):
        """
        Compute the Generalized Likelihood Ratio (GLR) statistic for a given rolling window of entropy values.
        """
        q = len(entropy_window)
        if q < 2:
            return 0  # Not enough data

        # Compute cumulative sums and variances
        Wq = np.cumsum(entropy_window)  # Cumulative sum
        P0_q = np.cumsum((entropy_window - np.mean(entropy_window))**2)

        G_values = []
        
        for theta in range(1, q):
            X_i_theta = (Wq[theta] - Wq[0]) / theta
            P_i_theta = P0_q[theta] - P0_q[0] - (0 * (theta - 0) / theta) * (0 - X_i_theta) ** 2

            S0_q = P0_q[q - 1] / (q - 1) if q > 1 else 1
            S0_theta = P0_q[theta - 1] / (theta - 1) if theta > 1 else 1
            S_theta_q = (P0_q[q - 1] - P0_q[theta - 1]) / (q - theta) if q - theta > 0 else 1

            G_theta_q = theta * np.log(S0_q / S0_theta) + (q - theta) * np.log(S0_q / S_theta_q)

            # Bartlett correction factor
            C = 1 + (11 / 12) * ((1 / theta) + (1 / (q - theta)) - (1 / q)) + ((1 / theta**2) + (1 / (q - theta)**2) - (1 / q**2))
            G_values.append(G_theta_q / C)
            # print("G_value:", G_theta_q / C)
        # print("G_max:", max(G_values))
        return max(G_values) if G_values else 0  # Return max GLR value


    def update(self, instance):
        """
        Process an incoming data instance and return True if a drift is detected, otherwise False.
        """
        x_array = np.array(list(instance.values()))

        self.count += 1

        # Check if the window is full
        if len(self.window_data) < self.window_len:
            self.window_data.append(x_array)
            return False  # Not enough data to analyze yet
        else:
            self.window_data.pop()
            self.window_data.append(x_array)

        # # Process every 5th instance to reduce computation
        if self.count % 5 == 0:
            ######################################################

            # Convert window data to a NumPy array
            window_data = np.array(self.window_data)

            # Step 2: Decompose window data into IMFs
            imfs = self.cal_imfs(window_data)

            # Step 3: Calculate entropy for the first two IMFs
            selected_imfs = imfs[
                            :2]  # Assuming the first two IMFs are the highest-frequency components
            entropy_value = self.cal_entropy(selected_imfs)

            # Step 4: Update entropy list and calculate GLR
            self.entropy_list.append(entropy_value)
            if len(self.entropy_list) > self.window_len:
                self.entropy_list.pop(0)

            glr_value = self.calculate_glr(self.entropy_list)

            # Step 5: Check GLR value against threshold
            if glr_value > self.threshold:
                print("drift in", self.count)
                return True  # Drift detected

        return False  # No drift detected

    def run_stream(self, stream, n_training_samples: int, classifier_path):
        return super().run_stream(stream, n_training_samples, classifier_path)
