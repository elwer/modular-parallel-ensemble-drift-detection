from .base import UnsupervisedDriftDetector

import numpy as np
import operator
import pandas as pd
from sklearn.linear_model import SGDClassifier
from itertools import islice

clf = SGDClassifier(n_jobs=-1, loss='hinge', random_state=1, warm_start=True)


def to_numpy_matrix(Y, self_loops=False):
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if Y.ndim != 2:
        raise ValueError("Input Y must be a 2D array")

    t = np.transpose(Y)
    freqs = t @ Y
    if not self_loops:
        np.fill_diagonal(freqs, 0)

    return freqs


class LD3(UnsupervisedDriftDetector):
    def __init__(self, k=2, window_size=500, detection_window_size=1,
                 label_count=10000, big_dataset=False):
        self.history = {}
        self.prev = []
        self.detection_window_size = detection_window_size
        self.k = k
        self.window_size = window_size
        self.labels = []
        self.warmup = True
        #self.label_count = label_count
        self.big_dataset = big_dataset
        self.decrease_rate = 0
        self.increase_rate = 1

        # self.index = 0
        # self.clf = SGDClassifier(n_jobs=-1, loss='hinge',  random_state=1, warm_start=True)

    def add_element(self, y):
        # 
        if len(self.labels) < self.window_size:
            self.labels.extend(y)
            # 

    def recip_rank(self, mat):
        num_labels = len(mat)
        ranks = np.flip(np.argsort(mat), axis=1)
        ranks = np.delete(ranks, [
            np.argwhere(i == ranks[i]).flatten()[0] + i * num_labels for i in
            range(num_labels)]).reshape(num_labels, num_labels - 1)
        return np.argsort(
            [1 / np.sum(1 / (np.argwhere(ranks == i)[:, 1] + 1)) for i in
             range(num_labels)])

    def add_to_history(self, item):
        if item not in self.history:
            self.history[item] = 1
        else:
            if self.history[item] < 0:
                self.history[item] = 0
            #self.history[item] += 1
            current_hist = np.array(list(
                sorted(self.history.items(), key=operator.itemgetter(1),
                       reverse=True)))[:self.k][:, 0]
            if item not in current_hist and item == self.prev:
                self.history[item] += 2 ** self.increase_rate
            else:
                self.history[item] += 1
        '''if len(self.history) < self.detection_window_size:
            self.history.append(item)
        else:
            self.history.pop(0)
            self.history.append(item)'''

    def detected_change(self):
        if len(self.labels) % self.window_size == 0 and len(self.labels) != 0:
            
            
            # 
            # if isinstance(self.labels, list):
            #     
            if len(self.prev) < 1:
                
                r = self.recip_rank(to_numpy_matrix(self.labels))[:self.k]
                
                #r = np.flip(np.argsort(to_numpy_matrix(self.labels).sum(axis=0)))[:self.k]
                for item in r:
                    self.add_to_history(item)
                #
                #self.prev = r
                self.prev = np.array(list(
                    sorted(self.history.items(), key=operator.itemgetter(1),
                           reverse=True))[:self.k])[:, 0]
                self.labels = []
            else:
                
                r = self.recip_rank(to_numpy_matrix(self.labels))[:self.k]
                #r = np.flip(np.argsort(to_numpy_matrix(self.labels).sum(axis=0)))[:self.k]
                for item in r:
                    self.add_to_history(item)

                #diff = np.setdiff1d(r, self.prev)
                current_hist = np.array(list(
                    sorted(self.history.items(), key=operator.itemgetter(1),
                           reverse=True)))
                
                
                if len(current_hist) > 1:
                    
                    if current_hist[0][1] <= current_hist[1][
                        1] and self.warmup:
                        
                        
                        #
                        
                        self.labels = []
                        self.prev = current_hist[:self.k][:, 0]
                        return False
                    elif current_hist[0][1] > current_hist[1][
                        1] and self.warmup:
                        
                        
                        #
                        
                        self.labels = []
                        self.prev = current_hist[:self.k][:, 0]
                        self.warmup = False
                        return False
                else:
                    
                    if current_hist[0][1] > 1:
                        
                        
                        #
                        
                        self.labels = []
                        self.prev = current_hist[:self.k][:, 0]
                        self.warmup = False
                        return False
                    else:
                        
                        
                        #
                        
                        self.labels = []
                        self.prev = current_hist[:self.k][:, 0]
                        return False

                if r[0] != self.prev[0]:
                    
                    self.history[self.prev[0]] -= 2 ** self.decrease_rate
                    current_hist = np.array(list(sorted(self.history.items(),
                                                        key=operator.itemgetter(
                                                            1), reverse=True)))
                    self.decrease_rate += 1
                else:
                    
                    self.decrease_rate = 0

                #
                #

                diff = np.setdiff1d(current_hist[:self.k][:, 0], self.prev)
                self.labels = []
                #self.prev = r
                self.prev = current_hist[:self.k][:, 0]
                if len(diff) > 0:
                    
                    self.warmup = True
                    self.decrease_rate = 0
                    self.increase_rate = 1
                    return True
        return False

    def update(self, x) -> bool:
        self.add_element(x)
        return self.detected_change()

    def run_stream(self, stream, n_training_samples: int, classifier_path):

        used_labels_set = set()

        recent_samples = []
        clf_samples = []
        index = 0
        # Initial training data
        for x, y in islice(stream, n_training_samples):
            classifier_path.fit(x, y)
            recent_samples.append((x, y))
            x_array = np.array(list(x.values()))
            clf_samples.append(x_array)

        clf_labels = np.zeros(n_training_samples)
        clf_labels[n_training_samples // 2:] = 1
        clf.fit(clf_samples, clf_labels)

        drifts = []
        labels = []
        predictions = []

        # Processing the rest of the stream for detecting drifts
        for i, (x, y) in enumerate(islice(stream, n_training_samples, None),
                                   start=n_training_samples):
            # 

            recent_samples.pop(0)
            recent_samples.append((x, y))

            predictions.append(classifier_path.predict(x))
            labels.append(y)

            clf_new_label = clf.predict(x_array.reshape(1, -1))
            
            clf.partial_fit(x_array.reshape(1, -1), clf_new_label)

            if self.update(
                    clf_new_label):  # Use the drift detector's update method
                
                drifts.append(i)
                used_labels = range(i - n_training_samples + 1, i + 1)
                used_labels_set.update(used_labels)
                # classifiers.reset()
                for new_x, new_y in recent_samples:
                    classifier_path.fit(new_x, new_y)

        return (self.drifts, self.labels, self.predictions,
                len(self.used_labels_set))
