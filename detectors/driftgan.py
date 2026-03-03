from itertools import islice
from copy import deepcopy
from random import random

import pandas as pd

from .base import SemisupervisedDriftDetector
from river.drift.page_hinkley import PageHinkley as PHT

from time import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adadelta

"""
This file contains the description for the Generator and Discriminator
"""
from torch import nn
from torch.nn import Sequential, Linear, ReLU, Module


class Generator(Module):
    def __init__(self, inp, out, sequence_length=2, num_layers=3):
        super(Generator, self).__init__()
        self.net = Sequential(
            Linear(inp*sequence_length, 128),
            Linear(128, 4096), ReLU(inplace=True),
            Linear(4096, inp)
        )

    def forward(self, x_):
        output = self.net(x_.reshape(x_.shape[0], x_.shape[1] * x_.shape[2]))
        # output = output.reshape(output.shape[0], output.shape[1] * output.shape[2])
        return output

    def move(self, device):
        pass


class Discriminator(Module):
    def __init__(self, inp, final_layer_incoming_connections=512):
        super(Discriminator, self).__init__()
        self.input_connections = inp
        self.neuron_count = 2
        self.incoming_connections = final_layer_incoming_connections

        self.net = self.create_network()

        self.neurons = Linear(final_layer_incoming_connections, self.neuron_count)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_):
        result = self.net(x_)
        result = self.neurons(result)
        result = self.softmax(result)
        return result

    def update(self):
        # self.reset_layers()
        self.neuron_count += 1
        layer = Linear(self.incoming_connections, self.neuron_count)
        self.neurons = layer
        return

    def reset_top_layer(self):
        # self.reset_layers()
        layer = Linear(self.incoming_connections, self.neuron_count)
        self.neurons = layer
        return

    def reset_layers(self):
        self.net = self.create_network()

    def create_network(self):
        net = Sequential(
            Linear(self.input_connections, 1024),
            Linear(1024, 1024), ReLU(inplace=True),
            Linear(1024, self.incoming_connections),
            nn.Sigmoid())

        return net

class DriftGan(SemisupervisedDriftDetector):
    """

    """

    def __init__(
        self,
        max_dataset_size = 50000,
        epochs = 150,
        repeat_factor = 25,
        equalize = True,
        sequence_length=10,
        steps_generator = 100,
        batch_size = 8,
        generator_batch_size = 8,
        test_batch_size = 4,
        lr = 0.001,
        weight_decay = 0.00005,

    ):
        """

        """
        super().__init__()
        # Set all parameters for the experiment
        # Maximum dataset size to be considered as keeping track of previous drifts slows the system down considerably
        # Default: 50000
        self.max_dataset_size = max_dataset_size
        # Training window size. Default: 100
        #self.training_window_size = training_window_size # defined by the setup n_training_samples
        # Training epochs. Default: 150
        self.epochs = epochs
        # Set repeat factor. 1/factor will be the number of instances from previous instances that are considered for training
        # Default: 25. This means 4% data from previous identical drift windows will be added to the current training data
        self.repeat_factor = repeat_factor
        # Equalize the number of training instances across different drifts. Default: True
        self.equalize = equalize
        # Sequence length for the generator. Default: 10
        self.sequence_length = sequence_length
        # Training steps. default: 100
        self.steps_generator = steps_generator
        # Set the batch_size of the discriminator. Default: 8
        self.batch_size = batch_size
        # Batch size for training the generator
        self.generator_batch_size = generator_batch_size
        # Number of instances that should have the same label for a drift to be confirmed. Default: 4
        self.test_batch_size = test_batch_size
        # Set the learning rate. Default: 0.001
        self.lr = lr
        # Set the weight decay rate. Default: 0.00005
        self.weight_decay = weight_decay
        # For the collate function to split the rows accordingly
        self.seq_len = sequence_length,
        # Set the training to cpu or gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





    def update(self, error) -> bool:
        self.detect_tool.update(error)
        return self.detect_tool.drift_detected



    def run_stream(self, stream, n_training_samples: int, classifier_path):

        #################
        # Standardization
        x_2be_std = []
        y_lab = []
        for x, y in stream:
            y_lab.append(y - 1)
            x_2be_std.append(list(x.values()))
        features = np.array(x_2be_std)
        mean = np.mean(features, axis=1).reshape(features.shape[0], 1)
        std = np.std(features, axis=1).reshape(features.shape[0], 1)

        #standardized_features = (features - mean) / (std + 0.000001)
        concatenated_features = features
        features = (features - mean) / (std + 0.000001)


        current_batch_size = self.batch_size

        y_pred = []
        y_true = []

        classes = np.unique(y_lab)
        x = concatenated_features[:n_training_samples, :]
        y = y_lab[:n_training_samples]

        drifts_detected = []
        generator_label = 1

        # Create the Generator and Discriminator objects
        generator = Generator(inp=features.shape[1], out=features.shape[1],
                              sequence_length=self.sequence_length)
        discriminator = Discriminator(inp=features.shape[1],
                                      final_layer_incoming_connections=512)

        generator.move(device=self.device)

        # Set the models to the device
        generator = generator.to(device=self.device)
        discriminator = discriminator.to(device=self.device)

        drift_indices = [(0, n_training_samples)]  # Initial training window
        drift_labels = []

        temp_label = [0]

        initial_epochs = self.epochs * 2

        predicted, classifier_path = fit_and_predict(clf=classifier_path, features=x, labels=y,
                                                     classes=classes)
        y_pred = y_pred + predicted.tolist()
        y_true = y_true + y

        # Create training dataset
        training_dataset = create_training_dataset(dataset=features,
                                                   indices=drift_indices,
                                                   drift_labels=[0])

        generator, discriminator = train_gan(features=training_dataset,
                                             device=self.device,
                                             discriminator=discriminator,
                                             generator=generator,
                                             epochs=initial_epochs,
                                             steps_generator=self.steps_generator,
                                             seed=np.random.randint(65536), batch_size=self.batch_size,
                                             lr=self.lr, equalize=self.equalize,
                                             max_label=generator_label,
                                             generator_batch_size=self.generator_batch_size,
                                             weight_decay=self.weight_decay,
                                             sequence_length=self.sequence_length)

        index = n_training_samples

        generator.eval()
        discriminator.eval()

        while index + n_training_samples < len(features):

            data = features[index:index + self.test_batch_size]
            data_labels = y_lab[index:index + self.test_batch_size]
            result = discriminator(
                torch.Tensor(data).to(torch.float).to(self.device))
            prob, max_idx = torch.max(result, dim=1)
            max_idx = max_idx.cpu().detach().numpy()
            if np.any(max_idx != max_idx[0]) or max_idx[0] == 0:
                predicted, classifier_path = predict_and_fit(clf=classifier_path,
                                                             features=concatenated_features[
                                                                  index:index + self.test_batch_size],
                                                             labels=data_labels,
                                                             classes=classes)
                y_pred = y_pred + predicted.tolist()
                y_true = y_true + data_labels
                index += self.test_batch_size
                continue

            max_idx = max_idx[0]
            # Drift detected
            drift_indices.append((index, index + n_training_samples))

            if temp_label[0] != 0:
                drift_labels.append(temp_label[
                                        0])  # add the index of the previous drift if it was a recurring drift

            else:
                drift_labels.append(generator_label)

            if max_idx != generator_label:
                # Increase the max_idx by 1 if it is above the previous drift
                if temp_label[0] <= max_idx and temp_label[0] != 0:
                    max_idx += 1
                temp_label = [max_idx]
                # We reset the top layer predictions because the drift order has changed and the network should be retrained
                discriminator.reset_top_layer()
                discriminator = discriminator.to(self.device)
                # 

            else:
                # If this is a new drift, label for the previous drift training dataset is the previous highest label
                # which is the generator label
                temp_label = [0]
                discriminator.update()
                discriminator = discriminator.to(self.device)
                generator_label += 1

            generator = Generator(inp=features.shape[1], out=features.shape[1],
                                  sequence_length=self.sequence_length)
            generator = generator.to(device=self.device)

            generator.train()
            discriminator.train()

            training_dataset = create_training_dataset(dataset=features,
                                                       indices=drift_indices,
                                                       drift_labels=drift_labels + temp_label)

            generator, discriminator = train_gan(features=training_dataset,
                                                 device=self.device,
                                                 discriminator=discriminator,
                                                 generator=generator,
                                                 epochs=self.epochs,
                                                 steps_generator=self.steps_generator,
                                                 seed=np.random.randint(65536),
                                                 batch_size=current_batch_size,
                                                 max_label=generator_label,
                                                 lr=self.lr, equalize=self.equalize,
                                                 weight_decay=self.weight_decay,
                                                 sequence_length=self.sequence_length)

            # Set the generator and discriminator to evaluation mode
            generator.eval()
            discriminator.eval()

            # Set the indices for the training window
            training_idx_start = index
            training_idx_end = training_idx_start + n_training_samples

            # If a previous drift has occurred use those for training the classifier but not predict on them
            if temp_label[0] != 0:
                classifier_path.reset()
                for indices, label in zip(drift_indices[:-1], drift_labels):
                    if label == temp_label[0]:
                        rows = concatenated_features[indices[0]:indices[1], :]
                        targets = y_lab[indices[0]:indices[1]]
                        # Randomly sample .1 of the data
                        len_indices = list(range(0, rows.shape[0]))
                        chosen_indices = random.sample(len_indices, int(
                            rows.shape[0] / self.repeat_factor))
                        # Append rows and targets. Do random.sample and then split the matrix
                        rows = rows[chosen_indices]
                        targets = [targets[x] for x in chosen_indices]
                        classifier_path.fit(X=rows, y=targets, classes=classes)

                predicted, classifier_path = predict_and_fit(clf=classifier_path,
                                                             features=concatenated_features[
                                                                  training_idx_start:training_idx_end,
                                                                  :],
                                                             labels=y_lab[
                                                                training_idx_start:training_idx_end],
                                                             classes=classes)

            else:
                predicted, classifier_path = fit_and_predict(clf=classifier_path,
                                                             features=concatenated_features[
                                                          training_idx_start:training_idx_end,
                                                          :],
                                                             labels=y_lab[
                                                        training_idx_start:training_idx_end],
                                                             classes=classes)

            # Add the predicted and true values to the list
            predicted = predicted.tolist()
            y_pred = y_pred + predicted
            y_true = y_true + y_lab[training_idx_start:training_idx_end]

            drifts_detected.append(index)

            
            index += n_training_samples

        # Test on the remaining features
        features_window = concatenated_features[index:, :]
        labels_window = y_lab[index:]
        y_hat, classifier_path = predict_and_fit(classifier_path, features=features_window,
                                                 labels=labels_window,
                                                 classes=classes)
        y_pred = y_pred + y_hat.tolist()
        y_true = y_true + labels_window

        portion_of_used_labels=[]
        return drifts_detected, y_true, y_pred, portion_of_used_labels

def collate(batch):
    """
    Function for collating the batch to be used by the data loader. This function does not handle labels
    :param batch:
    :return:
    """
    # Stack each tensor variable
    x = torch.stack([torch.tensor(x[:-1]) for x in batch])
    y = torch.Tensor([x[-1] for x in batch]).to(torch.long)
    # Return features and labels
    return x, y


def collate_generator(batch):
    """
    Function for collating the batch to be used by the data loader. This function does handle labels
    :param batch:
    :return:
    """
    global seq_len
    # Stack each tensor variable
    feature_length = int(len(batch[0]) / (seq_len + 1))
    # The last feature length corresponds to the feature we want to predict and
    # the last value is the label of the drift class
    x = torch.stack([torch.Tensor(np.reshape(x[:-feature_length-1], newshape=(seq_len, feature_length)))
                     for x in batch])
    y = torch.stack([torch.tensor(x[-feature_length-1:-1]) for x in batch])
    labels = torch.stack([torch.tensor(x[-1]) for x in batch])
    # Return features and targets
    return x.to(torch.double), y, labels


def fit_and_predict(clf, features, labels, classes):
    predicted = np.empty(shape=len(labels))
    predicted[0] = clf.predict(features[0])[0]
    clf.reset()
    clf.fit(features[0], labels[0])
    for idx in range(1, len(labels)):
        # TODO: make the classifier configurable, 0 refers to first classifier
        predicted[idx] = clf.predict([features[idx]])[0]
        clf.fit(features[idx].tolist(), labels[idx])

    return predicted, clf


def predict_and_fit(clf, features, labels, classes):
    predicted = np.empty(shape=len(labels))
    for idx in range(0, len(labels)):
        predicted[idx] = clf.predict([features[idx]])
        clf.fit([features[idx]], [labels[idx]])

    return predicted, clf


def create_training_dataset(dataset, indices, drift_labels):

    # If there is a periodicity, we switch all previous drifts to the same label
    modified_drift_labels = [x for x in drift_labels]
    if drift_labels[-1] != 0:
        modified_drift_labels = []
        for label in drift_labels:
            if label == drift_labels[-1]:
                modified_drift_labels.append(0)  # The current label
            elif label > drift_labels[-1]:
                modified_drift_labels.append(label-1)  # Decrease all labels that are greater than this
            else:
                modified_drift_labels.append(label)

    training_dataset = np.hstack((dataset[indices[0][0]:indices[0][1]],
                                  np.ones((indices[0][1]-indices[0][0], 1)) * modified_drift_labels[0]))
    for idx in range(1, len(modified_drift_labels)):
        training_dataset = np.vstack((training_dataset, np.hstack((dataset[indices[idx][0]:indices[idx][1]],
                                      np.ones((indices[idx][1]-indices[idx][0], 1)) * modified_drift_labels[idx]))))

    return training_dataset


def train_discriminator(real_data, fake_data, discriminator, generator, optimizer, loss_fn,
                        generator_labels, device):
    # for idx in range(steps):
    for features, labels in real_data:
        # Set the gradients as zero
        discriminator.zero_grad()
        optimizer.zero_grad()

        # Get the loss when the real data is compared to ones
        features = features.to(device).to(torch.float)
        labels = labels.to(device)
        # features = features.to(torch.float)

        # Get the output for the real features
        output_discriminator = discriminator(features)

        # The real data is without any concept drift. Evaluate loss against zeros
        real_data_loss = loss_fn(output_discriminator, labels)

        # Get the output from the generator for the generated data compared to ones which is drifted data
        generator_input = None
        for input_sequence, _, _ in fake_data:
            generator_input = input_sequence.to(device).to(torch.float)
            break
        generated_output = generator(generator_input)  # .double().to(device))

        generated_output_discriminator = discriminator(generated_output)

        # Here instead of ones it should be the label of the drift category
        generated_data_loss = loss_fn(generated_output_discriminator, generator_labels)

        # Add the loss and compute back prop
        total_iter_loss = generated_data_loss + real_data_loss
        total_iter_loss.backward()

        # Update parameters
        optimizer.step()

    return discriminator


def train_generator(data_loader, discriminator, generator, optimizer, loss_fn, loss_mse, steps, device):
    epoch_loss = 0
    for idx in range(steps):

        optimizer.zero_grad()
        generator.zero_grad()

        generated_input = target = labels = None
        for generator_input, target, l in data_loader:
            generated_input = generator_input.to(torch.float).to(device)
            target = target.to(torch.float).to(device)
            labels = l.to(torch.long).to(device)
            break

        # Generating data for input to generator
        generated_output = generator(generated_input)

        # Compute loss based on whether discriminator can discriminate real data from generated data
        generated_training_discriminator_output = discriminator(generated_output)

        # Compute loss based on ideal target values
        loss_generated = loss_fn(generated_training_discriminator_output, labels)

        loss_lstm = loss_mse(generated_output, target)

        total_generator_loss = loss_generated + loss_lstm

        # Back prop and parameter update
        total_generator_loss.backward()
        optimizer.step()
        epoch_loss += total_generator_loss.item()

    return generator


def equalize_classes(features, max_count=100):
    modified_dataset = None

    labels = features[:, -1]
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = min(min(counts), max_count)

    if min_count == max(counts) == max_count:
        return features

    for label, count in zip(unique_labels, counts):
        indices = np.where(features[:, -1] == label)[0]
        chosen_indices = np.random.choice(indices, min_count)
        if modified_dataset is None:
            modified_dataset = features[chosen_indices, :]
            continue
        modified_dataset = np.vstack((modified_dataset, features[chosen_indices, :]))
    return modified_dataset


def concatenate_features(data, sequence_len=2, has_label=True):
    if has_label is True:
        modified_data = data[:, :-1]
    else:
        modified_data = data

    idx = sequence_len
    modified_data = np.vstack((np.zeros((sequence_len - 1, len(modified_data[idx]))), modified_data))
    output = np.hstack((modified_data[idx - sequence_len:idx + 1, :].flatten(), data[idx-sequence_len][-1]))
    idx += 1
    while idx < len(modified_data)-1:
        output = np.vstack((output, np.hstack((modified_data[idx - sequence_len:idx + 1, :].flatten(),
                                               data[idx-sequence_len][-1]))))
        idx += 1

    # The last value
    output = np.vstack((output, np.hstack((modified_data[idx - sequence_len:, :].flatten(), data[-1][-1]))))
    output = np.vstack((output, np.hstack((modified_data[idx - sequence_len:idx, :].flatten(),
                                           modified_data[sequence_len - 1],
                                           data[0][-1]))))
    return output


def train_gan(features, device, discriminator, generator, epochs=100, steps_generator=100, weight_decay=0.0005,
              max_label=1, generator_batch_size=1, seed=0, batch_size=8, lr=0.001, equalize=True,
              sequence_length=2):

    # Set the seed for torch and numpy
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed)

    # Losses for the generator and discriminator
    loss_mse_generator = nn.MSELoss()
    loss_generator = nn.CrossEntropyLoss()
    loss_discriminator = nn.CrossEntropyLoss()

    # Create the optimizers for the models
    optimizer_generator = Adadelta(generator.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_discriminator = Adadelta(discriminator.parameters(), lr=lr, weight_decay=weight_decay)

    # Label vectors
    ones = Variable(torch.ones(generator_batch_size)).to(torch.long).to(device)

    # This data contains the current vector and next vector
    concatenated_data = concatenate_features(features, sequence_len=sequence_length)

    if equalize:
        features = equalize_classes(features)
        concatenated_data = equalize_classes(concatenated_data)

    # Define the data loader for training
    real_data = DataLoader(features, batch_size=batch_size, shuffle=True, collate_fn=collate)
    generator_data = DataLoader(concatenated_data, batch_size=generator_batch_size, shuffle=False,
                                collate_fn=collate_generator)

    # This is the label for new drifts (any input other than the currently learned distributions)
    generator_label = ones * max_label

    for epochs_trained in range(epochs):
        discriminator = train_discriminator(real_data=real_data, fake_data=generator_data, discriminator=discriminator,
                                            generator=generator, optimizer=optimizer_discriminator,
                                            loss_fn=loss_discriminator, generator_labels=generator_label, device=device)

        generator = train_generator(data_loader=generator_data, discriminator=discriminator, generator=generator,
                                    optimizer=optimizer_generator, loss_fn=loss_generator, loss_mse=loss_mse_generator,
                                    steps=steps_generator, device=device)
    return generator, discriminator