from .base import UnsupervisedDriftDetector, BatchDetector

from metrics.computational_metrics import computational_metrics


from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch
from itertools import islice





class Generator(Module):
    def __init__(self, inp, out):
        super(Generator, self).__init__()
        self.net = Sequential(Linear(inp, 128), nn.ReLU(),
                              Linear(128, 128), nn.Sigmoid(),
                              Linear(128, out))

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(Module):
    def __init__(self, inp, out):
        super(Discriminator, self).__init__()
        self.net = Sequential(Linear(inp, 128), ReLU(inplace=True),
                              Linear(128, 256),
                              Linear(256, 512),
                              Dropout(inplace=True),
                              Linear(512, out), nn.Sigmoid())

        # self.net.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        return x


class Network(Module):
    def __init__(self, inp, out):
        super(Network, self).__init__()
        self.net = Sequential(BatchNorm1d(num_features=inp),
                              Linear(inp, 128), ReLU(inplace=True),
                              Linear(128, 256),
                              Linear(256, 512),
                              Dropout(inplace=True),
                              Linear(512, out), nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        return x



class NNBDD(UnsupervisedDriftDetector, BatchDetector):
# class NNBDD(BatchDetector):

    def __init__(self, batch_size: int = 4, threshold: float = 0.5,
                 seed=np.random.randint(65536), epochs = 200,):
        super().__init__(seed=seed,
                         batch_size=batch_size)
        self.threshold = threshold
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.device = None
        self.generated_input = None
        self.standardized_features = None
        self.discriminator = None
        self.generator = None
        self.outliers = None
    
    def generate_noise(self, shape):
        """
        Function to generate noise of a required shape.
        :param shape: Tuple that specifies the required shape of noise returned
        :return:
        """
        return np.random.random(size=shape)


    def collate(self, batch):
        """
        Function for collating the batch to be used by the data loader. This function does not handle labels
        :param batch:
        :return:
        """
        # Stack each tensor variable
        x = torch.stack([torch.tensor(x) for x in batch])

        # Return features and labels
        return x, None


    def collate_train(self, batch):
        """
        Function for collating the batch to be used by the data loader. This function does handle labels
        :param batch:
        :return:
        """
        # Stack each tensor variable
        x = torch.stack([torch.tensor(x[:-1]) for x in batch])
        y = torch.stack([torch.tensor([x[-1]]) for x in batch])
        # Return features and labels
        return x, y
    

    def get_discriminator(self, standardized_features, device, epochs=100, steps_generator=100, steps_discriminator=100,
                        seed=0, noisy_input_size=8, batch_size=8, lr=0.0001, momentum=0.9):

        # Set the seed for torch and numpy
        torch.manual_seed(seed=seed)
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)
        np.random.seed(seed)

        # Create the loss functions for the discriminator and generator
        loss_discriminator = nn.BCELoss()
        loss_generator = nn.BCELoss()

        # Create the generator and discriminator objects and set them to double.
        # The input size to the generator is the noisy input size and the generated vector is the size of a feature vector
        generator = Generator(inp=noisy_input_size, out=standardized_features.shape[1])
        generator.double()
        discriminator = Discriminator(inp=standardized_features.shape[1], out=1)
        discriminator.double()

        loss_array_generator = []
        loss_array_discriminator = []

        # Create the optimizers for the generator and discriminator
        optimizer_generator = optim.SGD(generator.parameters(), lr=lr, momentum=momentum)
        optimizer_discriminator = optim.SGD(discriminator.parameters(), lr=lr, momentum=momentum)

        # Create the data loader for the features which is the real data
        real_data = DataLoader(standardized_features, batch_size=batch_size, shuffle=True, collate_fn=self.collate)

        # Set the generator and discriminator to the GPU/CPU depending on the parameter
        discriminator.to(device)
        generator.to(device)

        # Variables for computing loss of generator and discriminator
        ones = Variable(torch.ones(batch_size, 1)).double().to(device)  # Indicates real data
        zeros = Variable(torch.zeros(batch_size, 1)).double().to(device)  # Indicates generated data

        for epoch in range(epochs):

            total_loss_generator = 0
            total_loss_discriminator = 0

            for step_d in range(steps_discriminator):
                discriminator.zero_grad()
                x = None
                # Train discriminator on actual data
                for real_input, _ in real_data:
                    x = real_input
                    break

                # Get the loss when the real data is compared to ones
                x = x.to(device).double()
                output_discriminator = discriminator(x)
                # real_loss_discriminator = loss_discriminator(output_discriminator,
                                                            # ones)
                real_loss_discriminator = loss_discriminator(output_discriminator.float(), ones.float())


                # Train discriminator on drifted/noise data
                generator_input = self.generate_noise(shape=[batch_size, noisy_input_size])

                # Get the output from the generator for the generated data compared to zeroes
                generated_output = generator(torch.Tensor(generator_input).double().to(device))

                generated_output = generated_output.to(device)
                generated_output_discriminator = discriminator(generated_output)
                generated_loss_discriminator = loss_discriminator(generated_output_discriminator, zeros)

                # Add the loss and compute back prop
                total_iter_loss = generated_loss_discriminator + real_loss_discriminator
                total_iter_loss.backward()

                total_loss_discriminator += total_iter_loss

                # Update parameters
                optimizer_discriminator.step()

            for step_g in range(steps_generator):
                generator.zero_grad()

                # Generating data for input to generator
                generated_input = self.generate_noise(shape=[batch_size, noisy_input_size])
                generated_input = torch.Tensor(generated_input).double().to(device)
                generated_output = generator(generated_input)

                # Compute loss based on whether discriminator can discriminate real data from generated data
                generated_training_discriminator_output = discriminator(generated_output)
                generated_training_discriminator_output = generated_training_discriminator_output.to(device)

                # Compute loss based on ideal target values which are ones
                loss_generated = loss_generator(generated_training_discriminator_output,
                                                ones)

                # Back prop and parameter update
                loss_generated.backward()
                total_loss_generator += loss_generated
                optimizer_generator.step()

            epoch_loss_generator = total_loss_generator.cpu().detach().numpy()/steps_generator
            epoch_loss_discriminator = total_loss_discriminator.cpu().detach().numpy()/steps_discriminator

            loss_array_generator.append(epoch_loss_generator)
            loss_array_discriminator.append(epoch_loss_discriminator)

        return discriminator, generator

            
    def train_network(self, old_features, new_features, batch_size=4, lr=0.0001, momentum=0.9, epochs=100, device="cpu", seed=0):
        # Set the seed for torch and numpy
        torch.manual_seed(seed=seed)
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)
        np.random.seed(seed)

        # Create the labels for the new and old features
        ones = np.ones(shape=(len(new_features), 1))
        zeros = np.zeros(shape=(len(old_features), 1))
        x_old = np.hstack((old_features, zeros))
        x_new = np.hstack((new_features, ones))
        training_set = np.vstack((x_old, x_new))

        # Initialize the network, and send it to device
        network = Network(inp=old_features.shape[1], out=1)
        network = network.double()
        network = network.to(device)

        dl = DataLoader(training_set, batch_size=batch_size, shuffle=True, collate_fn=self.collate_train)
        optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum)
        loss_ = nn.BCELoss()
        for idx in range(epochs):
            for batch_x, batch_y in dl:
                network.zero_grad()
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = network(batch_x)
                curr_loss = loss_(out, batch_y)
                curr_loss.backward()
                optimizer.step()

        return network

    
    
    def standardize_samples(self, batch):
        features = np.array([np.array(list(x[0].values())) for x in batch])
        # Compute mean & std
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std == 0, 1, std)  # Prevent division by zero
        return features, mean, std


    def update(self, batch_id, output) -> bool:
        if np.mean(output.cpu().detach().numpy()) <= self.threshold:
            return True


    @computational_metrics
    def process_main_batch_stream(self, stream, n_training_samples: int,
                                  classifier):

        # Get the device the experiment will run on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        limit = int(self.recent_samples_size/2)
        self.generated_input = self.generate_noise(shape=[limit, 8])
        self.generated_input = torch.Tensor(self.generated_input).double().to(self.device)

        

        # Train the initial discriminator value
        features, _, std = self.standardize_samples(self.recent_samples)

        self.standardized_features = features / std
        self.discriminator, self.generator = self.get_discriminator(standardized_features=self.standardized_features, epochs=self.epochs, seed=self.seed,
                                                    device=self.device)
        
        self.outliers = self.generator(self.generated_input)
        self.outliers = self.outliers.cpu().detach().numpy()

        # Processing the rest of the stream for detecting drifts
        for batch_id, batch in enumerate(self.batch_stream(
                stream, n_training_samples)):

            if len(self.recent_samples)==self.recent_samples_size:
                # print("Processing batch ", batch_id)

                # Updating recent samples
                del self.recent_samples[:len(batch)]
                self.recent_samples.extend(batch)

                features, _, _ = self.standardize_samples(self.recent_samples)
                self.standardized_features = features / std
                output = self.discriminator(torch.from_numpy(self.standardized_features[-self.batch_size:, :]).to(self.device))

                # Check if mean predictions are below threshold
                if self.update(batch_id, output):  # Use the drift detector's update method
                                # Drift is detected.
                    self.handle_batch_update(batch_id, n_training_samples)
                    # print("Drift is detected in batch ", batch_id)
                    for i, (x, y) in enumerate(self.recent_samples):
                        classifier.reset()
                        classifier.fit(x, y)
                    
                    old_standardized_features = self.standardized_features
                    self.recent_samples = []

                for i, (x, y) in enumerate(batch):
                    self.predictions.append(classifier.predict(x))
                    self.labels.append(y)
            else:
                self.recent_samples.extend(batch)
                if len(self.recent_samples)>self.recent_samples_size:
                    self.recent_samples = self.recent_samples[-self.recent_samples_size:]
                if len(self.recent_samples)==self.recent_samples_size:
                    # Seed to be changed every time the discriminator is retrained
                    self.seed = np.random.randint(65536)
                    
                    features, _, std = self.standardize_samples(self.recent_samples)
                    self.standardized_features = features / std

                    # retrain the network
                    self.discriminator = self.train_network(old_features=old_standardized_features,
                                                new_features=self.standardized_features, batch_size=16,
                                                device=self.device, epochs=self.epochs, seed=self.seed)
                
        return (self.drifts, self.labels, self.predictions,
                len(self.used_labels_set))
    
    
    def run_stream(self, stream, n_training_samples: int, classifier_path):
        return super().run_batch_stream(stream, n_training_samples,
                                        classifier_path)