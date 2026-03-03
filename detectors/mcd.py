from .base import UnsupervisedDriftDetector

import torch
from torch.utils.data import DataLoader
from metrics.computational_metrics import computational_metrics


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size=200, output_size=150):
        super(Encoder, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class WindowedDataset(torch.utils.data.Dataset):
    """A dataset class for windowed data processing."""

    def __init__(self, dataset, window_size, slide, transform=None):
        """
        Initializes the dataset with windowing parameters.

        Args:
            dataset (torch.Tensor): The dataset to window.
            window_size (int): The size of each data window.
            slide (int): The sliding distance between windows.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset = dataset
        self.window_size = window_size
        self.slide = slide
        self.transform = transform

    def __len__(self):
        return max(1,
                   ((len(self.dataset) - self.window_size) // self.slide) + 1)

    def __getitem__(self, idx):
        start = idx * self.slide
        end = start + self.window_size
        data = self.dataset[start:end]
        if self.transform:
            data = self.transform(data)
        return data


class MCDDD(UnsupervisedDriftDetector):
    def __init__(self, epochs=1, sub_window_num=100, n=100, k=10, eps_small=1,
                 eps_big=10, temperature=0.1, lamb=1, percentile=0.95,
                 batch_size=5000, seed: int = None,
                 recent_samples_size: int = 500):
        super().__init__(seed=seed, recent_samples_size=recent_samples_size)
        self.n_samples = None
        self.n_columns = None
        self.learning_rate = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.epochs = epochs
        self.batch_size = None
        self.sub_window_num = sub_window_num
        self.slide = None
        self.n = n
        self.k = k
        self.eps_small = eps_small
        self.eps_big = eps_big
        self.temperature = temperature
        self.lamb = lamb
        self.percentile = percentile
        self.threshold = 0  # Drift detection threshold
        self.batch_size = batch_size

    def generate_samples(self, sub_win_data):
        sub_win_size = sub_win_data.size(0)
        indices = torch.randint(0, sub_win_size, (self.n * self.k,))
        samples = sub_win_data[indices].clone().detach().requires_grad_(True)
        samples = samples.view(self.k, self.n, -1)
        return samples.to(self.device)

    def compute_positive_loss(self, embeddings):
        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
        norm_diff = torch.norm(diff, p=2, dim=2)
        mask = torch.triu(torch.ones_like(norm_diff), diagonal=1).bool()
        masked_norm_diff = norm_diff.masked_select(mask)
        mean_loss = masked_norm_diff.mean()
        return mean_loss, masked_norm_diff

    def compute_negative_loss(self, embeddings1, embeddings2):
        diff = embeddings1.unsqueeze(1) - embeddings2.unsqueeze(0)
        norm_diff = torch.norm(diff, p=2, dim=2)
        mask = torch.eye(norm_diff.size(0), dtype=torch.bool,
                         device=norm_diff.device)
        masked_norm_diff = norm_diff.masked_select(~mask)
        mean_loss = masked_norm_diff.mean()
        return mean_loss

    def contrastive_loss_function(self, pos_losses, neg_losses, temperature):
        """
        like InfoNCE
        """
        exp_pos_losses = torch.exp(torch.stack(pos_losses) / temperature)
        exp_neg_losses = torch.exp(torch.stack(neg_losses) / temperature)
        numerator = torch.sum(exp_pos_losses)
        denominator = numerator + torch.sum(exp_neg_losses)
        loss = torch.log(numerator / denominator)
        return loss

    def gradient_penalty(self, samples, output):
        gradients = torch.autograd.grad(outputs=output, inputs=samples,
                                        grad_outputs=torch.ones_like(output),
                                        create_graph=True, retain_graph=True,
                                        only_inputs=True)[0]
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        penalty = ((gradients_norm - 1) ** 2).mean()
        return penalty

    def train(self, window_data):
        windows = torch.split(window_data,
                              int(window_data.size(0) / self.sub_window_num))
        sub_win_samples = [self.generate_samples(sub_win) for sub_win in
                           windows]

        for epoch in range(self.epochs):
            pos_losses = []
            weak_neg_losses = []

            for samples in sub_win_samples:
                # Positive sample pairs
                samples = samples.to(self.device)
                # h_p1 and h_p2
                embeddings = self.model(samples)
                embeddings_mean = embeddings.mean(dim=1)

                # Loss for positive sample pairs
                pos_loss, _ = self.compute_positive_loss(embeddings_mean)
                pos_losses.append(pos_loss)

                # Weak negative sample pairs
                unchanged_samples = samples[:self.k // 2]
                altered_samples = samples[self.k // 2:].clone() + torch.normal(
                    mean=0, std=self.eps_small,
                    size=samples[self.k // 2:].shape).to(self.device)

                # h_wn1 and h_wn2
                embeddings_unchanged = self.model(unchanged_samples).mean(
                    dim=1)
                embeddings_altered = self.model(altered_samples).mean(dim=1)

                # Loss for weak negative sample pairs
                weak_neg_loss = self.compute_negative_loss(
                    embeddings_unchanged, embeddings_altered)
                weak_neg_losses.append(weak_neg_loss)

            # Strong negative sample pairs
            # sub-window 1 and sub-window N_sub
            first_sub_win_samples = sub_win_samples[0]
            last_sub_win_samples = sub_win_samples[-1]
            last_sub_win_samples_altered = last_sub_win_samples.clone() + torch.normal(
                mean=0, std=self.eps_big, size=last_sub_win_samples.shape).to(
                self.device)

            # h_sn1 and h_sn2
            embeddings_first = self.model(first_sub_win_samples).mean(dim=1)
            embeddings_last_altered = self.model(
                last_sub_win_samples_altered).mean(dim=1)

            # Loss for strong sample pairs
            strong_neg_loss = self.compute_negative_loss(embeddings_first,
                                                         embeddings_last_altered)

            gp = self.gradient_penalty(samples, embeddings)
            # Total Loss with GP
            extended_neg_losses = weak_neg_losses + [strong_neg_loss]
            total_loss = self.contrastive_loss_function(pos_losses,
                                                        extended_neg_losses,
                                                        self.temperature) + self.lamb * gp

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        # Caculate thresold
        threshold_dis = self.calculate_threshold(windows)
        return threshold_dis

    def calculate_threshold(self, windows):
        all_poslosses = []
        for sub_win in windows:
            samples = self.generate_samples(sub_win)
            embeddings = self.model(samples).mean(dim=1)
            _, all_posloss = self.compute_positive_loss(embeddings)
            all_poslosses.append(all_posloss)
        all_losses_tensor = torch.cat(all_poslosses)
        threshold_dis = torch.quantile(all_losses_tensor, self.percentile)
        return threshold_dis

    def test(self, window_data):
        self.model.eval()  # evaluation

        # New sub-window data points at next sliding window
        windows = torch.split(window_data,
                              int(window_data.size(0) / self.sub_window_num))
        embeddings = []
        distances = []

        # h_j and h_j'
        for sub_win in windows:
            sub_win_samples = self.generate_samples(sub_win)
            with torch.no_grad():
                embedding = self.model(sub_win_samples).mean(dim=1)
                embeddings.append(embedding)

        # Distance between the last sub-window and all previous sub-windows
        # in the same sliding window
        last_interval_embedding = embeddings[-1]
        distances = []
        for embedding in embeddings[:-1]:
            distance = self.compute_negative_loss(embedding,
                                                  last_interval_embedding)
            distances.append(distance)

        return distances

    def update(self, x) -> bool:
        return x[-1] > self.threshold

    @computational_metrics
    def process_main_stream(self, dataloader, dataset, classifier):
        first_window = True
        # Training loop for each window of data
        for i, window_data in enumerate(dataloader):
            start_index = max(0,
                              i * self.slide + self.batch_size - self.slide)
            end_index = min(len(dataset), i * self.slide + self.batch_size)
            window_data = window_data.to(self.device).squeeze(0)

            # remove first [slide] elements to keep recent samples up2date
            # for potential re-training
            self.recent_samples = self.recent_samples[self.slide:]
            # get predictions and labels for the remaining data
            # update recent samples for potential re-training
            for data in dataset[end_index - self.slide:end_index]:
                self.recent_samples.append(data)
                self.predictions.append(classifier.predict(data[1][0]))
                self.labels.append(data[1][1])

            if not first_window:
                distances = self.test(window_data)
                if self.update(distances):
                    # Drift detection
                    print(f"Drift detected between {start_index}"
                          f" and {end_index}")
                    self.drifts.append(start_index)

                    # Train classifiers only with new samples that haven't
                    # been used before
                    for data in self.recent_samples:
                        if data[0] not in self.used_labels_set:
                            classifier.fit(data[1][0], data[1][1])
                            self.used_labels_set.add(data[0])

            self.threshold = self.train(window_data)
            first_window = False

        return (self.drifts, self.labels, self.predictions,
                len(self.used_labels_set))

    def run_stream(self, stream, n_training_samples: int, classifier_path):

        _, classifier = self.load_main_clf(stream, n_training_samples,
                                           classifier_path)
        # recent samples does not match for mcddd due
        # to different data iterator, reset and add n_training samples with
        # id manually
        self.recent_samples = []
        # Load dataset
        dataset = []
        for idx, sample in enumerate(stream):
            if idx < n_training_samples:
                self.recent_samples.append((idx, sample))
            dataset.append((idx, sample))

        # do not pass the label as a feature
        # dataset = (idx,(features, label))
        datasetTensor = torch.tensor([list(d[1][0].values()) for d in dataset],
                                     dtype=torch.float32)

        # Calculate required parameters based on the loaded dataset
        col_length = datasetTensor.shape[1]

        self.n_samples = stream.n_samples
        self.learning_rate = 0.005
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Encoder(input_size=col_length).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)
        self.slide = int(self.batch_size / self.sub_window_num)

        windowed_dataset = WindowedDataset(datasetTensor,
                                           window_size=self.batch_size,
                                           slide=self.slide)
        dataloader = DataLoader(windowed_dataset, batch_size=1, shuffle=False)

        return self.process_main_stream(dataloader, dataset, classifier)
