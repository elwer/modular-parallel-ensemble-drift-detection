import numpy as np
from river import datasets


class SineClustersHighDim(datasets.base.SyntheticDataset):
    """SineClusters variant with configurable number of features.

    Generates n_features-dimensional datapoints using sine/cosine concepts
    with Gaussian noise. Only the first 2 features determine the label
    (same as original SineClusters), so drift detection difficulty is
    preserved while detector.update() does more work per sample.
    """

    def __init__(
        self,
        drift_frequency,
        stream_length,
        seed: int or None = None,
        n_features: int = 16,
    ):
        super().__init__(
            task=datasets.base.MULTI_CLF,
            n_features=n_features,
        )
        self.drift_frequency = drift_frequency
        self.stream_length = stream_length
        self.n_features = n_features
        self.centroids = None
        self.sine_period = 313
        self.indices = np.linspace(0, 2 * np.pi, self.sine_period + 1)
        self.seed = seed
        self.rng = None
        self.get_features = None
        self.drifts = [i * self.drift_frequency for i in range(int(stream_length / drift_frequency))][1:]

    def get_label(self, features):
        features = features[:2]
        distances = np.linalg.norm(self.centroids - features, axis=1),
        closest_centroid = np.argmin(distances)
        return closest_centroid % 3

    def set_centroids(self):
        centroids = []
        for i in range(6):
            centroids.append(self.rng.uniform(low=-1, high=1, size=2))
        self.centroids = np.array(centroids)

    def __iter__(self):
        self.rng = np.random.default_rng(self.seed)
        i = 0
        while i < self.stream_length:
            if i % self.drift_frequency == 0:
                self.drift()
            x = self.get_features(i)
            x += self.rng.normal(0, 0.25, self.n_features)
            y = self.get_label(x)
            x = {i: x_component for i, x_component in enumerate(x)}
            i += 1
            yield x, y

    def drift(self):
        self.set_centroids()
        new_concept = self.rng.choice([self.concept_one, self.concept_two, self.concept_three], 1)[0]
        while new_concept == self.get_features:
            new_concept = self.rng.choice([self.concept_one, self.concept_two, self.concept_three], 1)[0]
        self.get_features = new_concept

    def _pad(self, base_values):
        """Take 4 base sine/cosine values and extend to n_features."""
        result = list(base_values)
        while len(result) < self.n_features:
            result.extend(base_values)
        return result[:self.n_features]

    def concept_one(self, i):
        x_sin = np.sin(self.indices[i % self.sine_period])
        x_cos = np.cos(self.indices[i % self.sine_period])
        return self._pad([x_sin, x_cos, x_sin, x_cos])

    def concept_two(self, i):
        x_sin = np.sin(self.indices[i % self.sine_period])
        x_cos = np.cos(self.indices[i % self.sine_period])
        return self._pad([-x_sin, x_cos, x_sin, x_cos])

    def concept_three(self, i):
        x_sin = np.sin(self.indices[i % self.sine_period])
        x_cos = np.cos(self.indices[i % self.sine_period])
        return self._pad([-x_cos, -x_cos, x_sin, x_cos])
