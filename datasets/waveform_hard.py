import numpy as np
from river.datasets.synth.waveform import Waveform


class WaveformDriftHard(Waveform):
    """Harder WaveformDrift2 variant: noise features + more h_functions."""

    def __init__(
        self,
        drift_frequency: int,
        stream_length: int,
        seed: int or None = None,
        has_noise: bool = True,
    ):
        super().__init__(seed, has_noise)
        base_h = np.asarray(self._H_FUNCTION)
        self.h_functions = [
            base_h + 0,
            base_h + 6,
            6 - base_h,
            base_h * -1,
            base_h + 3,
            3 - base_h,
            base_h * 0.5,
            base_h * 1.5,
        ]
        self._H_FUNCTION = self.h_functions[0]
        self.drift_frequency = drift_frequency
        self.stream_length = stream_length
        self.rng = None
        self.drifts = [i * self.drift_frequency for i in range(int(stream_length / drift_frequency))][1:]

    def drift(self):
        new_function = self.rng.choice(self.h_functions, 1)[0]
        if np.all(self._H_FUNCTION == new_function):
            return self.drift()
        else:
            return new_function

    def __iter__(self):
        self._H_FUNCTION = self.h_functions[0]
        self.rng = np.random.default_rng(seed=self.seed)
        i = 0
        for x, y in super().__iter__():
            if i == self.stream_length:
                break
            if i % self.drift_frequency == 0:
                self._H_FUNCTION = self.drift()
            i += 1
            yield x, y
