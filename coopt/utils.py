from collections import deque
import math as m
class RunningMeanStd_limited:
    def __init__(self, window_size=10):
        # initial window size
        self.window_size = window_size
        # no fixed maxlen, since we’ll trim manually
        self.window = deque()

    def set_window_size(self, new_size: int):
        """Change the window size at runtime and trim old values."""
        self.window_size = new_size
        # if our buffer is too long, drop extras from the left
        while len(self.window) > new_size:
            self.window.popleft()

    def update(self, x, window_size: int = None):
        """
        Add new sample(s).  
        Optionally override window_size for *this* update.
        """
        # if caller passed a one-off window_size, apply & trim
        if window_size is not None:
            self.set_window_size(window_size)

        # accept a single value or an iterable
        try:
            iterator = iter(x)
        except TypeError:
            iterator = (x,)

        for value in iterator:
            self.window.append(value)
            # enforce current window_size
            if len(self.window) > self.window_size:
                self.window.popleft()

    @property
    def count(self):
        return len(self.window)

    @property
    def mean(self):
        n = self.count
        return sum(self.window)/n if n else 0.0

    @property
    def variance(self):
        n = self.count
        if n == 0:
            return 0.0
        mu = self.mean
        return sum((xi-mu)**2 for xi in self.window)/n

    @property
    def std(self):
        return m.sqrt(self.variance)

    

class RunningMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.S = 0.0      # running sum of squared devs
        self.count = 0

    def update(self, x):
        # x can be a list of new samples; here we do one at a time
        for r in x:
            self.count += 1
            delta = r - self.mean
            self.mean += delta / self.count
            delta2 = r - self.mean
            self.S += delta * delta2

    @property
    def variance(self):
        return self.S / self.count if self.count > 0 else 0.0

    @property
    def std(self):
        return m.sqrt(self.variance)