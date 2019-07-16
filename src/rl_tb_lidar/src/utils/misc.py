import numpy as np

MAX_RANGE = 16.0
CLIP_RANGE = 2.0


class Cropper(object):
    """Crop middle section of array."""
    def __init__(self, real_turtlebot = False, size=None):
        self.real_turtlebot = real_turtlebot
        if self.real_turtlebot:
            if size is not None:
                self.size = int(size/2)
            else:
                self.size = 0
        else:
            if size is not None:
                self.size = int((360 - size)/2)
            else:
                self.size = 0

    def __call__(self, sample):
        if self.size > 0:
            if self.real_turtlebot:
                tmp_1 = sample[0:self.size]
                tmp_2 = sample[360-self.size:360]
                sample = np.concatenate((tmp_2, tmp_1), axis=None)
            else:
                sample = sample[self.size:-self.size]
        return sample


class ClipRange(object):
    """Crop middle section of array."""
    def __init__(self, clip=CLIP_RANGE, maximum=MAX_RANGE):
        self.clip = min(clip, maximum)

    def __call__(self, sample):
        sample[sample > self.clip] = self.clip
        return sample


class MinMaxScaler(object):
    """Scale values b/w 0.0 and 5.0. Use with output sigmoid layer."""
    def __init__(self, clip=CLIP_RANGE, maximum=MAX_RANGE):
        self.scaler = min(clip, maximum)

    def __call__(self, sample):
        sample = sample/self.scaler
        return sample


class ToTensor(object):
    """Convert array into Tensors."""
    def __call__(self, sample):
        size = len(sample)
        torch_sample = torch.from_numpy(sample.copy())
        return torch_sample.view(1, size).float()
