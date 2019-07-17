import numpy as np

MAX_RANGE = 16.0
CLIP_RANGE = 2.0


class Cropper(object):
    """Crop middle section of array."""
    def __init__(self, size=None):
        if size is not None:
            self.size = int((360 - size)/2)
        else:
            self.size = 0

    def __call__(self, sample):
        if self.size > 0:
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
