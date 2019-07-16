import numpy as np

from discretizer import GridDiscretizer
from misc import Cropper, ClipRange, MinMaxScaler, ToTensor

try:
    import torch
    from autoencoders.autoencoders import AutoEncoder, VAE, DiscreteLatentVAE, VectorQuantizedVAE, GumbelVAE
except ImportError:
    pass


MAX_RANGE = 16.0
MIN_RANGE = 0.3
CLIP_RANGE = 2.0
LEVELS = 4
SIZE = 4
GRIDS = 1


class FeaturesMixin(object):
    def __init__(self,
                 size = SIZE,
                 crop = None,
                 **kwargs):
        self.size = size
        self.cropper = Cropper(crop)

    def __call__(self, x):
        z = self.get_features(x)
        return z

    def sample(self):
        try:
            z_sample = np.random.randn(self.levels, size=self.size)
        except:
            z_sample = np.random.randn(self.size)
        return z_sample


class GridFeatures(FeaturesMixin):
    def __init__(self,
                 levels=LEVELS,
                 size=SIZE,
                 crop=None,
                 min_range=MIN_RANGE,
                 max_range=MAX_RANGE,
                 clip_range=CLIP_RANGE,
                 **kwargs):
        super(GridFeatures, self).__init__(size=size, crop=crop)
        self.grid = GridDiscretizer(randomize_bins=False,
                                    levels=levels,
                                    size=size,
                                    enumerate=False,
                                    crop=crop,
                                    min_range=min_range,
                                    max_range=max_range,
                                    clip_range=clip_range)

    def get_features(self, x):
        z = self.grid(x)
        return z


class TileCodingFeatures(FeaturesMixin):
    def __init__(self,
                 grids=GRIDS,
                 levels=LEVELS,
                 size=SIZE,
                 crop=None,
                 min_range=MIN_RANGE,
                 max_range=MAX_RANGE,
                 clip_range=CLIP_RANGE,
                 **kwargs):
        super(TileCodingFeatures, self).__init__(size=size, crop=crop)
        self.size = levels**size
        self.grids = [GridDiscretizer(randomize_bins=i,
                                      levels=levels,
                                      size=size,
                                      enumerate=True,
                                      crop=crop,
                                      min_range=min_range,
                                      max_range=max_range,
                                      clip_range=clip_range) for i in range(grids)]

    def get_features(self, x):
        z = np.zeros(self.size)
        for i, grid in enumerate(self.grids):
            z[grid(x)] += 1.0
        return z


class AEFeatures(FeaturesMixin):
    def __init__(self,
                 path,
                 levels=LEVELS,
                 size=SIZE,
                 crop=None,
                 min_range=MIN_RANGE,
                 max_range=MAX_RANGE,
                 clip_range=CLIP_RANGE,
                 **kwargs):
        super(AEFeatures, self).__init__(size=size, crop=crop)
        self.model = torch.load(path, map_location=lambda storage, loc: storage).eval()

        if isinstance(self.model, VectorQuantizedVAE):
            size = self.model.n_latents
            levels = self.model.categorical_dim
        elif isinstance(self.model, GumbelVAE):
            size = self.model.latent_dim
            levels = self.model.categorical_dim
        elif isinstance(self.model, AutoEncoder):
            size = self.model.latent_dim
        else:
            raise ValueError("Model can only be either VQ-VAE or Gumbel-VAE.")

        assert self.size == size, "Loaded model latent size ({}) not consistent with feature size ({})".format(size, self.size)
        if isinstance(self.model, DiscreteLatentVAE):
            assert kwargs["levels"] == levels, "Loaded model categorical dimension ({}) not consistent with provided feature levels ({})".format(levels, kwargs["levels"])
        assert int(self.cropper.size*2.0) == self.model.input_dim, "Loaded model input dimension ({}) not consistent with feature input dimension ({})".format(self.model.input_dim, int(self.cropper.size*2.0))

        self.clipper = ClipRange(clip=clip_range, maximum=max_range)
        self.scaler = MinMaxScaler(clip=clip_range, maximum=max_range)
        self.torcher = ToTensor()

    def get_features(self, x):
        x_cropped = self.cropper(x)
        x_clipped = self.clipper(x_cropped)
        x_scaled = self.scaler(x_clipped)
        x_tensor = self.torcher(x_scaled)
        if isinstance(self.model, VAE):
            z, _ = model.encode(x_tensor)
        elif isinstance(self.model, AutoEncoder):
            z = model.encode(x_tensor)
        return z.detach().numpy().reshape(self.size)


###################### REMEMBER TO CHANGE RACHID'S CODE TO A PROPER CLASS HERE !!!!!!!!!!!!!!!!!!!!!
class RandomFeatures(FeaturesMixin):
    def __init__(self, **kwargs):
        super(RandomFeatures, self).__init__(**kwargs)

    def get_features(self, x):
        z = x
        return z


class Features(FeaturesMixin):
    def __init__(self, features_type, **kwargs):
        super(Features, self).__init__(**kwargs)
        self.features_type = features_type

        if self.features_type == 'grid':
            self.feat = GridFeatures(**kwargs)
        elif self.features_type == 'tile':
            self.feat = TileCodingFeatures(**kwargs)
        elif self.features_type == 'autoencoder':
            self.feat = AEFeatures(**kwargs)
        elif self.features_type == 'rand':
            self.feat = RandomFeatures(**kwargs)
        else:
            raise ValueError("discretizer type can either be grid or vae, but got {} instead.".format(self.discretize_type))

        self.size = self.feat.size

    def get_features(self, x):
        return self.feat.get_features(x)
