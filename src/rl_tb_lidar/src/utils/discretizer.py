import numpy as np

from misc import Cropper, ClipRange, MinMaxScaler
from utils.sensormodel.sensormodel import truncnorm_rvs_recursive

try:
    import torch
    from autoencoders.autoencoders import VectorQuantizedVAE, GumbelVAE
    from misc import ToTensor
except ImportError:
    pass

MAX_RANGE = 16.0
MIN_RANGE = 0.3
CLIP_RANGE = 2.0
LEVELS = 4
SIZE = 4


class DiscretizerMixin(object):
    def __init__(self,
                 levels=LEVELS,
                 size=SIZE,
                 enumerate=True,
                 crop=None,
                 **kwargs):
        self.levels = levels
        self.size = size
        self.enumerate = enumerate
        self.cropper = Cropper(crop)

    def __call__(self, x):
        z = self.discretize(x)
        if self.enumerate:
            z = self.enumerate_discretization(z)
        return z

    def enumerate_discretization(self, z):
        z_enumerated = 0
        for i, level in enumerate(list(z)):
            z_enumerated = z_enumerated + level*(self.levels**i)
        return int(z_enumerated)

    def sample(self):
        z_sample = np.random.randint(self.levels, size=self.size)
        if self.enumerate:
            z_sample = self.enumerate_discretization(z_sample)
        return z_sample


class GridDiscretizer(DiscretizerMixin):
    def __init__(self,
                 randomize_bins=False,
                 levels=LEVELS,
                 size=SIZE,
                 enumerate=True,
                 crop=None,
                 min_range=MIN_RANGE,
                 max_range=MAX_RANGE,
                 clip_range=CLIP_RANGE,
                 **kwargs):
        super(GridDiscretizer, self).__init__(levels=levels,
                                              size=size,
                                              enumerate=enumerate,
                                              crop=crop)
        self.randomize_bins = randomize_bins
        min_range = min_range
        max_range = min(clip_range, max_range)

        start = np.log10(min_range)
        stop = np.log10(max_range)

        bins = list(np.logspace(start, stop, num=self.levels-1, endpoint=True))
        self.bins = np.array(kwargs.setdefault('bins', bins))
        if self.randomize_bins:
            self.bins = [self.bins]
            r = bins[1]-bins[0]
            for i in range(self.size - 1):
                rand_shift = truncnorm_rvs_recursive(np.zeros(1),
                                                     sigma = 2*r*0.33,
                                                     lower_clip = -r,
                                                     upper_clip = r)[0]
                self.bins.append( np.array(bins) + rand_shift )

    def discretize(self, x):
        x_cropped = self.cropper(x)
        x_splitted = np.split(x_cropped, self.size)
        min_x_splitted = np.min(x_splitted, axis=1)
        if self.randomize_bins:
            z = [np.digitize(min_x_splitted[i], self.bins[i]) for i in range(self.size)]
            z = np.asarray(z).astype('int')
        else:
            z = np.digitize(min_x_splitted, self.bins)
        return z


class VAEDiscretizer(DiscretizerMixin):
    def __init__(self,
                 path,
                 levels=LEVELS,
                 size=SIZE,
                 enumerate=True,
                 crop=None,
                 min_range=MIN_RANGE,
                 max_range=MAX_RANGE,
                 clip_range=CLIP_RANGE,
                 **kwargs):
        super(VAEDiscretizer, self).__init__(levels=levels,
                                             size=size,
                                             enumerate=enumerate,
                                             crop=crop)
        self.model = torch.load(path, map_location=lambda storage, loc: storage).eval()

        levels = self.model.categorical_dim
        if isinstance(self.model, VectorQuantizedVAE):
            size = self.model.n_latents
        elif isinstance(self.model, GumbelVAE):
            size = self.model.latent_dim
        else:
            raise ValueError("Model can only be either VQ-VAE or Gumbel-VAE.")

        assert self.size == size, "Loaded model latent size ({}) not consistent with discretization size ({})".format(size, self.size)
        assert self.levels == levels, "Loaded model latent levels ({}) not consistent with discretization levels ({})".format(levels, self.levels)
        assert int(self.cropper.size*2.0) == self.model.input_dim, "Loaded model input dimension ({}) not consistent with discretization input dimension ({})".format(self.model.input_dim, int(self.cropper.size*2.0))

        self.clipper = ClipRange(clip=CLIP_RANGE, maximum=MAX_RANGE)
        self.scaler = MinMaxScaler(clip=CLIP_RANGE, maximum=MAX_RANGE)
        self.torcher = ToTensor()

    def discretize(self, x):
        x_cropped = self.cropper(x)
        x_clipped = self.clipper(x_cropped)
        x_scaled = self.scaler(x_clipped)
        x_tensor = self.torcher(x_scaled)
        z = self.model.encode(x_tensor)
        return z.detach().numpy().reshape(self.size)


class Discretizer(DiscretizerMixin):
    def __init__(self, discretize_type, **kwargs):
        super(Discretizer, self).__init__(**kwargs)
        self.discretize_type = discretize_type

        if self.discretize_type == 'grid':
            self.disc = GridDiscretizer(**kwargs)
        elif self.discretize_type == 'autoencoder':
            self.disc = VAEDiscretizer(**kwargs)
        else:
            raise ValueError("discretizer type can either be grid or vae, but got {} instead.".format(self.discretize_type))

    def discretize(self, x):
        return self.disc.discretize(x)
