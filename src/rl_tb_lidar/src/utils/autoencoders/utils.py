import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from scipy import stats

CLIP_RANGE = 2.0
MAX_RANGE = 16.0

def truncnorm_rvs_recursive(x, sigma, lower_clip=0.0, upper_clip=MAX_RANGE):
    q = np.random.normal(x, sigma, size=x.size)
    c1 = q < lower_clip
    c2 = q > upper_clip
    c = np.logical_or(c1, c2)
    if np.any(c):
        q[c] = truncnorm_rvs_recursive(x[c], sigma, lower_clip, upper_clip)
    return q

def truncexpon_rvs_recursive(lamda, upper_clip):
    q = np.random.exponential(lamda, size=upper_clip.size)
    c = q > upper_clip
    if np.any(c):
        q[c] = truncexpon_rvs_recursive(lamda, upper_clip[c])
    return q

class Hit(object):
    def __init__(self, sigma=1.0, maximum=MAX_RANGE, minimum=0.0):
        self.sigma = sigma
        self.maximum = maximum
        self.minimum = minimum

    def P(self, z, z_exp):
        likelihood = stats.truncnorm.pdf(z, a=(self.minimum - z_exp)/self.sigma, b=(self.maximum - z_exp)/self.sigma, loc=z_exp, scale=self.sigma)
        return likelihood

    def sample(self, z_exp):
        c = z_exp > self.maximum
        if np.any(c):
            z_exp[c] = self.maximum
        # sample = stats.truncnorm.rvs(a=(self.minimum - z_exp) / self.sigma, b=(self.maximum - z_exp)/self.sigma, loc=z_exp, scale=self.sigma)
        sample = truncnorm_rvs_recursive(z_exp, self.sigma, lower_clip=self.minimum, upper_clip=self.maximum)
        return sample

class Short(object):
    def __init__(self, lamda=1.0, minimum=0.0, maximum=MAX_RANGE):
        self.lamda = lamda
        self.minimum = minimum
        self.maximum = maximum

    def P(self, z, z_exp):
        if z_exp > self.maximum:
            z_exp = self.maximum
        likelihood = stats.truncexpon.pdf(z, b=(z_exp-self.minimum)/self.lamda, loc=self.minimum, scale=self.lamda)
        return likelihood

    def sample(self, z_exp):
        c = z_exp > self.maximum
        if np.any(c):
            z_exp[c] = self.maximum
        # sample = stats.truncexpon.rvs(b=(z_exp-self.minimum)/self.lamda, loc=self.minimum, scale=self.lamda)
        sample = truncexpon_rvs_recursive(self.lamda, z_exp)
        return sample

class Max(object):
    def __init__(self, small=MAX_RANGE/1000.0, maximum=MAX_RANGE, minimum = 0.0):
        self.small = small
        self.maximum = maximum
        self.minimum = minimum

    def P(self, z, z_exp):
        if (z < self.maximum - self.small) or (z > self.maximum):
            return 0.0
        else:
            return 1.0/self.small

    def sample(self, z_exp):
        return self.maximum + self.small*(np.random.rand(z_exp.size) - 1)

class Rand(object):
    def __init__(self, maximum=MAX_RANGE, minimum=0.0):
        self.maximum = maximum
        self.minimum = minimum

    def P(self, z, z_exp):
        if (z > self.maximum) or (z < self.minimum):
            return 0.0
        return 1.0/self.maximum

    def sample(self, z_exp):
        return self.maximum*np.random.rand(z_exp.size)

# class SampleLIDAR(object):
#     """Sample from LIDAR sensor model distribution."""
#     def __init__(self, theta):
#         self.distribution_components = [Hit(sigma=theta[-2]),
#                                         Short(lamda=theta[-1]),
#                                         Max(),
#                                         Rand()]
#         self.z = np.asarray(theta[:-2])
#
#     def sample(self, ground_truth):
#         sample_component = np.random.choice(np.arange(4), p=self.z)
#         lidar_sample = self.distribution_components[sample_component].sample(ground_truth)
#         return lidar_sample

class SampleLIDAR(object):
    """Sample from LIDAR sensor model distribution."""
    def __init__(self, theta):
        self.hit = Hit(sigma=theta[-2])
        self.short = Short(lamda=theta[-1])
        self.max_ = Max()
        self.rand = Rand()
        self.z = np.asarray(theta[:-2])

    def sample(self, ground_truth):
        sample_component = np.random.choice(np.arange(4), size=ground_truth.size, p=self.z)
        lidar_sample = ground_truth.astype(float).copy()
        lidar_sample[sample_component == 0] = self.hit.sample(lidar_sample[sample_component == 0])
        lidar_sample[sample_component == 1] = self.short.sample(lidar_sample[sample_component == 1])
        lidar_sample[sample_component == 2] = self.max_.sample(lidar_sample[sample_component == 2])
        lidar_sample[sample_component == 3] = self.rand.sample(lidar_sample[sample_component == 3])
        return lidar_sample

class SensorNoise(object):
    """Add Sensor Noise."""
    # for 5  : [0.68989736, 0.01458186, 0.10597876, 0.18954202, 0.00444115, 0.55962999]
    # for 16 : [0.60652173, 0.02213939, 0.16294939, 0.20838950, 0.01062514, 0.52509905]
    # gt2 16 : [0.51408339, 0.00485265, 0.35703199, 0.12403196, 0.00870781, 0.48841981]
    # gt1 16 : [0.52078928, 0.03954691, 0.35703199, 0.08263181, 0.26700981, 0.67121838]
    def __init__(self, theta=[0.51408339, 0.00485265, 0.35703199, 0.12403196, 0.00870781, 0.48841981]):
        # self.sampler = np.vectorize(SampleLIDAR(theta).sample)
        self.sampler = SampleLIDAR(theta)

    def __call__(self, sample):
        noisy_sample = self.sampler.sample(sample)
        # noisy_sample = np.random.normal(sample, 0.05)
        # noisy_sample[noisy_sample < 0.0] = 0.0
        # noisy_sample[noisy_sample > 5.0] = 5.0
        return noisy_sample

class RandomFlip(object):
    """Randomly Flip Array."""
    def __call__(self, sample):
        if np.random.rand() < 0.5:
            return np.flip(sample)
        else:
            return sample

class RandomRoll(object):
    """Randomly Roll Array."""
    def __init__(self, low=0, high=180):
        self.low = low
        self.high = high+1
    def __call__(self, sample):
        roll = np.random.randint(self.low, self.high)
        return np.roll(sample, roll)

class CropFeatures(object):
    """Crop middle section of array."""
    def __init__(self, features=360):
        self.n = int((360 - features)/2)
    def __call__(self, sample, features=360):
        if self.n > 0:
            sample = sample[self.n:-self.n]
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
    def __call__(self, sample):
        sample = sample/min(MAX_RANGE, CLIP_RANGE)
        return sample

class StandardScaler(object):
    """Zero mean, unit variance."""
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
    def __call__(self, sample, mu, std):
        zero_mean_sample = sample - self.mu
        scaled_sample = np.divide(zero_mean_sample, self.std)
        return scaled_sample

class ToTensor(object):
    """Convert array into Tensors."""
    def __call__(self, sample):
        size = len(sample)
        torch_sample = torch.from_numpy(sample.copy())
        return torch_sample.view(1, size).float()

class LIDARDataset(Dataset):
    'LIDAR dataset for the turtlebot'
    def __init__(self,
                 csv_path=None,
                 data=None,
                 transform=None,
                 sample=1.0,
                 features=360,
                 data_distribution='bernoulli'):
        'Initialization'
        super(LIDARDataset, self).__init__()
        if data is None:
            data = pd.read_csv(csv_path, index_col=False)

        if sample <= 1.0:
            sample = sample * len(data)
        sample = int(sample)
        self.data = data.dropna().sample(n=sample)

        self.mu = data.mean(axis = 0).values
        self.std = data.std(axis = 0).values

        self.data_distribution = data_distribution
        if self.data_distribution == 'bernoulli':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler(self.mu, self.std)

        self.sampler = SensorNoise()
        self.clipper = ClipRange()
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        sample = self.data.iloc[index, :].values

        # sample = RandomFlip()(sample)
        # sample = RandomRoll()(sample, high=int(self.features/2))
        # sample = CropFeatures()(sample, features=self.features)
        #
        # noisy_sample = SensorNoise()(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        # generate noisy data based on LIDAR model
        noisy_sample = self.sampler(sample)
        # clip
        sample = self.clipper(sample)
        noisy_sample = self.clipper(noisy_sample)
        # scale
        sample = self.scaler(sample)
        noisy_sample = self.scaler(noisy_sample)
        # convert to tensor
        sample = ToTensor()(sample)
        noisy_sample = ToTensor()(noisy_sample)

        return sample, noisy_sample

def get_last_half(n):
    j = 0
    last_half = n
    for i in range(n):
        if last_half%2 == 0:
            j+=1
            last_half = last_half/2
        else:
            break
    return j, last_half

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def enumerate_discrete_latent(discrete_latent,
                              categorical_dim):
    enumeration = 0
    latent = list(discrete_latent)
    for i, feature in enumerate(latent):
        enumeration = enumeration + feature*(categorical_dim**i)
    return int(enumeration)

def enumerate_discrete_latents(discrete_latents,
                              categorical_dim):
    enumeration = torch.zeros(discrete_latents.size(0), dtype=torch.int)
    latents = list(discrete_latents)
    for i, latent in enumerate(latents):
        enumeration[i] = enumerate_discrete_latent(latent, categorical_dim)
    return enumeration
