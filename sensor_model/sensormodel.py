import numpy as np
from scipy import stats

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


class Likelihood(object):
    def __init__(self, theta, maximum=MAX_RANGE, minimum=0.0):
        self.maximum = maximum
        self.minimum = minimum
        self.theta = theta

    def P(self, z, z_exp):
        sigma_hit, lamda_short = self.theta[-2:]
        z_params = np.asarray(self.theta[:-2])
        unnorm_likelihood = np.asarray([Hit(sigma=sigma_hit).P(z, z_exp),
                                       Short(lamda=lamda_short).P(z, z_exp),
                                       Max().P(z, z_exp),
                                       Rand().P(z, z_exp)])
        likelihood = np.matmul(z_params.T, unnorm_likelihood)
        return likelihood


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
