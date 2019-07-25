import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd import Function


def vae_loss_function(x_tilde, x, mu=None, logvar=None, reduction='sum',
                        data_distribution = 'bernoulli'):
    if data_distribution == 'bernoulli':
        log_likelihood = - F.binary_cross_entropy(x_tilde, x, reduction=reduction)
    else:
        log_likelihood = - F.mse_loss(x_tilde, x, reduction=reduction)

    if mu is None:
        mu = torch.zeros_like(x)
    if logvar is None:
        logvar = torch.zeros_like(x)

    if reduction == 'sum':
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    ELBO = log_likelihood - KLD
    return - ELBO, ELBO.item(), log_likelihood.item()


def gumbel_loss_function(x_tilde, x, qy, categorical_dim = 5, reduction='sum',
                            data_distribution = 'bernoulli'):
    if data_distribution == 'bernoulli':
        log_likelihood = - F.binary_cross_entropy(x_tilde, x, reduction=reduction)
    else:
        log_likelihood = - F.mse_loss(x_tilde, x, reduction=reduction)

    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    if reduction == 'sum':
        KLD = torch.sum(qy * log_ratio, dim=-1).mean()
    else:
        KLD = torch.mean(qy * log_ratio, dim=-1).mean()
    ELBO = log_likelihood - KLD
    return - ELBO, ELBO.item(), log_likelihood.item()


def vqvae_elbo(x_tilde, x, categorical_dim = 1.0, reduction='sum',
                data_distribution = 'bernoulli'):
    if data_distribution == 'bernoulli':
        log_likelihood = - F.binary_cross_entropy(x_tilde, x, reduction=reduction)
    else:
        log_likelihood = - F.mse_loss(x_tilde, x, reduction=reduction)

    # if reduction=='sum':
    #     KLD = np.log(categorical_dim)*x.size(0)
    # else:
    #     KLD = np.log(categorical_dim)
    ELBO = log_likelihood #- KLD
    return ELBO, log_likelihood


def vqvae_loss_function(x_tilde, x,
                        z_e_x, z_q_x,
                        beta=1.0,
                        categorical_dim=1.0,
                        reduction='sum',
                        data_distribution = 'bernoulli'
                        ):
    # negative ELBO loss (equal to reconstruction loss if KL = 0 when K=1.0)
    ELBO, log_likelihood = vqvae_elbo(x_tilde, x,
                                     categorical_dim,
                                     reduction,
                                     data_distribution)
    # Vector quantization loss
    loss_vq = F.mse_loss(z_q_x, z_e_x.detach(), reduction=reduction)
    # Commitment loss
    loss_commit = F.mse_loss(z_e_x, z_q_x.detach(), reduction=reduction)
    loss = loss_vq + beta*loss_commit - ELBO
    return loss, ELBO.item(), log_likelihood.item()


# based on implementation of Yongfei Yan :
# https://github.com/YongfeiYan/Gumbel_Softmax_VAE
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.shape)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits,
                   temperature = 0.5,
                   hard=False,
                   latent_dim = 6,
                   categorical_dim = 5):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)


# based on implementation of Rithesh Kumar et al :
# https://github.com/ritheshkumar95/pytorch-vqvae
class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

vq = VectorQuantization.apply
class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)
