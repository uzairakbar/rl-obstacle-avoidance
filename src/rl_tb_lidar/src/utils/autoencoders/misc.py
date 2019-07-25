import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.interpolate as interp
import seaborn as sns
import umap

import torch
from torch import optim

from functions import vae_loss_function, vqvae_loss_function, gumbel_loss_function
from autoencoders import VAE, DiscreteLatentVAE, GumbelVAE, VectorQuantizedVAE, AutoEncoder
from utils import enumerate_discrete_latent

ARGS = {'batch_size': 128,
        'epochs': 3,
        'device': 'cpu',
        'log_interval': 100,
        'data_distribution': 'bernoulli',
        'data_samples': 120000,
        'train_test_split': 0.3,
        'lr': 1e-4,
        'n_latents': 6,
        'latent_dim': 30,
        'categorical_dim': 5,
        'optimizer': 'adam',
        'reduction': 'mean',
        'features': 360}

def find_rectangle(area):
    b, l = 0, 0
    M = np.int(np.ceil(np.sqrt(area)))
    ans = 0
    for i in range(M, 0, -1):
        if area%i == 0:
            l = area//i
            b = i
            break
    return b, l

def test_random_sample(model, test_loader, args):
    model.eval()
    org, noisy_org = iter(test_loader).next()
    org, noisy_org = org[0].view(1, 1, args["features"]), noisy_org[0].view(1, 1, args["features"])
    if isinstance(model, VectorQuantizedVAE):
        recon, _, _ = model(noisy_org)
    elif isinstance(model, GumbelVAE):
        recon, _ = model(noisy_org)
    elif isinstance(model, VAE):
        recon, _, _ = model(noisy_org)
    elif isinstance(model, AutoEncoder):
        recon = model(noisy_org)

    reshapeD1, reshapeD2 = find_rectangle(args["features"])
    # reshapeD1 = int(np.sqrt(args["features"]))
    # reshapeD2 = int(args["features"]/reshapeD1)
    org_reshaped, noisy_org_reshaped, recon_reshaped = (org.numpy()[0][0].reshape(reshapeD1, reshapeD2),
                                    noisy_org.numpy()[0][0].reshape(reshapeD1, reshapeD2),
                                    recon.detach().numpy()[0][0].reshape(reshapeD1, reshapeD2))
    if isinstance(model, DiscreteLatentVAE):
        print("Discrete Label: ", model.encode(noisy_org, enumerate_labels=True)[0])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(org_reshaped)
    ax2.imshow(noisy_org_reshaped)
    ax3.imshow(recon_reshaped)
    return fig, (ax1, ax2, ax3)

def plot_loss_graphs(logs, args, train=True, test=True, fig=None, ax=None):
    train_logs, test_logs = logs
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    x = np.linspace(0, args["epochs"], len(train_logs[0]))
    for i, axi in enumerate(ax):
        if train:
            axi.plot(x, train_logs[i])
        test_log = np.asarray(test_logs[i])
        test_log_interp = interp.interp1d(np.arange(test_log.size), test_log)
        test_log_stretch = test_log_interp(np.linspace(0, test_log.size-1, x.size))
        if test:
            axi.plot(x, test_log_stretch)
        # n_test = int(x.size*(1.0-1.0/args["epochs"]))
        # test_log_stretch = test_log_interp(np.linspace(0, test_log.size-1, n_test))
        # if test:
        #     axi.plot(x[x.size - n_test:], test_log_stretch)
        axi.grid()
    return fig, ax


# def visualize_latents(model, test_dataset, args):
#     model.eval()
#     continuous_code_list = []
#     discrete_code_list = []
#     plt_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle=True)
#     for i in range(len(plt_dataloader.dataset)):
#         org, noisy_org = plt_dataloader.dataset.__getitem__(i)
#         org, noisy_org = org[0].view(1, 1, 360), noisy_org[0].view(1, 1, args["features"])
#         if isinstance(model, VectorQuantizedVAE):
#             _, con_code, _ = model(noisy_org)
#         elif isinstance(model, GumbelVAE):
#             _, con_code = model(noisy_org)
#         elif isinstance(model, VAE):
#             con_code, _ = model.encode(noisy_org)
#         elif isinstance(model, AutoEncoder):
#             con_code = model.encode(noisy_org)
#         else:
#             raise ValueError("specified model is not implemented.")
#         continuous_code_list.append(con_code.detach().numpy().reshape(args["n_latents"]*args["latent_dim"]))
#         if isinstance(model, DiscreteLatentVAE):
#             disc_code = model.encode(noisy_org, enumerate_labels=True)[0]
#             discrete_code_list.append(disc_code)
#
#     visualizer = umap.UMAP(min_dist=0.0)
#     embedding = visualizer.fit_transform(np.asarray(continuous_code_list))
#
#
#     fig, ax = plt.subplots()
#     if isinstance(model, DiscreteLatentVAE):
#         unique_codes = np.unique(np.asarray(discrete_code_list))
#         num_unique_codes = len(unique_codes)
#         colors = cm.rainbow(np.linspace(0, 1, len(unique_codes)))
#
#         code_count = np.zeros(num_unique_codes)
#
#         for i, code in enumerate(unique_codes):
#             idx = np.where((np.asarray(discrete_code_list) == code))
#             code_embedding = embedding[idx]
#             code_count[i] = code_embedding.shape[0]
#             ax.scatter(code_embedding[:, 0], code_embedding[:, 1], c=[colors[i]], s = 1);
#         print("num codes used: {}/{}".format(num_unique_codes,
#             enumerate_discrete_latent(np.ones(args["n_latents"])*(args["categorical_dim"]-1),
#                                         categorical_dim = args["categorical_dim"])))
#         ax.grid()
#         return fig, ax
#     else:
#         ax.scatter(embedding[:, 0], embedding[:, 1], s = 1);
#         ax.grid()
#         return fig, ax



def visualize_latents(model, test_dataset, args):
    model.eval()
    continuous_code_list = []
    discrete_code_list = []
    plt_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle=True)
    for i in range(len(plt_dataloader.dataset)):
        org, noisy_org = plt_dataloader.dataset.__getitem__(i)
        org, noisy_org = org[0].view(1, 1, args["features"]), noisy_org[0].view(1, 1, args["features"])
        if isinstance(model, VectorQuantizedVAE):
            _, con_code, _ = model(noisy_org)
            con_code = con_code.detach().numpy().reshape(args["n_latents"]*args["latent_dim"])
        elif isinstance(model, GumbelVAE):
            _, con_code = model(noisy_org)
            con_code = con_code.detach().numpy().reshape(args["categorical_dim"]*args["n_latents"])
        elif isinstance(model, VAE):
            con_code, _ = model.encode(noisy_org)
            con_code = con_code.detach().numpy().reshape(args["latent_dim"])
        elif isinstance(model, AutoEncoder):
            con_code = model.encode(noisy_org)
            con_code = con_code.detach().numpy().reshape(args["latent_dim"])
        else:
            raise ValueError("specified model is not implemented.")
        continuous_code_list.append(con_code)
        if isinstance(model, DiscreteLatentVAE):
            disc_code = model.encode(noisy_org, enumerate_labels=True)[0]
            discrete_code_list.append(disc_code)

    print("matrix dimensions: ", np.asmatrix(continuous_code_list).shape)
    print("rank: ", np.linalg.matrix_rank(np.asmatrix(continuous_code_list)))
    visualizer = umap.UMAP(min_dist=0.0)
    embedding = visualizer.fit_transform(np.asarray(continuous_code_list))

    if isinstance(model, DiscreteLatentVAE):
        unique_codes = np.unique(np.asarray(discrete_code_list))
        num_unique_codes = len(unique_codes)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_codes)))

        code_count = np.zeros(num_unique_codes)
        # if isinstance(model, GumbelVAE):
        total_codes = enumerate_discrete_latent(np.ones(args["n_latents"])*(args["categorical_dim"]-1),
                                    categorical_dim = args["categorical_dim"])
        # elif isinstance(model, VectorQuantizedVAE):
        #     total_codes = enumerate_discrete_latent(np.ones(args["n_latents"])*(args["categorical_dim"]-1),
        #                                 categorical_dim = args["categorical_dim"])
        print("num codes used: {}/{} ({}%)".format(num_unique_codes,
                            total_codes, (num_unique_codes/total_codes)*100))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        for i, code in enumerate(unique_codes):
            idx = np.where((np.asarray(discrete_code_list) == code))
            code_embedding = embedding[idx]
            code_count[i] = code_embedding.shape[0]
            ax1.scatter(code_embedding[:, 0], code_embedding[:, 1], c=[colors[i]], s = 1);
        ax1.grid()

        ax2.bar(np.arange(num_unique_codes), code_count, color=colors)
        ax2.set_yscale('log')
        ax2.grid()
        return fig, (ax1, ax2)
    else:
        fig, ax = plt.subplots()
        ax.scatter(embedding[:, 0], embedding[:, 1], s = 1);
        ax.grid()
        return fig, ax
