import numpy as np
import torch
from torch import optim

from functions import vae_loss_function, vqvae_loss_function, gumbel_loss_function
from autoencoders import VectorQuantizedVAE, VAE, GumbelVAE, AutoEncoder

ARGS = {'batch_size': 128,
        'epochs': 3,
        'device': 'cpu',
        'log_interval': 100,
        'data_distribution': 'bernoulli',
        'data_samples': 120000,
        'train_test_split': 0.3,
        'lr': 1e-5,
        'n_latents': 6,
        'latent_dim': 30,
        'categorical_dim': 5,
        'optimizer': 'adam',
        'reduction': 'mean',
        'features': 360}

def test(model,
         epoch,
         test_loader,
         args = None):
    if args is None:
        args = ARGS

    device = torch.device(args["device"])
    model.eval()
    test_loss = 0
    test_ELBO = 0
    test_log_likelihood = 0
    with torch.no_grad():
        for i, (data, noisy_data) in enumerate(test_loader):
            data = data.to(device)
            noisy_data = noisy_data.to(device)

            if isinstance(model, VectorQuantizedVAE):
                recon_data, z_e_x, z_q_x = model(noisy_data)
                loss, ELBO, log_likelihood = vqvae_loss_function(recon_data, data, z_e_x, z_q_x,
                                                                categorical_dim = args["categorical_dim"],
                                                                reduction = args["reduction"],
                                                                data_distribution = args["data_distribution"])
            elif isinstance(model, GumbelVAE):
                recon_batch, qy = model(noisy_data, args["temp_min"], hard=False)
                loss, ELBO, log_likelihood = gumbel_loss_function(recon_batch, data, qy,
                                                                categorical_dim = args["categorical_dim"],
                                                                reduction = args["reduction"],
                                                                data_distribution = args["data_distribution"])
            elif isinstance(model, VAE):
                recon_data, mu, logvar = model(noisy_data)
                loss, ELBO, log_likelihood = vae_loss_function(recon_data, data, mu, logvar,
                                                                reduction = args["reduction"],
                                                                data_distribution = args["data_distribution"])
            elif isinstance(model, AutoEncoder):
                recon_data = model(noisy_data)
                loss, ELBO, log_likelihood = vae_loss_function(recon_data, data,
                                                                reduction = args["reduction"],
                                                                data_distribution = args["data_distribution"])
            else:
                raise Exception("model either not imported or implemented.")

            test_loss += loss.item()
            test_ELBO += ELBO
            test_log_likelihood += log_likelihood

    n = args["batch_size"]/len(test_loader.dataset)
    test_loss *= n
    test_ELBO *= n
    test_log_likelihood *= n
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return (test_loss, test_ELBO, test_log_likelihood)


def train_epoch(model,
                epoch,
                train_loader,
                test_loader,
                args=None,
                anneal_rate=None,
                temp_start=None):
    if args is None:
        args = ARGS

    if args['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    else:
        raise Exception("{} optimizer not implemented.".format(args['optimizer']))

    train_loss_log = []
    train_elbo_log = []
    train_ll_log = []

    device = torch.device(args["device"])
    model.train()
    train_loss = 0
    for batch_idx, (data, noisy_data) in enumerate(train_loader):
        data = data.to(device)
        noisy_data = noisy_data.to(device)
        optimizer.zero_grad()

        if isinstance(model, VectorQuantizedVAE):
            recon_data, z_e_x, z_q_x = model(noisy_data)
            loss, ELBO, log_likelihood = vqvae_loss_function(recon_data, data, z_e_x, z_q_x,
                                                            categorical_dim = args["categorical_dim"],
                                                            reduction = args["reduction"],
                                                            data_distribution = args["data_distribution"])
        elif isinstance(model, GumbelVAE):
            t = np.ceil(len(train_loader.dataset)/(args["batch_size"]))*(epoch-1) + batch_idx
            temp = np.maximum(temp_start*np.exp(- anneal_rate * (t)), args["temp_min"])
            recon_batch, qy = model(noisy_data, temp, hard=False)
            loss, ELBO, log_likelihood = gumbel_loss_function(recon_batch, data, qy,
                                                            categorical_dim = args["categorical_dim"],
                                                            reduction = args["reduction"],
                                                            data_distribution = args["data_distribution"])
        elif isinstance(model, VAE):
            recon_data, mu, logvar = model(noisy_data)
            loss, ELBO, log_likelihood = vae_loss_function(recon_data, data, mu, logvar,
                                                            reduction = args["reduction"],
                                                            data_distribution = args["data_distribution"])
        elif isinstance(model, AutoEncoder):
            recon_data = model(noisy_data)
            loss, ELBO, log_likelihood = vae_loss_function(recon_data, data,
                                                            reduction = args["reduction"],
                                                            data_distribution = args["data_distribution"])
        else:
            raise Exception("model either not imported or implemented.")

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        train_loss_log.append(loss.item())
        train_elbo_log.append(ELBO)
        train_ll_log.append(log_likelihood)

        if batch_idx % args['log_interval']== 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

    if isinstance(model, GumbelVAE):
        t = np.ceil(len(train_loader.dataset)/(args["batch_size"]))*(epoch)
        temp = np.maximum(temp_start*np.exp(- anneal_rate * (t)), args["temp_min"])
        print("Temp: ", temp)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss * args["batch_size"] / len(train_loader.dataset)))

    test_loss, test_ELBO, test_log_likelihood = test(model,
                                                        epoch,
                                                        test_loader,
                                                        args)

    train_logs = (train_loss_log, train_elbo_log, train_ll_log)
    test_logs = (test_loss, test_ELBO, test_log_likelihood)

    model.eval()
    return (model, train_logs, test_logs)


def trainer(model,
            epoch,
            train_loader,
            test_loader,
            args=None):
    if args is None:
        args = ARGS

    train_loss_log = []
    train_elbo_log = []
    train_ll_log = []
    test_loss_log = []
    test_elbo_log = []
    test_ll_log = []

    test_loss, test_ELBO, test_log_likelihood = test(model,
                                                        0,
                                                        test_loader,
                                                        args)

    test_loss_log.append(test_loss)
    test_elbo_log.append(test_ELBO)
    test_ll_log.append(test_log_likelihood)

    # Calculate anneal rate for GumbelVAE
    n = np.ceil(len(train_loader.dataset)/args["batch_size"]) * args['epochs']
    anneal_rate = (np.log(args["temp_start"])-np.log(args["temp_min"] * 0.9))/(n)
    if isinstance(model, GumbelVAE):
        print("Anneal rate set to: ", anneal_rate)
    for epoch in range(1, args['epochs'] + 1):
        model, train_logs, test_logs = train_epoch(model,
                                                    epoch,
                                                    train_loader,
                                                    test_loader,
                                                    args,
                                                    anneal_rate,
                                                    args["temp_start"])
        train_loss_log += train_logs[0]
        train_elbo_log += train_logs[1]
        train_ll_log += train_logs[2]
        test_loss_log.append(test_logs[0])
        test_elbo_log.append(test_logs[1])
        test_ll_log.append(test_logs[2])

    train_logs = (train_loss_log, train_elbo_log, train_ll_log)
    test_logs = (test_loss_log, test_elbo_log, test_ll_log)
    return (model, train_logs, test_logs)
