from email.policy import default
from matplotlib import testing
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from utils import torch_circular_gp, analysis
from model import *
from training import Trainer
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


### For first simulation experiments ###
class StochasticNeurons(torch.nn.Module):
    def __init__(
        self,
        N,
        num_ensemble=2,
        latent_dim=2,
        seed=304857,
        noise=False,
        tuning_width=10.0,
        scale=16.0,
    ):
        super(StochasticNeurons, self).__init__()
        self.num_ensemble = num_ensemble
        self.tuning_width = tuning_width
        self.scale = scale
        self.noise = noise
        self.latent_dim = latent_dim

        torch.manual_seed(seed)
        self.receptive_fields = torch.nn.Parameter(
            torch.rand(num_ensemble * N, latent_dim) * 2 * np.pi, requires_grad=False
        )
        ensemble_weights = np.zeros((N * num_ensemble, num_ensemble))
        for i in range(num_ensemble):
            ensemble_weights[i * N : (i + 1) * N, i] = 1
        self.ensemble_weights = torch.nn.Parameter(
            torch.tensor(ensemble_weights, dtype=torch.float), requires_grad=False
        )
        selector = torch.stack(
            [torch.eye(2 * latent_dim) for i in range(num_ensemble)], 0
        )
        self.selector = torch.nn.Parameter(selector, requires_grad=False)

    def forward(self, z):
        z_vector = angle2vector_flat(z)
        rf_vector = angle2vector_flat(self.receptive_fields)

        # early selection
        selector = self.ensemble_weights[..., None, None] * self.selector[None]
        selector = torch.concat(torch.split(selector, 1, dim=1), 3).view(
            -1, 2 * self.latent_dim, self.num_ensemble * 2 * self.latent_dim
        )
        selected = torch.matmul(selector, z_vector.T)
        dist = (rf_vector[..., None] - selected) ** 2
        pairs = sum_pairs(dist)
        if self.latent_dim == 2:
            pairs = sum_pairs(pairs)
        response = torch.exp(-pairs / self.tuning_width) * self.scale
        responses = response[:, 0]
        if self.noise:
            responses = torch.poisson(responses)
        responses = responses / self.scale

        return responses


def test_simulation():
    num_ensemble = 2
    num_neuron = 2
    model = StochasticNeurons(num_neuron, num_ensemble=num_ensemble)

    print("clean")
    plt.figure(figsize=(15, 3))
    for i in range(num_ensemble * 2):
        plt.subplot(1, num_ensemble * 2, i + 1)
        inputs = torch.zeros((100, num_ensemble * 2))
        inputs[:, i] = torch.linspace(0, 2 * np.pi, 100)
        responses = model(inputs)
        plt.plot(responses.detach().numpy().T)
        plt.legend(np.arange(num_neuron * num_ensemble))
    plt.show()

    print("noisy")
    model.noise = True
    plt.figure(figsize=(15, 3))
    for i in range(num_ensemble * 2):
        plt.subplot(1, num_ensemble * 2, i + 1)
        inputs = torch.zeros((100, num_ensemble * 2))
        inputs[:, i] = torch.linspace(0, 2 * np.pi, 100)
        responses = model(inputs)
        plt.plot(responses.detach().numpy().T)
        plt.legend(np.arange(num_neuron * num_ensemble))
    plt.show()

    # Persistence
    num_neuron = 50
    D = 500
    model = StochasticNeurons(num_neuron, num_ensemble=num_ensemble, noise=True)
    responses = model(torch.rand(D, num_ensemble * 2) * 2 * np.pi)

    # all
    t0 = time.time()
    VR = VietorisRipsPersistence(
        homology_dimensions=[0, 1, 2],
    )
    diagrams0 = VR.fit_transform([responses.detach().numpy().T])
    print(diagrams0.shape, time.time() - t0)
    fig0 = plot_diagram(diagrams0[0])
    fig0.show()

    # per ensemble
    for i in range(num_ensemble):
        t0 = time.time()
        VR = VietorisRipsPersistence(
            homology_dimensions=[0, 1, 2],
        )
        diagrams0 = VR.fit_transform(
            [responses[i * num_neuron : (i + 1) * num_neuron].detach().numpy().T]
        )
        print(i, diagrams0.shape, time.time() - t0)
        fig0 = plot_diagram(diagrams0[0])
        fig0.show()


def get_exp_name(args_old, args_new, ignoring=("writer")):
    """
    Returns a convenient experiment name for tensorboard that compares
    arguments given to argparse to the default settings. It then
    writes the arguments where the values differ from the
    default settings into the experiment name.
    """

    args_new = args_new.__dict__
    for key, val in args_new.items():
        if val == "false" or val == "False":
            args_new[key] = False
        if val == "true" or val == "True":
            args_new[key] = True

    exp_name = args_new["name"] + "_"
    for key in args_old:
        old_val = args_old[key]
        if old_val != args_new[key]:
            if key in ignoring:
                continue
            val = args_new[key]
            if isinstance(val, float):
                exp_name += f"{key[:15]}{val:.3f}-"
            elif isinstance(val, str):
                exp_name += f"{key[:15]}" + val[:5] + "-"
            else:
                exp_name += f"{key[:15]}" + str(val) + "-"

    return exp_name + f'--{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'


def parse_args(notebook=False):
    parser = argparse.ArgumentParser(description="Neural LVM Test Params")
    parser.add_argument("--weight_kl", type=float, help="VAE beta", default=1e-2)

    parser.add_argument("--name", default="", type=str, help="Add some nice name")
    parser.add_argument("--latent_style", default="hack", type=str, help="What VAE latent to use", choices=["hyper", "hack"])  # fmt: skip
    parser.add_argument("--num_sample", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_neuron_train", type=int, default=50)
    parser.add_argument("--num_neuron_test", type=int, default=50)
    parser.add_argument("--num_ensemble", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--z_smoothness", type=int, default=3)
    parser.add_argument("--num_test", type=int, default=10000)
    parser.add_argument("--feature_type", type=str, default="bump")
    args = parser.parse_args([]) if notebook else parser.parse_args()
    all_defaults = {key: parser.get_default(key) for key in vars(args)}
    return args, all_defaults


def main():
    args, default_args = parse_args()
    args.exp_name = get_exp_name(default_args, args)
    args.log_dir = f"./experiments/{args.exp_name}"

    writer = SummaryWriter(log_dir=args.log_dir)
    num_neuron = args.num_neuron_train + args.num_neuron_test
    neurons_train_ind = np.zeros(num_neuron * args.num_ensemble, dtype=bool)
    ind = np.random.choice(
        num_neuron * args.num_ensemble,
        args.num_neuron_train * args.num_ensemble,
        replace=False,
    )
    neurons_train_ind[ind] = True
    model = StochasticNeurons(
        num_neuron,
        num_ensemble=args.num_ensemble,
        noise=True,
        latent_dim=args.latent_dim,
    ).to(device)
    ensembler = LatentVariableModel(
        num_neuron_train=args.num_neuron_train * args.num_ensemble,
        num_neuron_test=args.num_neuron_test * args.num_ensemble,
        num_hidden=256,
        num_ensemble=args.num_ensemble,
        latent_dim=args.latent_dim,
        latent_style=args.latent_style,
        seed=args.seed,
        tuning_width=10.0,
        nonlinearity="exp",
        kernel_size=9,
        feature_type=args.feature_type,
    ).to(device)
    print("model", ensembler)
    print("number of trainable parameters in model:", (count_parameters(ensembler)))

    if args.z_smoothness > 0:  # gp latents
        z_train = torch_circular_gp(
            args.num_sample, args.latent_dim * args.num_ensemble, args.z_smoothness
        )
        z_test = torch_circular_gp(
            args.num_test, args.latent_dim * args.num_ensemble, args.z_smoothness
        )
    else:  # iid latents
        z_train = (
            torch.rand(args.num_sample, args.latent_dim * args.num_ensemble) * 2 * np.pi
        )
        z_test = (
            torch.rand(args.num_test, args.latent_dim * args.num_ensemble).to(device)
            * 2
            * np.pi
        )

    z_train = z_train.to(device)
    z_test = z_test.to(device)
    data_train = model(z_train).detach()
    model.noise = False
    data_test = model(z_test).detach()
    model.noise = True

    trainer = Trainer(
        model=ensembler,
        data_train=data_train.cpu().numpy(),
        data_test=data_test.cpu().numpy(),
        neurons_train_ind=neurons_train_ind,
        mode="full",
        z_train=None,
        z_test=None,
        num_steps=50000,
        batch_size=128,
        seed=args.seed+1,
        learning_rate=3e-3,
        weight_kl=args.weight_kl,
        log_dir=args.log_dir,
        writer=writer,
    )
    trainer.train()
    analysis(ensembler, model, trainer, z_test)
    print("Repeat analysis with good inference:")
    analysis(ensembler, model, trainer, z_test, do_inference=True)


if __name__ == "__main__":
    main()
