from re import L
import torch
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms

import json
from utils.utils import cprint

from model.probability import depth_categorical_VI
from model.resnet_dun import ArchUncertResNet
from model.training_wrappers import DUN_VI
from config.dun_minist_config import get_args

transform = transforms.ToTensor()

use_gpu = torch.cuda.is_available()
args = get_args()
kwargs = vars(args)
print("input args:\n", json.dumps(kwargs, indent=4, separators=(",", ":")))

train_data = {
    "mnist": datasets.MNIST(root="data/", train=True, download=True, transform=transform),
    "fashion_mnist": datasets.FashionMNIST(root="data/", train=True, download=True, transform=transform)
}
test_data = {
    "mnist": datasets.MNIST(root="data/", train=False, download=True, transform=transform),
    "fashion_mnist": datasets.FashionMNIST(root="data/", train=False, download=True, transform=transform)
}

def train_model(net, train_loader):
    
    marginal_loglike = 0
    err_train = 0
    train_loss = 0
    nb_samples = len(train_loader.dataset)

    for i, (x, y) in enumerate(train_loader):
        marg_loglike_estimate, minus_loglike, err = net.fit(x, y)

        marginal_loglike += marg_loglike_estimate * x.shape[0]
        err_train += err * x.shape[0]
        train_loss += minus_loglike * x.shape[0]
        nb_samples += len(x)

        if args.debug and  (i+1) % 2 == 0:
            print(
                "Iter", i+1, 
                "marg_loglike_estimate", marg_loglike_estimate * x.shape[0], 
                "err", err * x.shape[0], 
                "train_loss", minus_loglike * x.shape[0]
            )
    
    return marginal_loglike/nb_samples, err_train/nb_samples, train_loss/nb_samples


def evaluate(net, val_loader):
    nb_samples = len(val_loader.dataset)
    dev_loss = 0
    err_dev = 0

    for x, y in val_loader:
        minus_loglike, err = net.eval(x, y)

        dev_loss += minus_loglike * x.shape[0]
        err_dev += err * x.shape[0]

    return dev_loss/nb_samples, err_dev/nb_samples

def main():
    print(f'loading dataset <{args.dataset}>...')
    train_dataset = train_data[args.dataset] 
    test_dataset = test_data[args.dataset]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               num_workers=0,
                                               drop_last=True,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128, 
                                              num_workers=0,
                                              shuffle=False)
    
    print('building model...')

    start_depth = 1
    end_depth = 13
    num_classes = 10
    initial_conv = "1x3"
    input_chanels = 1
    n_layers = end_depth - start_depth

    prior_probs = [1 / (n_layers)] * (n_layers)
    prob_model = depth_categorical_VI(prior_probs, cuda=use_gpu)

    model = ArchUncertResNet(layers=[3, 4, 6, 3], start_depth=start_depth, end_depth=end_depth, 
                        num_classes=num_classes,
                        zero_init_residual=True, initial_conv=initial_conv, concat_pool=False,
                        input_chanels=input_chanels, p_drop=0)

    N_train = len(train_loader.dataset)
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    milestones = [40, 70]

    net = DUN_VI(model, prob_model, N_train, lr=lr, momentum=momentum, weight_decay=weight_decay, cuda=use_gpu,
                 schedule=milestones, regression=False, pred_sig=None)

    print('training...')
    for epoch in range(1, args.n_epoch+1):  
        marginal_loglike, err_train, train_loss = train_model(net, train_loader)
        net.update_lr()

        print('\n depth approx posterior', net.prob_model.current_posterior.data.cpu().numpy())
        print(
            f"it {epoch}/{args.n_epoch}, "
            f"ELBO/evidence {marginal_loglike:.4f}, "
            f"pred minus loglike = {train_loss:.4f}, "
            f"err = {err_train:.4f}"
        )
        
        dev_loss, err_dev = evaluate(net, test_loader)
        cprint('g', f'     pred minus loglike = {dev_loss}, err = {err_dev}\n', end="")

if __name__ == '__main__':
    main()