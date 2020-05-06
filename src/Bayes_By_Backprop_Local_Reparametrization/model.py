from src.priors import isotropic_gauss_prior
from src.priors import *
from src.base_net import *
from src.Bayes_By_Backprop.model import w_to_std, sample_weights, BBP_Bayes_Net
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np

def KLD_cost(mu_p, sig_p, mu_q, sig_q):
    """KLD between two gaussian distribution

    Parameters
    ----------
    mu_p : :obj:`torch.tensor`
    sig_p : :obj:`torch.tensor`
    mu_q : :obj:`torch.tensor`
    sig_q : :obj:`torch.tensor`

    Returns
    -------
    :obj:`torch.tensor`
        KL divergence
    """
    KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2)
                 + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    # https://arxiv.org/abs/1312.6114 
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return KLD


class BayesLinear_local_reparam(nn.Module):
    """Linear Layer where activations are sampled from a fully factorised normal which is given by aggregating
     the moments of each weight's normal distribution. The KL divergence is obtained in closed form. Only works
      with gaussian priors.
    """
    def __init__(self, n_in, n_out, prior_sig):
        super(BayesLinear_local_reparam, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior_sig = prior_sig

        # Learnable parameters
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.1, 0.1))
        self.W_p = nn.Parameter(
            torch.Tensor(self.n_in, self.n_out).uniform_(-3, -2))

        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-3, -2))

    def forward(self, X, sample=False):
        #         print(self.training)

        if not self.training and not sample:  # This is just a placeholder function
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, 0, 0

        else:
            # calculate std
            std_w = w_to_std(self.W_p)
            std_b = w_to_std(self.b_p)

            act_W_mu = torch.mm(X, self.W_mu)  # self.W_mu + std_w * eps_W
            act_W_std = torch.sqrt(torch.mm(X.pow(2), std_w.pow(2)))

            # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
            # the same random sample is used for every element in the minibatch output
            eps_W = Variable(self.W_mu.data.new(act_W_std.size()).normal_(mean=0, std=1))
            eps_b = Variable(self.b_mu.data.new(std_b.size()).normal_(mean=0, std=1))

            act_W_out = act_W_mu + act_W_std * eps_W  # (batch_size, n_output)
            act_b_out = self.b_mu + std_b * eps_b

            output = act_W_out + act_b_out.unsqueeze(0).expand(X.shape[0], -1)

            kld = KLD_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.W_mu, sig_q=std_w) \
                + KLD_cost(mu_p=0, sig_p=0.1, mu_q=self.b_mu, sig_q=std_b)
            return output, kld, 0


class bayes_linear_LR_2L(nn.Module):
    def __init__(self, input_dim, output_dim, nhid, prior_sig):
        super(bayes_linear_LR_2L, self).__init__()

        n_hid = nhid
        self.prior_sig = prior_sig

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.bfc1 = BayesLinear_local_reparam(input_dim, n_hid, self.prior_sig)
        self.bfc2 = BayesLinear_local_reparam(n_hid, n_hid, self.prior_sig)
        self.bfc3 = BayesLinear_local_reparam(n_hid, output_dim, self.prior_sig)

        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)

    def forward(self, x, sample=False):
        """forward

        Parameters
        ----------
        x : torch.tensor
            Input.
        sample : bool, optional
            Whether sample weights or not, by default False

        Returns
        -------
        torch.tensor
            Output.
        float
            total log(q(theta)), where q is the variational posterior distribution.
            0 if `sample=False`.
        float
            total log(p(theta)), where p is the prior distribution.
            0 of `sample=False`.
        """
        tlqw = 0
        tlpw = 0

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x, lqw, lpw = self.bfc1(x, sample)
        tlqw += lqw
        tlpw += lpw
        # -----------------
        x = self.act(x)
        # -----------------
        x, lqw, lpw = self.bfc2(x, sample)
        tlqw += lqw
        tlpw += lpw
        # -----------------
        x = self.act(x)
        # -----------------
        y, lqw, lpw = self.bfc3(x, sample)
        tlqw += lqw
        tlpw += lpw

        return y, tlqw, tlpw

    def sample_predict(self, x, Nsamples):
        """Sample predicted outputs from the network.
        
        Used for estimating the data's likelihood by approximately marginalising 
        the weights with MC

        Parameters
        ----------
        x : torch.tensor
            Input.
        Nsamples : int
            Sample size for MC approximation

        Returns
        -------
        torch.tensor
            Predicted output.        
        list
            Vector consists of tlqw for each samples
        list
            Vector consists of tlpw for each samples
        """

        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tlqw_vec = np.zeros(Nsamples)
        tlpw_vec = np.zeros(Nsamples)

        for i in range(Nsamples):
            y, tlqw, tlpw = self.forward(x, sample=True)
            predictions[i] = y
            tlqw_vec[i] = tlqw
            tlpw_vec[i] = tlpw

        return predictions, tlqw_vec, tlpw_vec

class BBP_Bayes_Net_LR(BBP_Bayes_Net):
    """
    BBP_Bayes_Net with Local Reparameterization
    """
    def __init__(
        self, prior_sig=0.1, **kwargs
    ):
        self.prior_sig = prior_sig
        super(BBP_Bayes_Net_LR, self).__init__(
            prior_instance=isotropic_gauss_prior(mu=0.0, sigma=prior_sig), **kwargs)

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = bayes_linear_LR_2L(
            input_dim=self.channels_in * self.side_in * self.side_in, 
            output_dim=self.classes,
            nhid=self.nhid, prior_sig=self.prior_sig)
        if self.cuda:
            self.model = self.model.cuda()
        #             cudnn.benchmark = True
        tot_nb_parameters = self.get_nb_parameters()

        if tot_nb_parameters < 1e3:
            print('    Total params: %f' % (tot_nb_parameters)) 
        elif tot_nb_parameters < 1e6:
            print('    Total params: %.2fK' % (tot_nb_parameters / 1e3)) 
        else:
            print('    Total params: %.2fM' % (tot_nb_parameters / 1e6))
