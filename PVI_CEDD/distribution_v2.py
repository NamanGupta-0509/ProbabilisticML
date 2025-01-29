from torch import nn
import torch
import numpy as np


# =============================================================================
# Base exponential family distribution
# =============================================================================

class ExponentialFamilyDistribution(nn.Module):
    def __init__(self, std_params=None, nat_params=None, is_trainable=False):

        super().__init__()

        # Specify whether the distribution is trainable wrt its NPs
        self.is_trainable = is_trainable

        # Set all to None.
        self._nat_params = None
        self._std_params = None
        self._unc_params = None
        self._mean_params = None

        # Initialise standard and natural parameters.
        if std_params is not None:
            self.std_params = std_params

        elif nat_params is not None:
            if is_trainable:
                self.std_params = self._std_from_nat(nat_params)
            else:
                self.nat_params = nat_params

        else:
            # No initial parameter values specified.
            raise ValueError(
                "No initial parameterisation specified. "
                "Cannot create optimisable parameters."
            )

    def _clear_params(self):
        """
        Sets all the parameters of self to None.
        """
        self._nat_params = None
        self._std_params = None
        self._unc_params = None
        self._mean_params = None

    @property
    def std_params(self):

        if self.is_trainable:
            return self._std_from_unc(self._unc_params)

        elif self._std_params is None:
            return self._std_from_nat(self._nat_params)

        else:
            return self._std_params

    @std_params.setter
    def std_params(self, std_params):
        self._clear_params()

        if self.is_trainable:
            self._unc_params = nn.ParameterDict(self._unc_from_std(std_params))
        else:
            self._std_params = std_params

    @property
    def nat_params(self):

        # If _nat_params None or distribution trainable compute nat params
        if self.is_trainable or self._nat_params is None:
            self._nat_params = self._nat_from_std(self.std_params)

        return self._nat_params

    @nat_params.setter
    def nat_params(self, nat_params):
        self._clear_params()

        if self.is_trainable:
            self._unc_params = nn.ParameterDict(
                self._unc_from_std(self._std_from_nat(nat_params))
            )
        else:
            self._nat_params = nat_params

    @property
    def mean_params(self):
        if self.is_trainable or self._mean_params is None:
            self._mean_params = self._mean_from_std(self.std_params)

        return self._mean_params

    def non_trainable_copy(self):
        """
        :return: A non-trainable copy with identical parameters.
        """
        if self.is_trainable:
            nat_params = None
            std_params = {k: v.detach().clone() for k, v in self.std_params.items()}

        else:
            if self._std_params is not None:
                std_params = {k: v.detach().clone() for k, v in self.std_params.items()}
                nat_params = None

            elif self._nat_params is not None:
                nat_params = {k: v.detach().clone() for k, v in self.nat_params.items()}
                std_params = None

            else:
                std_params = None
                nat_params = None

        return type(self)(std_params, nat_params, is_trainable=False)

    def trainable_copy(self):
        """
        :return: A trainable copy with identical parameters.
        """
        if self.is_trainable:
            nat_params = None
            std_params = {k: v.detach().clone() for k, v in self.std_params.items()}

        else:
            if self._std_params is not None:
                std_params = {k: v.detach().clone() for k, v in self.std_params.items()}
                nat_params = None

            elif self._nat_params is not None:
                nat_params = {k: v.detach().clone() for k, v in self.nat_params.items()}
                std_params = None

            else:
                std_params = None
                nat_params = None

        return type(self)(std_params, nat_params, is_trainable=True)

    def replace_factor(self, t_old=None, t_new=None, **kwargs):
        # Compute change in natural parameters.
        if t_old is not None and t_new is not None:
            delta_np = {
                k: (t_new.nat_params[k] - t_old.nat_params[k])
                for k in self.nat_params.keys()
            }
        elif t_old is not None and t_new is None:
            delta_np = {k: -t_old.nat_params[k] for k in self.nat_params.keys()}
        elif t_old is None and t_new is not None:
            delta_np = {k: t_new.nat_params[k] for k in self.nat_params.keys()}
        else:
            raise ValueError("t_old or t_new must not be None")

        q_new_nps = {k: v + delta_np[k] for k, v in self.nat_params.items()}

        return self.create_new(nat_params=q_new_nps, **kwargs)

    @property
    def distribution(self):
        return self.torch_dist_class(**self.std_params)

    def kl_divergence(self, p, calc_log_ap=True):
        assert type(p) == type(self), "Distributions must be the same type."

        # Log-partition function.
        log_a = self.log_a().squeeze()

        # Stack natural parameters into single vector.
        np1 = torch.cat(
            [np.flatten(start_dim=self.batch_dims) for np in self.nat_params.values()],
            dim=-1,
        )
        np2 = torch.cat(
            [np.flatten(start_dim=p.batch_dims) for np in p.nat_params.values()], dim=-1
        )

        # Stack mean parameters of q.
        m1 = torch.cat(
            [mp.flatten(start_dim=self.batch_dims) for mp in self.mean_params.values()],
            dim=-1,
        )

        # Compute KL-divergence.
        kl = (np1 - np2).unsqueeze(-2).matmul(m1.unsqueeze(-1)).squeeze()
        kl -= log_a

        if calc_log_ap:
            kl += p.log_a()

        return kl

    def log_prob(self, *args, **kwargs):
        return self.distribution.log_prob(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.distribution.sample(*args, **kwargs)

    def rsample(self, *args, **kwargs):
        return self.distribution.rsample(*args, **kwargs)
    @classmethod
    def create_new(cls, **kwargs):
        return cls(**kwargs)
    
    


# =============================================================================
# Mean field gaussian distribution
# =============================================================================


class MeanFieldGaussianDistribution(ExponentialFamilyDistribution):
    @property
    def torch_dist_class(self):
        return torch.distributions.Normal

    @property
    def batch_dims(self):
        batch_dims = len(self.nat_params["np1"].shape) - 1

        return batch_dims

    def log_a(self, nat_params=None):
        if nat_params is None:
            nat_params = self.nat_params

        np1 = nat_params["np1"]
        np2 = nat_params["np2"]

        batch_dims = len(self.nat_params["np1"].shape) - 1
        if batch_dims == 0:
            d = 1
        else:
            d = np1.shape[-1]

        log_a = -0.5 * np.log(np.pi) * d
        log_a += (-(np1 ** 2) / (4 * np2) - 0.5 * (-2 * np2).log()).sum(-1)

        return log_a

    @staticmethod
    def _std_from_unc(unc_params):
        loc = unc_params["loc"]
        log_var = unc_params["log_var"]

        std = {"loc": loc, "scale": torch.exp(log_var) ** 0.5}

        return std

    @staticmethod
    def _unc_from_std(std_params):
        loc = std_params["loc"].detach()
        scale = std_params["scale"].detach()

        unc = {
            "loc": torch.nn.Parameter(loc),
            "log_var": torch.nn.Parameter(2 * torch.log(scale)),
        }

        return unc

    @staticmethod
    def _nat_from_std(std_params):

        loc = std_params["loc"]
        scale = std_params["scale"]

        nat = {"np1": loc * scale ** -2, "np2": -0.5 * scale ** -2}

        return nat

    @staticmethod
    def _std_from_nat(nat_params):

        np1 = nat_params["np1"]
        np2 = nat_params["np2"]

        std = {"loc": -0.5 * np1 / np2, "scale": (-0.5 / np2) ** 0.5}

        return std

    @staticmethod
    def _mean_from_std(std_params):
        loc = std_params["loc"]
        scale = std_params["scale"]
        mp = {
            "m1": loc,
            "m2": scale ** 2 + loc ** 2,
        }
        return mp

    


# =============================================================================
# Base class for approximating likelihood factors of the exponential family
# Mean field Gaussian factor
# =============================================================================

class MeanFieldGaussianFactor():

    distribution_class = MeanFieldGaussianDistribution

    def __init__(self, nat_params, log_coeff=0.0):

        self.nat_params = nat_params
        self.log_coeff = log_coeff

    def compute_refined_factor(
        self, q1, q2, damping=1.0, valid_dist=False, update_log_coeff=True
    ):
        # Convert distributions to log-coefficients and natural parameters
        np1 = q1.nat_params
        np2 = q2.nat_params

        # Compute natural parameters of the new t-factor (detach gradients)
        delta_np = {
            k: (np1[k].detach().clone() - np2[k].detach().clone())
            for k in self.nat_params.keys()
        }
        nat_params = {
            k: v.detach().clone() + delta_np[k] * damping
            for k, v in self.nat_params.items()
        }

        # if valid_dist:
        #     # Constraint natural parameters to form valid distribution.
        #     nat_params = self.valid_nat_from_nat(nat_params)
        nat_params = self.valid_nat_from_nat(nat_params)

        # if update_log_coeff and not valid_dist:
        #     # TODO: does not work unless valid_dist = False.
        #     log_coeff = self.log_coeff + (q2.log_a() - q1.log_a()) * damping
        #     log_coeff = log_coeff.detach().clone()
        # else:
        #     log_coeff = 0.0
        log_coeff = 0.0

        # Create and return refined t of the same type
        t = type(self)(nat_params=nat_params, log_coeff=log_coeff)

        return t
    
    def valid_nat_from_nat(self, nat_params):
        prec = -2 * nat_params["np2"]

        prec[prec <= 0] = 1e-6 # min precision
        nat_params["np1"][prec <= 0] = 0
        loc = nat_params["np1"] / prec

        nat_params["np2"] = -0.5 * prec
        nat_params["np1"] = loc * prec

        return nat_params