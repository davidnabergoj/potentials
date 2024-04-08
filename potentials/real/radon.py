from potentials.base import StructuredPotential
import torch


class RadonVaryingInterceptsAndSlopes(StructuredPotential):
    # https://www.tensorflow.org/probability/examples/Multilevel_Modeling_Primer
    # TODO add dataset
    def __init__(self):
        # number of parameters: 2 + 85 + 1 + 2 + 85 = 175
        super().__init__(event_shape=(175,))
        self.features = ...
        self.labels = ...

    def compute(self, model_params):
        # Extract parameters
        mu_a = model_params[..., 0]
        log_sigma_a = model_params[..., 1]
        mu_b = model_params[..., 2]
        log_sigma_b = model_params[..., 3]
        log_sigma_y = model_params[..., 4]
        a = model_params[..., 5:90]
        b = model_params[..., 90:175]

        # Transform log scales to scales
        sigma_a = torch.exp(log_sigma_a)
        sigma_b = torch.exp(log_sigma_b)
        sigma_y = torch.exp(log_sigma_y)

        log_det_sigma_a = log_sigma_a
        log_det_sigma_b = log_sigma_b
        log_det_sigma_y = log_sigma_y
        log_det = log_det_sigma_a + log_det_sigma_b + log_det_sigma_y

        # Compute probabilities
        # TODO use torch.distributions.Independent for log_prob_a, log_prob_b, log_prob_y
        log_prob_mu_a = torch.distributions.Normal(loc=0, scale=1e5).log_prob(mu_a)
        log_prob_sigma_a = torch.distributions.HalfCauchy(scale=5).log_prob(sigma_a)
        log_prob_mu_b = torch.distributions.Normal(loc=0, scale=1e5).log_prob(mu_b)
        log_prob_sigma_b = torch.distributions.HalfCauchy(scale=5).log_prob(sigma_b)
        log_prob_sigma_y = torch.distributions.HalfCauchy(scale=5).log_prob(sigma_y)
        log_prob_a = torch.distributions.Normal(loc=mu_a, scale=sigma_a).log_prob(a)
        log_prob_b = torch.distributions.Normal(loc=mu_b, scale=sigma_b).log_prob(b)
        log_prob_y = torch.distributions.Normal(loc=a * self.features + b, scale=sigma_y).log_prob(self.target)

        log_prob = (
                log_prob_mu_a
                + log_prob_sigma_a
                + log_prob_mu_b
                + log_prob_sigma_b
                + log_prob_sigma_y
                + log_prob_a
                + log_prob_b
                + log_prob_y
        )

        return -(log_prob + log_det)
