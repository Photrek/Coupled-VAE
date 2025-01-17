import torch
from typing import List, Union
import itertools
from torch.nn import functional as F


# Coupled Logarithm Function
def coupled_logarithm(value: Union[int, float, torch.Tensor], kappa: float = 0.0, dim: int = 1) -> torch.Tensor:
    value = torch.tensor(value, dtype=torch.float32) if isinstance(value, (int, float)) else value
    assert isinstance(value, torch.Tensor), "value must be an int, float, or torch.Tensor."
    assert 0. not in value, "value must not contain zero(s)."
    
    if kappa == 0.0:
        return torch.log(value)
    else:
        return (1. / kappa) * (value ** (kappa / (1. + dim * kappa)) - 1.)

# Coupled Exponential Function
def coupled_exponential(value: Union[int, float, torch.Tensor], kappa: float = 0.0, dim: int = 1) -> torch.Tensor:
    value = torch.tensor(value, dtype=torch.float32) if isinstance(value, (int, float)) else value
    assert isinstance(value, torch.Tensor), "value must be an int, float, or torch.Tensor."
    
    if kappa == 0.0:
        return torch.exp(value)
    else:
        condition_1 = (1 + kappa * value) > 0
        condition_2 = ((1 + kappa * value) <= 0) & (((1 + dim * kappa) / kappa) > 0)
        
        coupled_exp_value = torch.where(condition_1, (1 + kappa * value) ** ((1 + dim * kappa) / kappa), float('inf'))
        coupled_exp_value = torch.where(condition_2, torch.tensor(0.0), coupled_exp_value)
    
    return coupled_exp_value

def coupled_sum(values: torch.Tensor, kappa: float = 0.0, max_order: int = 1) -> torch.Tensor:
    """
    Computes the coupled sum with interactions up to a specified maximum order.
    
    Parameters
    ----------
    values : Tensor of input values to compute the coupled sum.
    kappa : Coupling parameter that modifies the sum.
    max_order : Maximum order of interactions to consider.
    
    Returns
    -------
    Coupled sum as a torch.Tensor.
    """
    assert isinstance(values, torch.Tensor), "values must be a torch.Tensor."
    values = values.flatten()
    n = values.numel()  # Number of elements in the tensor
    
    # Handle the standard sum case
    if kappa == 0.0:
        return torch.sum(values)
    
    # First-order term
    coupled_sum = torch.sum(values)
    
    # Add higher-order interaction terms up to max_order
    for order in range(2, min(max_order, n) + 1):
        interaction_terms = [torch.prod(torch.tensor(combo)) for combo in itertools.combinations(values, r=order)]
        coupled_sum += (kappa ** (order - 1)) * torch.sum(torch.stack(interaction_terms))
        
    return coupled_sum

# Coupled Product Function for Multiple Variables
def coupled_product(values: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
    assert isinstance(values, torch.Tensor), "values must be a torch.Tensor."
    
    if kappa == 0.0:
        return torch.prod(values)
    else:
        n = values.numel()
        powered_terms = torch.sum(values ** kappa) - (n - 1)
        return powered_terms ** (1 / kappa)

# Coupled Power Function
def coupled_power(x: torch.Tensor, a: float, kappa: float = 0.0) -> torch.Tensor:
    if kappa == 0.0:
        return x ** a
    else:
        return ((a * x ** (kappa * a)) - (a - 1)) ** (1 / (kappa * a))
    
import torch

def coupled_subtraction(x: torch.Tensor, y: torch.Tensor, kappa: float) -> torch.Tensor:
    """
    Corrected coupled subtraction function.
    """
    return (x - y) / (1 + kappa * y)

# Coupled Divergence Function
def coupled_gaussians_divergence(mu, logvar, mu_hat, logvar_hat, kappa):
    if kappa == 0.0:
        std = torch.exp(0.5 * logvar)
        std_hat = torch.exp(0.5 * logvar_hat)
        kld = torch.sum(logvar_hat - logvar + (std ** 2 + (mu - mu_hat) ** 2) / (2 * std_hat ** 2) - 0.5, dim=1)
        return kld

    sigma = torch.exp(0.5 * logvar)
    sigma_hat = torch.exp(0.5 * logvar_hat)
    d = mu.shape[1]
    
    coupled_term_1 = 1 + d * kappa
    coupled_term_2 = 1 + d * kappa + 2 * kappa

    term1 = (2 * torch.pi) ** (kappa / coupled_term_1)
    term2 = torch.sqrt(coupled_term_2 / (coupled_term_1 + 2 * kappa * (sigma - sigma_hat ** 2)))
    term3_exp = torch.exp(
        ((mu - mu_hat) ** 2) * coupled_term_2 * kappa / (coupled_term_1 * (coupled_term_1 + 2 * kappa * (sigma ** 2 - sigma_hat ** 2)))
    )
    
    term4 = (2 * torch.pi * sigma_hat ** 2) ** (kappa / coupled_term_1) * torch.sqrt(torch.tensor(coupled_term_2 / coupled_term_1))
    
    coupled_div = (1 / (2 * kappa)) * torch.prod(term1 * term2 * term3_exp, dim=1) - torch.prod(term4, dim=1)

    return coupled_div

def coupled_cross_entropy(pred, target, kappa):
    """
    Computes the coupled cross-entropy loss.

    :param pred: (Tensor) Predicted probabilities [B x C x H x W]
    :param target: (Tensor) Ground truth labels [B x C x H x W]
    :param kappa: (float) Coupling parameter
    :return: (Tensor) Coupled cross-entropy loss
    """
    # Ensure predictions are in the range (0, 1)
    pred = torch.sigmoid(pred)
    
    # Coupled logarithm for predictions
    log_q = coupled_logarithm(pred, kappa)
    log_1_minus_q = coupled_logarithm(1 - pred, kappa)
    
    # Compute coupled cross-entropy
    coupled_ce = target * log_q + (1 - target) * log_1_minus_q
    
    return -coupled_ce.sum()  # Sum over all dimensions except batch size

def coupled_mse(input: torch.Tensor, recons: torch.Tensor, kappa: float = 0.0) -> torch.Tensor:
    """
    Computes the coupled MSE loss, a generalization of MSE with a coupled logarithm and exponential term.
    
    :param input: Ground truth image (torch.Tensor)
    :param recons: Reconstructed image (torch.Tensor)
    :param kappa: Coupling parameter (float)
    :return: Coupled MSE loss (torch.Tensor)
    """
    # Calculate the squared differences
    squared_diff = (input - recons) ** 2
    
    # Apply exponential to the squared differences
    exp_squared_diff = torch.exp(squared_diff)
    
    # Apply the coupled logarithm to the exponential squared differences
    coupled_mse_loss = coupled_logarithm(exp_squared_diff, kappa)
    
    # Sum over all pixels and average over the batch
    return coupled_mse_loss.mean()

def compute_elbo(recons, input, mu, logvar, mu_hat, logvar_hat, kappa, kld_weight):
    """
    Computes the Evidence Lower Bound (ELBO) using coupled MSE and coupled Gaussian divergence.
    """
    # Use coupled MSE as the reconstruction loss
    recons_loss = coupled_mse(input, recons, kappa)
    kld_loss = coupled_gaussians_divergence(mu, logvar, mu_hat, logvar_hat, kappa)
    
    # Compute ELBO
    elbo = recons_loss + kld_weight * kld_loss.mean()
    return elbo, recons_loss, kld_loss

def compute_coupled_probability(q_values, alpha, kappa, dim):
    """
    Compute the coupled probability Q based on the given q-values.
    Args:
        q_values (torch.Tensor): Input probabilities or latent space activations.
        alpha (float): Hyperparameter from the config file.
        kappa (float): Coupling parameter.
        dim (int): Dimensionality of the latent space.
    Returns:
        torch.Tensor: Coupled probabilities normalized.
    """
    r = alpha * kappa / (1 + dim * kappa)  # Compute 'r' based on the correct formula
    coupled_probs = q_values ** (1 + r)  # Raise q_values to power (1 + r)
    normalized_probs = coupled_probs / torch.sum(coupled_probs, dim=1, keepdim=True)  # Normalize
    return normalized_probs

def regular_mse(input: torch.Tensor, recons: torch.Tensor) -> torch.Tensor:
    """
    Computes the standard Mean Squared Error (MSE) loss.

    :param input: Ground truth image (torch.Tensor)
    :param recons: Reconstructed image (torch.Tensor)
    :return: Regular MSE loss (torch.Tensor)
    """
    mse_loss = F.mse_loss(recons, input, reduction='mean')
    return mse_loss


def regular_gaussian_divergence(mu: torch.Tensor, logvar: torch.Tensor, prior_variance: float) -> torch.Tensor:
  """
  Computes the KL divergence between two Gaussians.
  """
  prior_var = torch.tensor(prior_variance, device=mu.device)
  prior_logvar = torch.log(prior_var)
  
  kld = -0.5 * torch.sum(
    1 + logvar - prior_logvar - (mu.pow(2) + logvar.exp()) / prior_var,
    dim=1
  )
  return kld.mean()  # Average over the batch

def regular_elbo(
    recons: torch.Tensor, input: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kld_weight: float, prior_variance: float) -> torch.Tensor:
        """
        Computes the Evidence Lower Bound (ELBO) using regular MSE and Gaussian divergence.
    
        :param recons: Reconstructed image (torch.Tensor)
        :param input: Ground truth image (torch.Tensor)
        :param mu: Mean of the latent space distribution (torch.Tensor)
        :param logvar: Log variance of the latent space distribution (torch.Tensor)
        :param kld_weight: Weighting factor for the KL divergence
        :return: Tuple containing ELBO, reconstruction loss, and KL divergence
        """
        recons_loss = regular_mse(input, recons)
        kld_loss = regular_gaussian_divergence(mu, logvar, prior_variance)
        elbo = recons_loss + kld_weight * kld_loss
        return elbo, recons_loss, kld_loss