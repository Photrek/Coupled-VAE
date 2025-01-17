from pytorch_lightning.callbacks import Callback
import torch

class LossLoggerCallback(Callback):
    def __init__(self, config_name):
        super().__init__()
        self.config_name = config_name

        # Create the file immediately when the callback is initialized
        with open(self.config_name, "w") as f:
            f.write(f"Epoch\tPrior_Variance\tPosterior_Variance\tLoss\tReconstruction_Loss\tKL_Divergence\tMu_Mean\tMu_Std\n")

    def on_train_epoch_end(self, trainer, pl_module):
        # Access the logged metrics for loss, Reconstruction Loss, and KL Divergence
        loss = trainer.callback_metrics.get('loss', None)
        recons_loss = trainer.callback_metrics.get('Reconstruction_Loss', None)
        kld = trainer.callback_metrics.get('KLD', None)

        # Access prior variance and mu
        prior_variance = pl_module.model.params.get('prior_variance', None)  # Access prior variance from the model
        posterior_variance = self._compute_posterior_variance(pl_module)
        mu_mean, mu_std = self._compute_mu(pl_module)

        # Convert variables to safe values for logging
        prior_variance_str = f"{prior_variance:.4f}" if prior_variance is not None else "NA"
        posterior_variance_str = f"{posterior_variance:.4f}" if posterior_variance is not None else "NA"
        loss_str = f"{loss.item():.4f}" if loss is not None else "NA"
        recons_loss_str = f"{recons_loss.item():.4f}" if recons_loss is not None else "NA"
        kld_str = f"{kld.item():.4f}" if kld is not None else "NA"
        mu_mean_str = f"{mu_mean:.4f}" if mu_mean is not None else "NA"
        mu_std_str = f"{mu_std:.4f}" if mu_std is not None else "NA"

        # Print the values to the console
        print(f"Epoch {trainer.current_epoch}:")
        print(f"  Prior Variance: {prior_variance_str}")
        print(f"  Posterior Variance: {posterior_variance_str}")
        print(f"  Mu Mean: {mu_mean_str}, Mu Std: {mu_std_str}")
        print(f"  Loss: {loss_str}")
        print(f"  Reconstruction Loss: {recons_loss_str}")
        print(f"  KL Divergence: {kld_str}")

        # Write the metrics to the file
        try:
            with open(self.config_name, "a") as f:
                f.write(
                    f"{trainer.current_epoch}\t"
                    f"{prior_variance_str}\t"
                    f"{posterior_variance_str}\t"
                    f"{loss_str}\t"
                    f"{recons_loss_str}\t"
                    f"{kld_str}\t"
                    f"{mu_mean_str}\t"
                    f"{mu_std_str}\n"
                )
        except Exception as e:
            print(f"Error writing to file: {e}")
            print(
                f"Current values - Epoch: {trainer.current_epoch}, "
                f"Prior Variance: {prior_variance}, Posterior Variance: {posterior_variance}, "
                f"Loss: {loss}, Reconstruction Loss: {recons_loss}, KL Divergence: {kld}, "
                f"Mu Mean: {mu_mean}, Mu Std: {mu_std}"
            )

    @staticmethod
    def _compute_posterior_variance(pl_module):
        """
        Computes the average posterior variance from the model's output.
        """
        if hasattr(pl_module.model, 'logvar') and pl_module.model.logvar is not None:
            logvar = pl_module.model.logvar
            posterior_variance = torch.mean(torch.exp(logvar)).item()
            print(f"Logvar accessed in callback: {logvar}")
            print(f"Posterior variance: {posterior_variance}")
            return posterior_variance
        else:
            print("Warning: Model does not have `logvar` attribute or it is None.")
            return None

    @staticmethod
    def _compute_mu(pl_module):
        """
        Access and return the mean and standard deviation of mu from the model's output.
        """
        if hasattr(pl_module.model, 'mu') and pl_module.model.mu is not None:
            mu = pl_module.model.mu
            mu_mean = mu.mean().item()
            mu_std = mu.std().item()
            print(f"Mu accessed in callback: Mean = {mu_mean}, Std = {mu_std}")
            return mu_mean, mu_std
        else:
            print("Warning: Model does not have `mu` attribute or it is None.")
            return None, None