import torch
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


def train(model, true_dag_adj, max_epochs=30000, frequency=10, patience=30000, model_path='saved_model', full_config_dict={}):
    model.to(device)
    true_dag_adj = true_dag_adj.to(device)
    model.train()
    prob_abs_losses, sampled_mse_losses = [], []
    best_losses = float("Inf")

    for epoch in range(max_epochs):
        sampled_dag_adj = model.sample()
        sampled_mse_loss = ((true_dag_adj - sampled_dag_adj) ** 2).sum()

        model.optimizer.zero_grad()
        sampled_mse_loss.backward()
        model.optimizer.step()

        if epoch % frequency == 0:
            prob_mask = model.get_prob_mask()
            prob_abs_loss = torch.abs(true_dag_adj - prob_mask).sum()
            prob_abs_losses.append(prob_abs_loss.detach().cpu().numpy())
            sampled_mse_losses.append(sampled_mse_loss.detach().cpu().numpy())

            print("Epoch {} -> prob_abs_loss {} | sampled_mse_loss {}".format(epoch, prob_abs_losses[-1], sampled_mse_losses[-1]))

            if best_losses > prob_abs_losses[-1]:
                best_losses = prob_abs_losses[-1]
                torch.save({'epoch': epoch, 'model_config_dict': full_config_dict, 'model_state_dict': model.state_dict(), 'loss': best_losses}, model_path)
                print('Model saved')

            if np.isnan(prob_abs_losses[-1]):
                print('Detected NaN Loss')
                break

            if int(epoch / frequency) > patience and prob_abs_losses[-patience] <= min(prob_abs_losses[-patience:]):
                print('Early Stopping.')
                break

    return model, prob_abs_losses, sampled_mse_losses
