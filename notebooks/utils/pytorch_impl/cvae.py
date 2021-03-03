import torch 
import utils.pytorch_impl.vae as v 

from comet_ml import Experiment
from torch import nn

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

def _kld_loss(mean, log_var):
    """ KLD loss for normal distribution"""
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

def cvae_loss(cvae_output, input_target, input_background):
    """
    This function will add reconstruction loss along with KLD
    :param cvae_output: cvae  model output
    :param input_targetL
    :param input_background:
    """    
    # MSE target
    loss = nn.functional.mse_loss(cvae_output['target'], input_target, reduction='sum')
    # MSE background
    loss += nn.functional.mse_loss(cvae_output['background'], input_background, reduction='sum')
    # KLD loss target s
    loss += _kld_loss(cvae_output['tg_qs_mean'], cvae_output['tg_qs_log_var'])
    # KLD loss target z
    loss += _kld_loss(cvae_output['tg_qz_mean'], cvae_output['tg_qz_log_var'])
    # KLD loss background z
    loss += _kld_loss(cvae_output['bg_qz_mean'], cvae_output['bg_qz_log_var'])

    return loss

def train_cvae(model, dataloader_train, dataloader_val, lr, weight_decay, epochs):
    """ function for training cvae 
    
    Parameters:
        model (nn.Module): 
        dataloader_train (DataLoader(DoubleFeatureDataset)): train dataloader with DoubleFeatureDataset as dataset object
        dataloader_val (DataLoader(DoubleFeatureDataset)): validation dataloader with DoubleFeatureDataset as dataset object
        lr (float): learning rate
        weigth_decay (float): weight_decay for adam optimizer
        epochs (int): number of epochs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)

    for epoch in range(epochs):
        ###################
        # train the model #
        ###################
        print(f'-> training at epoch {epoch}', flush=True)
        model.train()
        with tqdm(total=len(dataloader_train)) as pbar:
            for target, background in tqdm(dataloader_train, position=0, leave=True):
                # scale data accross batch
                target[:, 0, :] = torch.Tensor(MinMaxScaler().fit_transform(StandardScaler().fit_transform(target[:, 0, :])))
                background[:, 0, :] = torch.Tensor(MinMaxScaler().fit_transform(StandardScaler().fit_transform(background[:, 0, :])))
                # transfer data to device
                target = target.to(device)
                background = background.to(device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass
                output_dict = model(target, background)
                # calculate the MSE loss with KL
                loss = cvae_loss(output_dict, target, background)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update tqdm bar
                pbar.update(1)

        ###################
        # val the model   #
        ###################
        print(f'-> validating at epoch {epoch}', flush=True)
        model.eval()
        with tqdm(total=len(dataloader_val)) as pbar:
            for target_val, background_val in tqdm(dataloader_val, position=0, leave=True):
                # scale data accross batch
                target_val[:, 0, :] = torch.Tensor(MinMaxScaler().fit_transform(StandardScaler().fit_transform(target_val[:, 0, :])))
                background_val[:, 0, :] = torch.Tensor(MinMaxScaler().fit_transform(StandardScaler().fit_transform(background_val[:, 0, :])))
                # transfer data to device
                target_val = target_val.to(device)
                background_val = background_val.to(device)
                # forward pass
                output_dict_val = model(target_val, background_val)
                # calculate the MSE loss with KL
                vloss = cvae_loss(output_dict_val, target_val, background_val)
                # update tqdm bar
                pbar.update(1)

        print(f'Epoch [{epoch+1}/{epochs}], LOSS: {loss.item()}, VAL_LOSS: {vloss.item()}')

    return model
    

class cVAE(nn.Module):
    """ Class for Contrastive autoencoder """
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):

        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.s_encoder = v.Encoder(encoder_layer_sizes, latent_size)
        self.z_encoder = v.Encoder(encoder_layer_sizes, latent_size)
        self.decoder = v.Decoder(decoder_layer_sizes, 2 * latent_size)

    def forward(self, target, background):
        tg_s_mean, tg_s_log_var = self.s_encoder(target)
        tg_z_mean, tg_z_log_var = self.z_encoder(target)
        bg_z_mean, bg_z_log_var = self.z_encoder(background)    
    
        tg_s = v.reparameterize(tg_s_mean, tg_s_log_var)
        tg_z = v.reparameterize(tg_z_mean, tg_z_log_var)
        bg_z = v.reparameterize(bg_z_mean, bg_z_log_var)

        tg_output = self.decoder(torch.cat((tg_z, tg_s), axis=2))
        bg_output = self.decoder(torch.cat((bg_z, torch.zeros_like(tg_s)), axis=2))

        return {'target': tg_output,
                'tg_qs_mean': tg_s_mean,
                'tg_qs_log_var': tg_s_log_var,
                'tg_qz_mean': tg_z_mean,
                'tg_qz_log_var': tg_z_log_var,
                'background': bg_output,
                'bg_qz_mean': bg_z_mean,
                'bg_qz_log_var': bg_z_log_var}