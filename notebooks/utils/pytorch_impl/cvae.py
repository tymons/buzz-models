import torch 
import utils.pytorch_impl.vae as v 

from comet_ml import Experiment
from torch import nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from utils.data_utils import batch_normalize, batch_standarize

def permutate_latent(latents_batch, inplace=False):
    """ Function for element permutation along specified axis
    
    Parameters:
        latent_batch (torch.tensor): input matrix to be permutated
        inplace (bool): modify original tensor or not
    Returns
    """
    data = latents_batch.detach().clone() if inplace == False else latents_batch

    for column_idx in range(latents_batch.shape[-1]):
        rand_indicies = torch.randperm(latents_batch[:, :, column_idx].shape[0])
        latents_batch[:, :, column_idx] = latents_batch[:, :, column_idx][rand_indicies]

    return data


def discriminator_loss(log_ratio_p, log_ratio_q):
    loss_p = nn.functional.binary_cross_entropy_with_logits(log_ratio_p, torch.ones_like(log_ratio_p), reduction='mean')
    loss_q = nn.functional.binary_cross_entropy_with_logits(log_ratio_q, torch.zeros_like(log_ratio_q), reduction='mean')
    return loss_p + loss_q

def _kld_loss(mean, log_var):
    """ KLD loss for normal distribution"""
    return torch.mean(-0.5 * torch.mean(1 + log_var - mean ** 2 - log_var.exp(), dim = 0), dim =1).item()

def cvae_loss(cvae_output, input_target, input_background, kld_weight, discriminator=None, discriminator_aplha=None):
    """
    This function will add reconstruction loss along with KLD
    :param cvae_output: cvae  model output
    :param input_targetL
    :param input_background:
    """    
    print(f'{cvae_output["target"].shape}/{input_target.shape}')

    # MSE target
    loss = nn.functional.mse_loss(cvae_output['target'], input_target, reduction='mean')
    # MSE background
    loss += nn.functional.mse_loss(cvae_output['background'], input_background, reduction='mean')
    # KLD loss target s
    loss += kld_weight * _kld_loss(cvae_output['tg_qs_mean'], cvae_output['tg_qs_log_var'])
    # KLD loss target z
    loss += kld_weight * _kld_loss(cvae_output['tg_qz_mean'], cvae_output['tg_qz_log_var'])
    # KLD loss background z
    loss += kld_weight * _kld_loss(cvae_output['bg_qz_mean'], cvae_output['bg_qz_log_var'])

    if discriminator and discriminator_aplha:
        # total correction loss
        q = torch.cat((cvae_output["latent_qs_target"], cvae_output["latent_qz_target"]), axis=2)
        q_score, _ = discriminator(q, torch.zeros_like(q))
        disc_loss = discriminator_aplha * torch.mean(torch.log(q_score/(1-q_score)))
        loss += disc_loss

    return loss

def train_cvae(model, model_params, dataloader_train, dataloader_val, disc=None, disc_params=None):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['learning_rate'], weight_decay=model_params['weight_decay'])
    optimizer_discriminator = torch.optim.Adam(disc.parameters(), lr=disc_params['learning_rate'], weight_decay=disc_params['weight_decay']) if disc else None
    model.to(device)

    alpha = disc_params['alpha'] if disc else None
    if disc:
        disc.to(device)
        disc.train()

    loss = []
    dloss = []
    for epoch in range(model_params['epochs']):
        ###################
        # train the model #
        ###################
        print(f'-> training at epoch {epoch}', flush=True)
        model.train()          
        with tqdm(total=len(dataloader_train)) as pbar:
            for target, background in tqdm(dataloader_train, position=0, leave=True):
                # cvae
                target = batch_normalize(batch_standarize(target))
                background = batch_normalize(batch_standarize(background))
                target = target.to(device)
                background = background.to(device)
                optimizer.zero_grad()
                output_dict = model(target, background)
                loss = cvae_loss(output_dict, target, background, 1, disc, alpha)
                loss.backward()
                optimizer.step()

                # discirminator
                if disc:
                    q = torch.cat((output_dict["latent_qs_target"].detach(), output_dict["latent_qz_target"].detach()), axis=2).to(device)
                    q_bar = permutate_latent(q)
                    q_bar = q_bar.to(device)

                    optimizer_discriminator.zero_grad()
                    q_score, q_bar_score = disc(q, q_bar)
                    dloss = discriminator_loss(q_score, q_bar_score)
                    dloss.backward()
                    optimizer_discriminator.step()
                
                pbar.update(1)

        ###################
        # val the model   #
        ###################
        print(f'-> validating at epoch {epoch}', flush=True)
        with torch.no_grad():
            model.eval()
            with tqdm(total=len(dataloader_val)) as pbar:
                for target_val, background_val in tqdm(dataloader_val, position=0, leave=True):
                    target_val = batch_normalize(batch_standarize(target_val))
                    background_val = batch_normalize(batch_standarize(background_val))
                    target_val = target_val.to(device)
                    background_val = background_val.to(device)
                    output_dict_val = model(target_val, background_val)
                    vloss = cvae_loss(output_dict_val, target_val, background_val, 1)

                    pbar.update(1)

        print(f'Epoch [{epoch+1}/{model_params["epochs"]}], LOSS: {loss.item():.5f}, VAL_LOSS: {vloss.item():.5f} DISC_LOSS: {(dloss.item() if disc else -1):.5f}')
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
                'bg_qz_log_var': bg_z_log_var,
                'latent_qs_target': tg_s,       # we need this for disentangle and ensure that s and z distributions 
                'latent_qz_target': tg_z}       # for target are independent