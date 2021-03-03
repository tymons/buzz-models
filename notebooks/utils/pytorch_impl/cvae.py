from comet_ml import Experiment

from torch import nn
from utils.pytorch_impl.vae import Encoder, Decoder, reparameterize, vae_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def cvae_loss(target, background, recon_target, recon_background):
    """
    This function will add reconstruction loss along with KLD
    :param target: input target data
    :param background: input background data
    :param recon_target: tuple with target reconstruction s mean/var and z mean/var
    :param recon_background: tuple with background reconstruction z mean/var
    """    
    # BCE target
    loss = F.mse_loss(recon_target[0].view(-1, 28*28), target.view(-1, 28*28), reduction='sum')
    # BCE background
    loss += F.mse_loss(recon_background[0].view(-1, 28*28), background.view(-1, 28*28), reduction='sum')
    # KLD loss target s
    loss += _kld_loss(recon_target[1], recon_target[2])
    # KLD loss target z
    loss += _kld_loss(recon_target[3], recon_target[4])
    # KLD loss background z
    loss += _kld_loss(recon_background[1], recon_background[2])

    return (loss)

def train_cvae(model, hyperparameters):
    """ function for training cvae 
    
    Parameters:
        model (nn.Module)
        hyperparameters (dict)

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])

    model.to(device)

    for epoch in range(1, hyperparameters + 1):
        ###################
        # train the model #
        ###################
        print(f'-> training at epoch {epoch}', flush=True)
        model.train()
        with tqdm(total=len(dataloader_train)) as pbar:
            for input_data, labels in tqdm(dataloader_train, position=0, leave=True):
                if scale:
                    # scale data accross batch
                    input_data[:, 0, :] = torch.Tensor(MinMaxScaler().fit_transform(StandardScaler().fit_transform(input_data[:, 0, :])))

                # transfer data to device
                input_data = input_data.to(device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass
                outputs, mu, log_var = model(input_data)
                # calculate the BCE loss with KL
                loss = cvae_loss(outputs, input_data, mu, log_var)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update tqdm bar
                pbar.update(1)

    # ###################
    # # val the model   #
    # ###################
    # print(f'-> validating at epoch {epoch}', flush=True)
    # model.eval()
    # with tqdm(total=len(dataloader_val)) as pbar:
    #     for input_data_val, labels in tqdm(dataloader_val, position=0, leave=True):
    #         if scale:
    #             # scale data accross batch
    #             input_data_val[:, 0, :] = torch.Tensor(MinMaxScaler().fit_transform(StandardScaler().fit_transform(input_data_val[:, 0, :])))

    #         # transfer data to device
    #         input_data_val = input_data_val.to(device)
    #         # forward pass
    #         val_outputs, val_mu, val_log_var  = model(input_data_val)
    #         # calculate the BCE loss with KL
    #         vloss = vae_loss(val_outputs, input_data_val, val_mu, val_log_var)
    #         # update running val loss
    #         val_loss.append(vloss.item())
    #         pbar.update(1)

class cVAE(nn.Module):
    """ Class for Contrastive autoencoder """
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):

        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.s_encoder = Encoder(encoder_layer_sizes, latent_size)
        self.z_encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, 2 * latent_size)

    def forward(self, target, background):
        tg_s_mean, tg_s_log_var = self.encoder_s(target)
        tg_z_mean, tg_z_log_var = self.encoder_z(target)
        bg_z_mean, bg_z_log_var = self.encoder_z(background)    
    
        tg_s = self.reparameterize(tg_s_mean, tg_s_log_var)
        tg_z = self.reparameterize(tg_z_mean, tg_z_log_var)
        bg_z = self.reparameterize(bg_z_mean, bg_z_log_var)

        tg_output = self.decoder(torch.cat((tg_z, tg_s), axis=2))
        bg_output = self.decoder(torch.cat((bg_z, torch.zeros_like(tg_s)), axis=2))

        return (tg_output, tg_s_mean, tg_s_log_var, tg_z_mean, tg_z_log_var), (bg_output, bg_z_mean, bg_z_log_var)