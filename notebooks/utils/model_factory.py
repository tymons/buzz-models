import torch
import logging

from utils.models.vae import VAE
from utils.models.cvae import cVAE
from utils.models.conv_vae import ConvolutionalVAE
from utils.models.conv_cvae import ConvolutionalCVAE

from torchsummary import summary
from colorama import Back

def model_check(model, input_shape):
    """ Function for model check """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        logging.debug(summary(model.to(device), input_shape))
    except Exception as e:
        logging.error(Back.RED + 'model self-check failure: ' + str(e))
        raise Exception('model self-check failure')

    return model

class HiveModelFactory():
    """ Factory for ml models """
    
    def _get_vae_model(config, input_shape):
        """ Function for building Variational Autoencoder """
        fc_config = config.get('fully_connected', {})
        encoder_layer_sizes = fc_config.get('encoder_layer_sizes', [256, 32, 16])
        decoder_layer_sizes = fc_config.get('decoder_layer_sizes', [16, 32, 256])
        latent_size = fc_config.get('latent_size', 2)

        return model_check(VAE(encoder_layer_sizes, latent_size, decoder_layer_sizes, input_shape[0]), (1,) + input_shape)

    def _get_cvae_model(config, input_shape):
        """ Function for building Contrastive Variational Autoencoder """
        fc_config = config.get('fully_connected', {})
        encoder_layer_sizes = fc_config.get('encoder_layer_sizes', [256, 32, 16])
        decoder_layer_sizes = fc_config.get('decoder_layer_sizes', [16, 32, 256])
        latent_size = fc_config.get('latent_size', 2)
        
        validate_input_shape = (1,) + input_shape
        return model_check(cVAE(encoder_layer_sizes, latent_size, decoder_layer_sizes, input_shape[0]), \
                            [validate_input_shape, validate_input_shape])

    def _get_conv_vae_model(config, input_shape):
        """ Function for building Convolutional Variational Autoencoder """
        conv_config = config.get('convolutional', {})
        encoder_conv_sizes = conv_config.get('encoder_no_feature_maps', [128, 64, 32, 16])
        encoder_mlp_sizes = conv_config.get('encoder_mlp_layer_sizes', [1024, 512, 128])
        decoder_conv_sizes = conv_config.get('decoder_no_feature_maps', [16, 32, 64, 128])
        decoder_mlp_sizes = conv_config.get('decoder_mlp_layer_sizes', [128, 512, 1024])
        latent_size = conv_config.get('latent_size', 16)
        
        return model_check(ConvolutionalVAE(encoder_conv_sizes, encoder_mlp_sizes, decoder_conv_sizes, decoder_mlp_sizes, \
                                latent_size, input_shape), (1,) + input_shape)

    def _get_conv_cvae_model(config, input_shape):
        """ Function for building Convolutional Contrastive Variational Autoencoder """
        conv_config = config.get('convolutional', {})
        encoder_conv_sizes = conv_config.get('encoder_no_feature_maps', [128, 64, 32, 16])
        encoder_mlp_sizes = conv_config.get('encoder_mlp_layer_sizes', [1024, 512, 128])
        decoder_conv_sizes = conv_config.get('decoder_no_feature_maps', [16, 32, 64, 128])
        decoder_mlp_sizes = conv_config.get('decoder_mlp_layer_sizes', [128, 512, 1024])
        latent_size = conv_config.get('latent_size', 16)

        validate_input_shape = (1,) + input_shape
        return model_check(ConvolutionalCVAE(encoder_conv_sizes, encoder_mlp_sizes, decoder_conv_sizes, decoder_mlp_sizes, \
                                    latent_size, input_shape), [validate_input_shape, validate_input_shape])
    
    def _get_ae_model(config, input_size):
        # TODO: implement
        pass

    def _get_conv_ae_model(config, input_size):
        # TODO: implement
        pass
    
    @classmethod
    def build_model(cls, model_type, config, input_shape):
        """ Function for building ml model 
        
        Parameters:
            model_type (str): should be one of: {vae, cvae, conv_vae, conv_cvae, ae, conv_ae}
            config (dict):
            input_size (tuple): 

        Returns:
            model 
        """
        model_func = getattr(cls, f'_get_{model_type}_model', lambda: 'invalid model type')
        return model_func(config, input_shape)