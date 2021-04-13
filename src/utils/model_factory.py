import torch
import logging
import traceback

from torchsummary import summary

from .models.vae import VAE
from .models.cvae import cVAE
from .models.conv_vae import ConvolutionalVAE
from .models.conv_cvae import ConvolutionalCVAE
from .models.ae import Autoencoder
from .models.conv_ae import ConvolutionalAE
from .models.discriminator import Discriminator


def model_check(model, input_shape):
    """ Function for model check """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        summary(model.to(device), input_shape)
        logging.debug(f'model check success! {model}')
        return model
    except Exception as e:
        logging.error('model self-check failure!')
        logging.error(traceback.print_exc())
        return None


class HiveModelFactory():
    """ Factory for ml models """
    def _get_vae_model(config, input_shape):
        """ Function for building Variational Autoencoder """
        encoder_layer_sizes = config.get('encoder_layer_sizes', [256, 32, 16])
        decoder_layer_sizes = config.get('decoder_layer_sizes', [16, 32, 256])
        latent_size = config.get('latent_size', 2)

        config_used = {
            'encoder_mlp_layers': encoder_layer_sizes,
            'decoder_layer_sizes': decoder_layer_sizes,
            'latent': latent_size
        }
        logging.info(f'building vae model with config: encoder_layers({encoder_layer_sizes}), decoder_layer_sizes({decoder_layer_sizes}),  latent({latent_size})')
        return model_check(VAE(encoder_layer_sizes, latent_size, decoder_layer_sizes, input_shape[0]), (1,) + input_shape), config_used

    def _get_cvae_model(config, input_shape):
        """ Function for building Contrastive Variational Autoencoder """
        encoder_layer_sizes = config.get('encoder_layer_sizes', [256, 32, 16])
        decoder_layer_sizes = config.get('decoder_layer_sizes', [16, 32, 256])
        latent_size = config.get('latent_size', 2)
        
        config_used = {
            'encoder_mlp_layers': encoder_layer_sizes,
            'decoder_layer_sizes': decoder_layer_sizes,
            'latent': latent_size
        }
        validate_input_shape = (1,) + input_shape

        logging.info(f'building cvae model with config: encoder_layers({encoder_layer_sizes}), decoder_layer_sizes({decoder_layer_sizes}),  latent({latent_size})')
        return model_check(cVAE(encoder_layer_sizes, latent_size, decoder_layer_sizes, input_shape[0]), \
                            [validate_input_shape, validate_input_shape]), config_used

    def _get_conv_vae_model(config, input_shape):
        """ Function for building Convolutional Variational Autoencoder """
        encoder_conv_sizes = config.get('encoder_feature_maps', [128, 64, 32, 16])
        encoder_mlp_sizes = config.get('encoder_mlp_layer_sizes', [1024, 512, 128])
        decoder_conv_sizes = config.get('decoder_feature_maps', [16, 32, 64, 128])
        decoder_mlp_sizes = config.get('decoder_mlp_layer_sizes', [128, 512, 1024])
        latent_size = config.get('latent_size', 16)
        
        config_used = {
            'encoder_feature_map': encoder_conv_sizes,
            'encoder_mlp_layers': encoder_mlp_sizes,
            'decoder_feature_map': decoder_conv_sizes,
            'decoder_mlp_layers': decoder_mlp_sizes,
            'latent': latent_size
        }
        logging.info(f'building conv vae model with config: encoder_conv_sizes({encoder_conv_sizes}), encoder_mlp_sizes({encoder_mlp_sizes}), '
                        f'decoder_conv_sizes({decoder_conv_sizes}), decoder_mlp_sizes({decoder_mlp_sizes}), latent({latent_size})')
        return model_check(ConvolutionalVAE(encoder_conv_sizes, encoder_mlp_sizes, decoder_conv_sizes, decoder_mlp_sizes, \
                                latent_size, input_shape), (1,) + input_shape), config_used

    def _get_conv_cvae_model(config, input_shape):
        """ Function for building Convolutional Contrastive Variational Autoencoder """
        encoder_conv_sizes = config.get('encoder_feature_maps', [128, 64, 32, 16])
        encoder_mlp_sizes = config.get('encoder_mlp_layer_sizes', [1024, 512, 128])
        decoder_conv_sizes = config.get('decoder_feature_maps', [16, 32, 64, 128])
        decoder_mlp_sizes = config.get('decoder_mlp_layer_sizes', [128, 512, 1024])
        latent_size = config.get('latent_size', 16)

        config_used = {
            'encoder_feature_map': encoder_conv_sizes,
            'encoder_mlp_layers': encoder_mlp_sizes,
            'decoder_feature_map': decoder_conv_sizes,
            'decoder_mlp_layers': decoder_mlp_sizes,
            'latent': latent_size
        }
        validate_input_shape = (1,) + input_shape
        logging.info(f'building conv cvae model with config: encoder_conv_sizes({encoder_conv_sizes}), encoder_mlp_sizes({encoder_mlp_sizes}), '
                        f'decoder_conv_sizes({decoder_conv_sizes}), decoder_mlp_sizes({decoder_mlp_sizes}), latent({latent_size})')
        return model_check(ConvolutionalCVAE(encoder_conv_sizes, encoder_mlp_sizes, decoder_conv_sizes, decoder_mlp_sizes, \
                                    latent_size, input_shape), [validate_input_shape, validate_input_shape]), config_used
    
    def _get_ae_model(config, input_shape):
        """ Function for building vanilla autoencoder """
        encoder_layer_sizes = config.get('encoder_layer_sizes', [256, 32, 16])
        decoder_layer_sizes = config.get('decoder_layer_sizes', [16, 32, 256])
        latent_size = config.get('latent_size', 2)

        config_used = {
            'encoder_mlp_layers': encoder_layer_sizes,
            'decoder_layer_sizes': decoder_layer_sizes,
            'latent': latent_size
        }
        logging.info(f'building ae model with config: encoder_layers({encoder_layer_sizes}), decoder_layer_sizes({decoder_layer_sizes}),  latent({latent_size})')
        return model_check(Autoencoder(encoder_layer_sizes, latent_size, decoder_layer_sizes, input_shape[0]), (1,) + input_shape), config_used

    def _get_conv_ae_model(config, input_shape):
        """ Function for building convolutional autoencoder """
        encoder_conv_sizes = config.get('encoder_feature_maps', [128, 64, 32, 16])
        encoder_mlp_sizes = config.get('encoder_mlp_layer_sizes', [1024, 512, 128])
        decoder_conv_sizes = config.get('decoder_feature_maps', [16, 32, 64, 128])
        decoder_mlp_sizes = config.get('decoder_mlp_layer_sizes', [128, 512, 1024])
        latent_size = config.get('latent_size', 16)

        config_used = {
            'encoder_feature_map': encoder_conv_sizes,
            'encoder_mlp_layers': encoder_mlp_sizes,
            'decoder_feature_map': decoder_conv_sizes,
            'decoder_mlp_layers': decoder_mlp_sizes,
            'latent': latent_size
        }
        logging.info(f'building conv ae model with config: encoder_conv_sizes({encoder_conv_sizes}), encoder_mlp_sizes({encoder_mlp_sizes}), '
                        f'decoder_conv_sizes({decoder_conv_sizes}), decoder_mlp_sizes({decoder_mlp_sizes}), latent({latent_size})')
        return model_check(ConvolutionalAE(encoder_conv_sizes, encoder_mlp_sizes, decoder_conv_sizes, decoder_mlp_sizes, \
                        latent_size, input_shape), (1,) + input_shape), config_used
        
    def _get_discriminator_model(config, input_shape):
        """ Function for building discirminator model """
        layer_sizes = config.get('layers', [32, 4])

        config_used = { 'discriminator_layers': layer_sizes }
        validation_input_shape = (1,) + input_shape
        logging.info(f'building discriminator model with config: discriminator_layers({layer_sizes})')
        return model_check(Discriminator(layer_sizes, input_shape[0]), [validation_input_shape, validation_input_shape]), config_used

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
        model, params_dict = model_func(config, input_shape)
        model_params_dict = {f"MODEL_{key}": val for key, val in params_dict.items()}

        return model, model_params_dict