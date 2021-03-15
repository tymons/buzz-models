from utils.models.vae import VAE
from utils.models.cvae import cVAE
from utils.models.conv_vae import ConvolutionalVAE
from utils.models.conv_cvae import ConvolutionalCVAE

class HiveModelFactory():
    """ Factory for ml models """

    def _get_vae_model(config, input_size):
        """ Function for building Variational Autoencoder """
        fc_config = config.get('fully_connected', {})
        encoder_layer_sizes = fc_config.get('encoder_layer_sizes', [256, 32, 16])
        decoder_layer_sizes = fc_config.get('decoder_layer_sizes', [16, 32, 256])
        latent_size = fc_config.get('latent_size', 2)

        return VAE(encoder_layer_sizes, latent_size, decoder_layer_sizes)

    def _get_cvae_model(config, input_size):
        """ Function for building Contrastive Variational Autoencoder """
        fc_config = config.get('fully_connected', {})
        encoder_layer_sizes = fc_config.get('encoder_layer_sizes', [256, 32, 16])
        decoder_layer_sizes = fc_config.get('decoder_layer_sizes', [16, 32, 256])
        latent_size = fc_config.get('latent_size', 2)
        
        return cVAE(encoder_layer_sizes, latent_size, decoder_layer_sizes)

    def _get_conv_vae_model(config, input_size):
        """ Function for building Convolutional Variational Autoencoder """

        assert input_size, "input size should be known for convolutional model!"

        conv_config = config.get('convolutional', {})
        encoder_conv_sizes = conv_config.get('encoder_no_feature_maps', [128, 64, 32, 16])
        encoder_mlp_sizes = conv_config.get('encoder_mlp_layer_sizes', [1024, 512, 128])
        decoder_conv_sizes = conv_config.get('decoder_no_feature_maps', [16, 32, 64, 128])
        decoder_mlp_sizes = conv_config.get('decoder_mlp_layer_sizes', [128, 512, 1024])
        latent_size = conv_config.get('latent_size', 16)
        
        return ConvolutionalVAE(encoder_conv_sizes, encoder_mlp_sizes, decoder_conv_sizes, decoder_mlp_sizes, \
                                latent_size, input_size)

    def _get_conv_cvae_model(config, input_size):
        """ Function for building Convolutional Contrastive Variational Autoencoder """
        assert input_size, "input size should be known for convolutional model!"

        conv_config = config.get('convolutional', {})
        encoder_conv_sizes = conv_config.get('encoder_no_feature_maps', [128, 64, 32, 16])
        encoder_mlp_sizes = conv_config.get('encoder_mlp_layer_sizes', [1024, 512, 128])
        decoder_conv_sizes = conv_config.get('decoder_no_feature_maps', [16, 32, 64, 128])
        decoder_mlp_sizes = conv_config.get('decoder_mlp_layer_sizes', [128, 512, 1024])
        latent_size = conv_config.get('latent_size', 16)

        return ConvolutionalCVAE(encoder_conv_sizes, encoder_mlp_sizes, decoder_conv_sizes, decoder_mlp_sizes, \
                                    latent_size, input_size)
    
    def _get_ae_model(config, input_size):
        # TODO: implement
        pass

    def _get_conv_ae_model(config, input_size):
        # TODO: implement
        pass
    
    @classmethod
    def build_model(cls, model_type, config, input_size=None):
        """ Function for building ml model 
        
        Parameters:
            model_type (str): should be one of: {vae, cvae, conv_vae, conv_cvae, ae, conv_ae}
            config (dict):
            input_size (tuple): 

        Returns:
            model 
        """
        model = getattr(cls, f'_get_{model_type}_model', lambda: 'invalid model type')
        return model(config, input_size)