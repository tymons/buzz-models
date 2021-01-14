import tensorflow as tf

from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras import backend as K

def contrastive_keras_vae(input_dim=4, intermediate_dim=12, latent_dim=2, beta=1, disentangle=False, gamma=0):
    input_shape = (input_dim, )
    batch_size = 100
    tg_inputs = tf.keras.layers.Input(shape=input_shape, name='tg_input')
    bg_inputs = tf.keras.layers.Input(shape=input_shape, name='bg_input')
    
    if isinstance(intermediate_dim, int):
        intermediate_dim = [intermediate_dim]
    
    z_h_layers = []
    for dim in intermediate_dim:
        z_h_layers.append(tf.keras.layers.Dense(dim, activation='relu'))
    z_mean_layer = tf.keras.layers.Dense(latent_dim, name='z_mean')
    z_log_var_layer = tf.keras.layers.Dense(latent_dim, name='z_log_var')
    z_layer = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')

    def z_encoder_func(inputs):
        z_h = inputs
        for z_h_layer in z_h_layers:
            z_h = z_h_layer(z_h)
        z_mean = z_mean_layer(z_h)
        z_log_var = z_log_var_layer(z_h)
        z = z_layer([z_mean, z_log_var])
        return z_mean, z_log_var, z

    s_h_layers = []
    for dim in intermediate_dim:
        s_h_layers.append(tf.keras.layers.Dense(dim, activation='relu'))
    s_mean_layer = tf.keras.layers.Dense(latent_dim, name='s_mean')
    s_log_var_layer = tf.keras.layers.Dense(latent_dim, name='s_log_var')
    s_layer = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='s')

    def s_encoder_func(inputs):
        s_h = inputs
        for s_h_layer in s_h_layers:
            s_h = s_h_layer(s_h)
        s_mean = s_mean_layer(s_h)
        s_log_var = s_log_var_layer(s_h)
        s = s_layer([s_mean, s_log_var])
        return s_mean, s_log_var, s

    tg_z_mean, tg_z_log_var, tg_z = z_encoder_func(tg_inputs)
    tg_s_mean, tg_s_log_var, tg_s = s_encoder_func(tg_inputs)
    bg_s_mean, bg_s_log_var, bg_s = s_encoder_func(bg_inputs)

    z_encoder = tf.keras.models.Model(tg_inputs, [tg_z_mean, tg_z_log_var, tg_z], name='z_encoder')
    s_encoder = tf.keras.models.Model(tg_inputs, [tg_s_mean, tg_s_log_var, tg_s], name='s_encoder')

    # build decoder model
    cvae_latent_inputs = tf.keras.layers.Input(shape=(2 * latent_dim,), name='sampled')
    cvae_h = cvae_latent_inputs
    for dim in intermediate_dim:
        cvae_h = tf.keras.layers.Dense(dim, activation='relu')(cvae_h)
    cvae_outputs = tf.keras.layers.Dense(input_dim)(cvae_h)

    cvae_decoder = tf.keras.models.Model(inputs=cvae_latent_inputs, outputs=cvae_outputs, name='decoder')

    # decoder.summary()

    def zeros_like(x):
        return tf.zeros_like(x)

    tg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, tg_s], -1))
    zeros = tf.keras.layers.Lambda(zeros_like)(tg_z)
    bg_outputs = cvae_decoder(tf.keras.layers.concatenate([zeros, bg_s], -1))
    fg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, zeros], -1))

    cvae = tf.keras.models.Model(inputs=[tg_inputs, bg_inputs], 
                                 outputs=[tg_outputs, bg_outputs], 
                                 name='contrastive_vae')

    cvae_fg = tf.keras.models.Model(inputs=tg_inputs, 
                                    outputs=fg_outputs, 
                                    name='contrastive_vae_fg')

    # print(cvae.summary())
    if disentangle:
        discriminator = Dense(1, activation='sigmoid')
        
        z1 = Lambda(lambda x: x[:int(batch_size/2),:])(tg_z)
        z2 = Lambda(lambda x: x[int(batch_size/2):,:])(tg_z)
        s1 = Lambda(lambda x: x[:int(batch_size/2),:])(tg_s)
        s2 = Lambda(lambda x: x[int(batch_size/2):,:])(tg_s)
        q_bar = tf.keras.layers.concatenate(
            [tf.keras.layers.concatenate([s1, z2], axis=1),
            tf.keras.layers.concatenate([s2, z1], axis=1)],
            axis=0)
        q = tf.keras.layers.concatenate(
            [tf.keras.layers.concatenate([s1, z1], axis=1),
            tf.keras.layers.concatenate([s2, z2], axis=1)],
            axis=0)
        q_bar_score = discriminator(q_bar)
        q_score = discriminator(q)        
        tc_loss = K.log(q_score / (1 - q_score)) 
        
        discriminator_loss = - K.log(q_score) - K.log(1 - q_bar_score)

    reconstruction_loss = tf.keras.losses.mse(tg_inputs, tg_outputs)
    reconstruction_loss += tf.keras.losses.mse(bg_inputs, bg_outputs)
    reconstruction_loss *= input_dim


    kl_loss = 1 + tg_z_log_var - tf.keras.backend.square(tg_z_mean) - tf.keras.backend.exp(tg_z_log_var)
    kl_loss += 1 + tg_s_log_var - tf.keras.backend.square(tg_s_mean) - tf.keras.backend.exp(tg_s_log_var)
    kl_loss += 1 + bg_s_log_var - tf.keras.backend.square(bg_s_mean) - tf.keras.backend.exp(bg_s_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    if disentangle:
        cvae_loss = K.mean(reconstruction_loss) + beta*K.mean(kl_loss) + gamma * K.mean(tc_loss) + K.mean(discriminator_loss)
    else:
        cvae_loss = K.mean(reconstruction_loss) + K.mean(kl_loss)

    cvae.add_loss(cvae_loss)
    cvae.compile(optimizer='adam')
    
    return cvae, cvae_fg, z_encoder, s_encoder, cvae_decoder