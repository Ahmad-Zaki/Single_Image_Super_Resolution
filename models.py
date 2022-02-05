import numpy as np
from tensorflow.keras.layers import (Input, BatchNormalization, Add, Lambda,
                                     Dense, Conv2D, LeakyReLU, PReLU)
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.nn import depth_to_space



def srresnet(num_res_blocks: int = 16):
    """
    Creates SRResNet model.
    
    Parameters
    ----------
    num_res_blocks: int
        Number of residual blocks in the model
        Default=16 

    Returns
    -------
        SRResNet Model object.
    """
    def PReLU_activation(name):
        return PReLU(Constant(value=0.25), shared_axes=[1,2], name=name)
    
    def residual_block(layer_input, filters, block_number):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same', name=f"conv_res_{block_number}_1")(layer_input)
        d = PReLU_activation(f"prelu_res_{block_number}")(d)
        d = BatchNormalization(momentum=0.8, name=f"BN_res_{block_number}_1")(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same', name=f"conv_res_{block_number}_2")(d)
        d = BatchNormalization(momentum=0.8, name=f"BN_res_{block_number}_2")(d)
        d = Add(name=f"add_res_{block_number}")([d, layer_input])
        return d
    
    def upsample_block(layer_input, scale, i) :
        u = Conv2D(256, kernel_size=3, strides=1, padding='same', name=f"conv_up_{i}")(layer_input)
        u = depth_to_space(u, 2, name=f"pix_shuf_{i}")
        return PReLU_activation(name=f"prelu_up_{i}")(u)
    
    # ==================
    # Model Construction
    # ==================
    
    lr_image = Input(shape=(None, None, 3))
    c1 = Conv2D(64, kernel_size=9, strides=1, padding='same', name="Conv_ip")(lr_image)
    c1 = PReLU_activation(name="prelu_ip")(c1)
    
    r = residual_block(c1, 64, 0)
    for i in range(1,num_res_blocks):
      r = residual_block(r, 64, i)
    
    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same', name="conv_out")(r)
    c2 = BatchNormalization(momentum=0.8, name="BN_out")(c2)
    c2 = Add(name="add_out")([c2, c1])
    
    u1 = upsample_block(c2, 2, 1)
    u2 = upsample_block(u1, 2, 2)
    
    c3 = Conv2D(3, kernel_size=9, strides=1, padding='same', activation="sigmoid", name="conv_final")(u2)
    
    return Model(lr_image, c3, name="SRResNet")



def edsr(num_filters: int = 64, num_res_blocks: int = 16):
    """
    Creates an EDSR model.
    
    Parameters
    ----------
    num_filters: int
        Number of filters per convolution layer.
        Default=64

    num_res_blocks: int
        Number of residual blocks in the model
        Default=16 

    Returns
    -------
        EDSR Model object.
    """
    DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
    normalize = lambda x: (x - DIV2K_RGB_MEAN) / 127.5
    denormalize = lambda x: x * 127.5 + DIV2K_RGB_MEAN
    pixel_shuffle = lambda x: depth_to_space(x, 2)

    def residual_block(layer_input, filters, block_number):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu', name=f"conv_res_{block_number}_1")(layer_input)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same', name=f"conv_res_{block_number}_2")(d)
        d = Add(name=f"add_res_{block_number}")([d, layer_input])
        return d

    def upsample_block(layer_input, i) :
        u = Conv2D(num_filters*4, kernel_size=3, strides=1, padding='same', name=f"conv_up_{i}")(layer_input)
        u = Lambda(pixel_shuffle, name=f"pix_shuf_{i}")(u)
        return u

    # ==================
    # Model Construction
    # ==================

    x_in = Input(shape=(None, None, 3), name="LR Batch")
    x = Lambda(normalize, name="normalize_input")(x_in)

    x = r = Conv2D(num_filters, 3, padding='same', name="Conv_ip")(x)
    for i in range(num_res_blocks):
        r = residual_block(r, num_filters, i)

    c2 = Conv2D(num_filters, 3, padding='same', name="conv_out")(r)
    c2 = Add(name="add_out")([x, c2])

    u1 = upsample_block(c2, 1)
    u2 = upsample_block(u1, 2)
    c3 = Conv2D(3, 3, padding='same', name="conv_final")(u2)

    x_out = Lambda(denormalize, name="denormalize_output")(c3)
    return Model(x_in, x_out, name="EDSR")



def srgan_discriminator():
    """
    Creates a Discriminator model.

    Returns
    -------
        SRGAN_Discriminator Model object.
    """
    def disc_block(input, no_kernels, strides) :
        """discriminator block described in paper"""
        c = Conv2D(no_kernels, kernel_size=3, strides=strides, padding='same')(input)
        c = LeakyReLU(alpha=0.2)(c)
        c = BatchNormalization(momentum=0.8)(c)
        return c
    
    # ==================
    # Model Construction
    # ==================

    x_in = Input(shape=(None, None, 3))

    c = Conv2D(64, kernel_size=3, strides=1, padding='same')(x_in)
    c = LeakyReLU(alpha=0.2)(c)

    d1 = disc_block(c , no_kernels=64 , strides=2)
    d2 = disc_block(d1, no_kernels=128, strides=1)

    d3 = disc_block(d2, no_kernels=128, strides=2)
    d4 = disc_block(d3, no_kernels=256, strides=1)

    d5 = disc_block(d4, no_kernels=256, strides=2)
    d6 = disc_block(d5, no_kernels=512, strides=1)
    
    d7 = disc_block(d6, no_kernels=512, strides=2)

    dense1 = Dense(1024)(d7)
    dense1 = LeakyReLU(alpha=0.2)(dense1)

    dense2 = Dense(1, activation='sigmoid')(dense1)

    return Model(x_in, dense2, name="SRGAN_Discriminator")