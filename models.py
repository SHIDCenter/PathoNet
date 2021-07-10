import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import (Input,Add,add,concatenate,Activation,concatenate,
                        Concatenate,Dropout,BatchNormalization,Reshape,Permute,
                        Dense,UpSampling2D,Flatten,Lambda,Activation,Conv2D,
                        DepthwiseConv2D,ZeroPadding2D,GlobalAveragePooling2D,
                        MaxPooling2D,AveragePooling2D,LeakyReLU,Conv2DTranspose)
                        
from keras.regularizers import l2
from keras.utils.layer_utils import get_source_inputs
from keras.utils.data_utils import get_file
from keras.activations import relu
from keras.optimizers import SGD, Adam


WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_X_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
WEIGHTS_PATH_MOBILE_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5"
weight_decay = 1e-5


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs.shape[-1]  # inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand
        print("=========>>>>>>",expansion * in_channels)
        x = Conv2D(int(expansion * in_channels), kernel_size=1, padding='same',
                   use_bias=False, activation=None,name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x

def residualDilatedInceptionModule(y, nb_channels, _strides=(1, 1),t="e"):
    if t=="d":
        y = Conv2D(nb_channels, kernel_size=(1, 1), strides=(1, 1),kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), padding='same', use_bias=False)(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = Conv2D(nb_channels, kernel_size=(1, 1), strides=(1, 1),kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), padding='same', use_bias=False)(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)


    A1 = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides,kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), padding='same', use_bias=False)(y)
    A1 = BatchNormalization()(A1)
    A1 = LeakyReLU()(A1)
    A1 = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides,kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), padding='same', use_bias=False)(A1)
    A1 = BatchNormalization()(A1)
    A1 = LeakyReLU()(A1)


    A4 = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides,kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4),  dilation_rate=4, padding='same', use_bias=False)(y)
    A4 = BatchNormalization()(A4)
    A4 = LeakyReLU()(A4)
    A4 = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides,kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4),  dilation_rate=4, padding='same', use_bias=False)(A4)
    A4 = BatchNormalization()(A4)
    A4 = LeakyReLU()(A4)

    if (t=="e"):
        y=concatenate([y,y])
    y=add([A1,A4,y])
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    return y

def _conv_bn_relu(nb_filter, row, col, subsample = (1,1)):
    def f(input):
        conv_a = Conv2D(nb_filter, (row, col), strides = subsample,
                               kernel_initializer = 'orthogonal', 
                               padding='same', use_bias = False)(input)
        
        norm_a = BatchNormalization()(conv_a)
        act_a = Activation(activation = 'relu')(norm_a)
        return act_a
    return f



def _conv_bn_relu_x2(nb_filter, row, col, subsample = (1,1)):
    def f(input):
        conv_a = Conv2D(nb_filter, (row, col), strides = subsample,
                               kernel_initializer = 'orthogonal', padding = 'same',use_bias = False,
                               kernel_regularizer = l2(weight_decay),
                               bias_regularizer = l2(weight_decay))(input)
        norm_a = BatchNormalization()(conv_a)
        act_a = Activation(activation = 'relu')(norm_a)
        conv_b = Conv2D(nb_filter, (row, col), strides = subsample,
                               kernel_initializer = 'orthogonal', padding = 'same',use_bias = False,
                               kernel_regularizer = l2(weight_decay),
                               bias_regularizer = l2(weight_decay))(act_a)
        norm_b = BatchNormalization()(conv_b)
        act_b = Activation(activation = 'relu')(norm_b)
        return act_b
    return f

def FCRN_A(input_dim, classes=3, pretrained_weights = None):
    input_ = Input (shape = (input_dim))

    block1 = Conv2D(32, (3, 3),kernel_initializer = 'orthogonal', padding='same')(input_)
    act1 = Activation(activation = 'relu')(block1)
    pool1 = MaxPooling2D(pool_size=(2,2))(act1)

    block2 = Conv2D(64, (3, 3),kernel_initializer = 'orthogonal', padding='same')(pool1)
    act2 = Activation(activation = 'relu')(block2)
    pool2 = MaxPooling2D(pool_size=(2,2))(act2)

    block3 = Conv2D(128, (3, 3),kernel_initializer = 'orthogonal', padding='same')(pool2)
    act3 = Activation(activation = 'relu')(block3)
    pool3 = MaxPooling2D(pool_size=(2,2))(act3)

    block4 = Conv2D(512, (3, 3),kernel_initializer = 'orthogonal', padding='same')(pool3)
    act4 = Activation(activation = 'relu')(block4)

    up5=UpSampling2D(size = (2,2))(act4)
    act5 = Activation(activation = 'relu')(up5)
    block5 = Conv2D(128, (3, 3),kernel_initializer = 'orthogonal', padding='same')(act5)

    up6=UpSampling2D(size = (2,2))(block5)
    act6 = Activation(activation = 'relu')(up6)
    block6 = Conv2D(64, (3, 3),kernel_initializer = 'orthogonal', padding='same')(act6)

    up7=UpSampling2D(size = (2,2))(block6)
    act7 = Activation(activation = 'relu')(up7)
    block7 = Conv2D(32, (3, 3),kernel_initializer = 'orthogonal', padding='same')(act7)

    density_pred =  Conv2D(classes, 1, 1, bias = False, activation='linear',init='orthogonal',name='pred',border_mode='same')(block7)

    model = Model (input = input_, output = density_pred)

    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def FCRN_B(input_dim, classes=3, pretrained_weights = None):
    input_ = Input (shape = (input_dim))

    block1 = Conv2D(32, (3, 3),kernel_initializer = 'orthogonal', padding='same')(input_)
    act1 = Activation(activation = 'relu')(block1)
    pool1 = MaxPooling2D(pool_size=(2,2))(act1)

    block2 = Conv2D(64, (3, 3),kernel_initializer = 'orthogonal', padding='same')(pool1)
    act2 = Activation(activation = 'relu')(block2)

    block3 = Conv2D(128, (3, 3),kernel_initializer = 'orthogonal', padding='same')(act2)
    act3 = Activation(activation = 'relu')(block3)
    pool3 = MaxPooling2D(pool_size=(2,2))(act3)

    block4 = Conv2D(256, (5, 5),kernel_initializer = 'orthogonal', padding='same')(pool3)
    act4 = Activation(activation = 'relu')(block4)

    block5 = Conv2D(256, (3, 3),kernel_initializer = 'orthogonal', padding='same')(act4)
    act5 = Activation(activation = 'relu')(block5)

    up6=UpSampling2D(size = (2,2))(act5)
    act6 = Activation(activation = 'relu')(up6)
    block6 = Conv2D(256, (5, 5),kernel_initializer = 'orthogonal', padding='same')(act6)

    up7=UpSampling2D(size = (2,2))(block6)
    act7 = Activation(activation = 'relu')(up7)
    density_pred =  Conv2D(classes, 1, 1, bias = False, activation='linear',init='orthogonal',name='pred',border_mode='same')(act7)
    model = Model (input = input_, output = density_pred)

    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def PathoNet(input_size = (256,256,3), classes=3, pretrained_weights = None):
    inputs = Input(input_size) 

    block1= Conv2D(16, 3, padding = 'same', kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), use_bias=False)(inputs)
    block1 = BatchNormalization()(block1)
    block1 = LeakyReLU()(block1)
    block1= Conv2D(16, 3, padding = 'same', kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), use_bias=False)(block1)
    block1 = BatchNormalization()(block1)
    block1 = LeakyReLU()(block1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(block1)

    block2= residualDilatedInceptionModule(pool1,32,t="e")
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)

    block3= residualDilatedInceptionModule(pool2,64,t="e")
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)

    block4= residualDilatedInceptionModule(pool3,128,t="e")
    pool4 = MaxPooling2D(pool_size=(2, 2))(block4)
    drop4 = Dropout(0.1)(pool4)

    block5= residualDilatedInceptionModule(drop4,256,t="e")
    drop5 = Dropout(0.1)(block5)

    up6 = residualDilatedInceptionModule((UpSampling2D(size = (2,2))(drop5)),128,t="d")
    merge6 = concatenate([block4,up6], axis = 3)

    up7 = residualDilatedInceptionModule((UpSampling2D(size = (2,2))(merge6)),64,t="d")
    merge7 = concatenate([block3,up7], axis = 3)

    up8 = residualDilatedInceptionModule((UpSampling2D(size = (2,2))(merge7)),32,t="d")
    merge8 = concatenate([block2,up8], axis = 3)

    up9 = residualDilatedInceptionModule((UpSampling2D(size = (2,2))(merge8)),16,t="d")
    merge9 = concatenate([block1,up9], axis = 3)

    block9=Conv2D(16, 3, padding = 'same', kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), use_bias=False)(merge9)
    block9 = BatchNormalization()(block9)
    block9 = LeakyReLU()(block9)
    block9=Conv2D(16, 3, padding = 'same', kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), use_bias=False)(block9)
    block9 = BatchNormalization()(block9)
    block9 = LeakyReLU()(block9)
    block9=Conv2D(8, 3, padding = 'same', kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), use_bias=False)(block9)
    block9 = BatchNormalization()(block9)
    block9 = LeakyReLU()(block9)
    conv10 = Conv2D(classes, 1, activation = 'relu')(block9)

    model = Model(input = inputs, output = conv10)

    
    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def Unet(input_size,classes=3,pretrained_weights = None):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    btn1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(btn1)
    btn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(btn1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    btn2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(btn2)
    btn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(btn2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    btn3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(btn3)
    btn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(btn3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    btn4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(btn4)
    btn4 = BatchNormalization()(conv4)
    #drop4 = Dropout(0.5)(btn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(btn4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    btn5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(btn5)
    btn5 = BatchNormalization()(conv5)
    #drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(btn5))
    merge6 = concatenate([btn4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    btn6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(btn6)
    btn6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(btn6))
    merge7 = concatenate([btn3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    btn7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(btn7)
    btn7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(btn7))
    merge8 = concatenate([btn2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    btn8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(btn8)
    btn8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(btn8))
    merge9 = concatenate([btn1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    btn9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(btn9)
    btn9 = BatchNormalization()(conv9)
    conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(btn9)
    btn9 = BatchNormalization()(conv9)
    conv10 = Conv2D(classes, 1, activation = 'linear')(btn9)

    model = Model(input = inputs, output = conv10)
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def Deeplabv3(weights=None, input_tensor=None, input_shape=(256, 256, 3), classes=3, backbone='mobilenetv2',
              OS=16, alpha=1., activation=None):
    """ Instantiates the Deeplabv3+ architecture
    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
    # Arguments
        weights: one of 'pascal_voc' (pre-trained on pascal voc),
            'cityscapes' (pre-trained on cityscape) or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
        classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
            If number of classes not aligned with the weights used, last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        activation: optional activation to add to the top of the network.
            One of 'softmax', 'sigmoid' or None
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
    """

    if not (weights in {'pascal_voc', 'cityscapes', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `pascal_voc`, or `cityscapes` '
                         '(pre-trained on PASCAL VOC)')

    if not (backbone in {'xception', 'mobilenetv2'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`  or `mobilenetv2` ')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    if backbone == 'xception':
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        x = Conv2D(32, (3, 3), strides=(2, 2),
                   name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
        x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = Activation('relu')(x)

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = Activation('relu')(x)

        x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                            skip_connection_type='conv', stride=2,
                            depth_activation=False)
        x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                   skip_connection_type='conv', stride=2,
                                   depth_activation=False, return_skip=True)

        x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                            skip_connection_type='conv', stride=entry_block3_stride,
                            depth_activation=False)
        for i in range(16):
            x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                depth_activation=False)

        x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                            skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                            depth_activation=False)
        x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                            skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                            depth_activation=True)

    else:
        OS = 8
        first_block_filters = _make_divisible(32 * alpha, 8)
        x = Conv2D(first_block_filters,
                   kernel_size=3,
                   strides=(2, 2), padding='same',
                   use_bias=False, name='Conv')(img_input)
        x = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        x = Activation(relu6, name='Conv_Relu6')(x)

        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                expansion=1, block_id=0, skip_connection=False)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1, skip_connection=False)
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                expansion=6, block_id=2, skip_connection=True)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3, skip_connection=False)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=4, skip_connection=True)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                                expansion=6, block_id=6, skip_connection=False)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=7, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=8, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=9, skip_connection=True)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=10, skip_connection=False)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=11, skip_connection=True)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=12, skip_connection=True)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                expansion=6, block_id=13, skip_connection=False)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=14, skip_connection=True)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=15, skip_connection=True)

        x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    import tensorflow as tf
    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(lambda x: tf.image.resize_bilinear(x, size_before[1:3]))(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        size_before2 = tf.keras.backend.int_shape(x)
        x = Lambda(lambda xx: tf.image.resize_bilinear(xx,size_before2[1:3] * tf.constant(OS // 4)))(x)

        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation('relu')(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    if (weights == 'pascal_voc' and classes == 21) or (weights == 'cityscapes' and classes == 19):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(lambda xx: tf.image.resize_bilinear(xx,size_before3[1:3]))(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation)(x)

    model = Model(inputs, x, name='deeplabv3plus')

    # load weights

    if weights == 'pascal_voc':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_X,
                                    cache_subdir='models')
        else:
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_MOBILE,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
    elif weights == 'cityscapes':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                    WEIGHTS_PATH_X_CS,
                                    cache_subdir='models')
        else:
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                    WEIGHTS_PATH_MOBILE_CS,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
    return model

def modelCreator(modelName,inputShape,classes,weights=None):
    if modelName=="PathoNet":
        model=PathoNet(input_size = inputShape, classes=classes,pretrained_weights = weights)
    elif modelName=="FCRN_A":
        model=FCRN_A(inputShape,classes=classes,pretrained_weights = weights)
    elif modelName=="FCRN_B":
        model=FCRN_B(inputShape,classes=classes,pretrained_weights = weights)
    elif modelName=="Unet":
        model=Unet(inputShape,classes=classes,pretrained_weights = weights)
    elif modelName=="Deeplab_xception":
        model=Deeplabv3(weights=None, input_shape=inputShape, classes=classes, backbone='xception',
              OS=16, alpha=1., activation=None)
        model.summary()
        if weights!= None:
            model.load_weights(weights)
    elif modelName=="Deeplab_mobilenet":
        model=Deeplabv3(weights=None, input_shape=inputShape, classes=classes, backbone='mobilenetv2',
              OS=16, alpha=1., activation=None)
        model.summary()
        if weights!= None:
            model.load_weights(weights)
    else:
        raise ValueError('The `model` argument should be either '
                         'PathoNet,FRRN_A,FCRN_B,Deeplab_xception or Deeplab_mobilenet')
    return model
    