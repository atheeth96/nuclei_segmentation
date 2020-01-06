import keras
from keras.models import *
from keras.layers import *

from keras.preprocessing import image
import keras.backend as K



def unet(pretrained_weights = None,input_size = (None,None,3),act='relu',optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy']):
    inputs = Input(input_size)
    assert act in ['relu','leaky_relu']
    assert optimizer in ['adam','sgd']
    if act=='relu':
        activation1='relu'
    else:
        activation1=LeakyReLU(alpha=0.5)
    activation='relu'
    
    
    
    
    conv1 = Conv2D(64, 3, activation = activation1, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #bach_size x 512 x 512 x 64
    conv1 = Conv2D(64, 3, activation = activation1, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    #bach_size x 512 x 512 x 64
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #bach_size x 256 x 256 x 64
    conv2 = Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #bach_size x 256 x 256 x 128
    conv2 = Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    #bach_size x 256 x 256 x 128
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #bach_size x 128 x 128 x 128
    conv3 = Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #bach_size x 128 x 128 x 256
    conv3 = Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #bach_size x 128 x 128 x 256
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #bach_size x 64 x 64 x 256
    conv4 = Conv2D(512, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #bach_size x 64 x 64 x 512
    conv4 = Conv2D(512, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #bach_size x 64 x 64 x 512
    drop4 = Dropout(0.5)(conv4)
    
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    #conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    #conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #drop5 = Dropout(0.5)(conv5)

    #up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    #merge6 = concatenate([drop4,up6], axis = 3)
    #conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    
    up7 = Conv2D(256, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))#(conv6))
    #bach_size x 128 x 128 x 256
    merge7 = concatenate([conv3,up7], axis = 3)
    #bach_size x 128 x 128 x 512
    conv7 = Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    #bach_size x 128 x 128 x 256
    conv7 = Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #bach_size x 128 x 128 x 256

    up8 = Conv2D(128, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    #bach_size x 256 x 256 x 128
    merge8 = concatenate([conv2,up8], axis = 3)
    #bach_size x 256 x 256 x 256
    conv8 = Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    #bach_size x 256 x 256 x 128
    conv8 = Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    #bach_size x 256 x 256 x 128

    up9 = Conv2D(64, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #bach_size x 512 x 512 x 64
    merge9 = concatenate([conv1,up9], axis = 3)
     #bach_size x 512 x 512 x 128
    conv9 = Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge9)
     #bach_size x 512 x 512 x 64
    conv9 = Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv9)
     #bach_size x 512 x 512 x 64
    conv9 = Conv2D(2, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv9)
     #bach_size x 512 x 512 x 2
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
     #bach_size x 512 x 512 x 1

    model = Model(input = inputs, output = conv10)
    
    if optimizer=='adam':
        OPT=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000, amsgrad=False)
        print('adam optimizer')
    elif optimizer=='sgd':
        OPT=keras.optimizers.SGD(lr=0.01,momentum=0.8,nesterov=True)
        print('SGD optimizer')
    #dice_coef_loss
    model.compile(optimizer = OPT, loss = loss, metrics = metrics)    

    return model

def expend_as(tensor, rep):
        my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
        return my_repeat

def AttnGatingBlock(x, g, inter_shape):
    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    # my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
    # upsample_psi=my_repeat([upsample_psi])
    upsample_psi = expend_as(upsample_psi, shape_x[3])

    y = multiply([upsample_psi, x])

    # print(K.is_keras_tensor(upsample_psi))

    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn

def UnetGatingSignal(input, is_batchnorm=True):
    shape = K.int_shape(input)
    x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attn_unet(pretrained_weights = None,input_size = (512,512,3),act='relu',optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']):
    inputs = Input(input_size)
    assert act in ['relu','leaky_relu']
    assert optimizer in ['adam','sgd']
    if act=='relu':
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        #bach_size x 512 x 512 x 64
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        #bach_size x 512 x 512 x 64
    else:
        conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1=LeakyReLU(alpha=0.5)(conv1)
        #bach_size x 512 x 512 x 64
        conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv1=LeakyReLU(alpha=0.5)(conv1)
        #bach_size x 512 x 512 x 64
    activation='relu'
  
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #bach_size x 256 x 256 x 64
    conv2 = Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #bach_size x 256 x 256 x 128
    conv2 = Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    #bach_size x 256 x 256 x 128
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #bach_size x 128 x 128 x 128
    conv3 = Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #bach_size x 128 x 128 x 256
    conv3 = Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #bach_size x 128 x 128 x 256
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #bach_size x 64 x 64 x 256
    conv4 = Conv2D(512, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #bach_size x 64 x 64 x 512
    conv4 = Conv2D(512, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #bach_size x 64 x 64 x 512
    drop4 = Dropout(0.5)(conv4)
    
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    #conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    #conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #drop5 = Dropout(0.5)(conv5)

    #up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    #merge6 = concatenate([drop4,up6], axis = 3)
    #conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    
    g1=UnetGatingSignal(drop4)
    #bach_size x 64 x 64 x 1024
    att1=AttnGatingBlock(conv3, g1, 512)
    #bach_size x 128 x 128 x 256
    up7 = Conv2D(256, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))#(conv6))
    #bach_size x 128 x 128 x 256
    merge7 = concatenate([att1,up7], axis = 3)
    #bach_size x 128 x 128 x 512
    conv7 = Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    #bach_size x 128 x 128 x 256
    conv7 = Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #bach_size x 128 x 128 x 256
    
    g2=UnetGatingSignal(conv7)
    #bach_size x 128 x 128 x 512
    att2=AttnGatingBlock(conv2, g2, 256)
    #bach_size x 256 x 256 x 128

    up8 = Conv2D(128, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    #bach_size x 256 x 256 x 128
    merge8 = concatenate([att2,up8], axis = 3)
    #bach_size x 256 x 256 x 256
    conv8 = Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    #bach_size x 256 x 256 x 128
    conv8 = Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    #bach_size x 256 x 256 x 128
    
    g3=UnetGatingSignal(conv8)
    #bach_size x 256 x 256 x 256
    att3=AttnGatingBlock(conv1, g3, 128)
    #bach_size x 512 x 512 x 64

    up9 = Conv2D(64, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    #bach_size x 512 x 512 x 64
    merge9 = concatenate([att3,up9], axis = 3)
     #bach_size x 512 x 512 x 128
    conv9 = Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge9)
     #bach_size x 512 x 512 x 64
    conv9 = Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv9)
     #bach_size x 512 x 512 x 64
    conv9 = Conv2D(2, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv9)
     #bach_size x 512 x 512 x 2
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
     #bach_size x 512 x 512 x 1

    model = Model(inputs = inputs, outputs = conv10)
    
    if optimizer=='adam':
        OPT=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000, amsgrad=False)
        print('adam optimizer')
    elif optimizer=='sgd':
        OPT=keras.optimizers.SGD(lr=0.01,momentum=0.8,nesterov=True)
        print('SGD optimizer')
    model.compile(optimizer = OPT, loss = loss, metrics = metrics)
    

    return model

