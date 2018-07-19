
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, MaxPooling2D, TimeDistributed, Bidirectional, LSTM, Conv2D, ZeroPadding2D, Permute, Reshape
K.set_image_data_format('channels_last')
from pottan_ocr import string_converter as converter



def KerasCrnn(imgH=32, nc=1, nclass=converter.totalGlyphs, nh=64 ):

    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 0]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]

    cnn = Sequential()

    def convRelu(i, batchNormalization=False ):
        nIn = nc if i == 0 else nm[i - 1]
        nOut = nm[i]
        padding = 'same' if ps[i] else 'valid'

        if( i == 0):
            cnn.add( Conv2D( nOut, ks[i], strides=ss[i], input_shape=( 32, None, 1 ), padding=padding, name='conv{0}'.format(i) ) )
        else:
            cnn.add( Conv2D( nOut, ks[i], strides=ss[i], padding=padding, name='conv{0}'.format(i) ) )
        if batchNormalization:
            cnn.add( BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5, name='batchnorm{0}'.format(i) ))
        cnn.add( Activation('relu', name='relu{0}'.format(i) ))

    convRelu(0)
    cnn.add( MaxPooling2D( pool_size=2, strides=2, name='pooling{0}'.format(0) ) )  # 64x16x64
    convRelu(1)
    cnn.add( MaxPooling2D( pool_size=2, strides=2, name='pooling{0}'.format(1) ) )  # 128x8x32
    convRelu(2, True)
    convRelu(3)
    cnn.add( ZeroPadding2D( padding=(0,1) ))
    cnn.add( MaxPooling2D( pool_size=(2, 2), strides=(2, 1), padding='valid', name='pooling{0}'.format(2) ) )  # 256x4x16
    convRelu(4, True)
    convRelu(5)
    cnn.add( ZeroPadding2D( padding=(0,1) ))
    cnn.add( MaxPooling2D( pool_size=(2, 2), strides=(2, 1), padding='valid', name='pooling{0}'.format(3) ) )  # 512x2x16
    convRelu(6, True )  # 512x1x16

    cnn.add(Reshape((-1, 512)))
    cnn.add(Bidirectional( LSTM(64 , return_sequences=True, use_bias=True, recurrent_activation='sigmoid', )) )
    cnn.add( TimeDistributed( Dense(64) ) )
    cnn.add(Bidirectional( LSTM(64 , return_sequences=True, use_bias=True, recurrent_activation='sigmoid', )) )
    cnn.add( TimeDistributed(Dense( nclass ) ) )

    cnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta())
    return cnn


