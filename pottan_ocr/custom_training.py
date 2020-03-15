#!/usr/bin/env python

#  ഓം ബ്രഹ്മാർപ്പണം
#
#  File name: pottan_ocr/custom_training.py
#  Author: Harish.K<harish2704@gmail.com>
#  Copyright 2020 Harish.K<harish2704@gmail.com>
#  Date created: Sun Mar 15 2020 20:09:25 GMT+0530 (GMT+05:30)
#  Date last modified: Sun Mar 15 2020 20:09:25 GMT+0530 (GMT+05:30)
#  Python Version: 3.x


import sys
sys.path.append('.')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--traindata', required=True, help='path to dataset')
parser.add_argument('--nh', type=int, default=64, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--repeat', type=int, default=10, help='number of time to repeat an image by randomly padding top/ bottom padding')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
parser.add_argument('--model', help="path to trained model (to continue training)")
parser.add_argument('--outfile', default='./pottan_ocr.h5', help='Where to store samples and models')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt = parser.parse_args()
print(opt)

import tensorflow as tf
import keras
from datetime import datetime
from keras import models
from keras.callbacks import Callback
from pottan_ocr.model import KerasCrnn
from keras.optimizers import RMSprop, Adadelta, Adam
from keras.layers import Input, Lambda
from keras import backend as K, Model
from keras.utils import Sequence
from pottan_ocr.utils import config, readJson
from pottan_ocr import string_converter as converter
from pottan_ocr.custom_dataset import CustomDataset
import numpy as np
from keras.utils import plot_model
from os import environ

imageHeight = config['imageHeight']
backBone = KerasCrnn( nh=opt.nh )
train_loader = CustomDataset( opt.traindata, backBone, line_height=imageHeight, repeat=opt.repeat )

if( opt.model ):
    backBone_ = keras.models.load_model( opt.model )
    backBone.set_weights( backBone_.get_weights() )
    backBone.summary()
    print( 'Loaded "%s"' % opt.model )

#  plot_model( backBone, to_file='model.png', show_shapes=True)


if( opt.adadelta ):
    optimizer = Adadelta()
elif( opt.adam ):
    optimizer = Adam( lr=opt.lr  )
else:
    optimizer = RMSprop( lr=opt.lr )


def ctc_lambda_func( args ):
    prediction, labels, prediction_lengths, label_lengths = args
    #  prediction = prediction[:, 2:, :]
    return K.ctc_batch_cost( labels, K.softmax( prediction ), prediction_lengths, label_lengths )

images = Input(name='the_images', shape=[ imageHeight, None, 1 ], dtype='float32')
labels = Input(name='the_labels', shape=[ train_loader.max_text_length ], dtype='int32')
prediction_lengths = Input(name='prediction_lengths', shape=[1], dtype='int32')
label_lengths = Input(name='label_lengths', shape=[1], dtype='int32')

prediction = backBone( images )
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')( [ prediction, labels, prediction_lengths, label_lengths ])
fullModel = Model( inputs=[ images, labels, prediction_lengths, label_lengths ], outputs=loss_out )

fullModel.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer,  metrics=['accuracy'])



class WeightsSaver(Callback):
    def __init__(self):
        self.fname = opt.outfile + '_' + datetime.now().strftime('%d%m%Y_%H%M%S')
        self.i = 1
        self.j = 1
        self.saveInterval = 100 if 'SAVE_INTERVAL' not in environ else int( environ['SAVE_INTERVAL'])

    def save(self):
        name = '%s_%i.h5' % (self.fname, self.i)
        backBone.save( name )
        print( 'Saved "%s"' % name)
        self.i = self.i + 1

    def on_epoch_end(self, epoch, logs={}):
        self.save()

    def on_batch_end(self, epoch, logs={}):
        if(self.j%self.saveInterval == 0):
            self.save()
        self.j = self.j + 1

#  from keras.callbacks import  TensorBoard
#  tensorboardLogs = TensorBoard( update_freq='batch' )

model_saver = WeightsSaver()
fullModel.fit_generator(generator=train_loader,
                    epochs=opt.niter,
                    initial_epoch=0, callbacks=[ model_saver ])

