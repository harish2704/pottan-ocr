import sys
sys.path.append('.')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--traindata', required=True, help='path to dataset')
parser.add_argument('--traindata_limit', type=int, default=51200, help='Limit the training dataset size')
parser.add_argument('--traindata_cache', default=None, help='Cache Directory for caching generated train data')
parser.add_argument('--valdata', required=True, help='path to dataset')
parser.add_argument('--valdata_limit', type=int, default=4096, help='Limit the validation dataset size')
parser.add_argument('--valdata_cache', default=None, help='Cache Directory for caching generated validation data')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nh', type=int, default=64, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
parser.add_argument('--crnn', help="path to crnn (to continue training)")
parser.add_argument('--outfile', default='./crnn.h5', help='Where to store samples and models')
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
from pottan_ocr.dataset import TextDataset, normalizeBatch
import numpy as np
from keras.utils import plot_model
from os import environ

#Maximum length of a sample text line
labelWidth = 120
targetW = config['trainImageWidth']
targetH = config['imageHeight']
batchSize = opt.batchSize
backBone = KerasCrnn( nh=opt.nh )


if( opt.crnn ):
    backBone_ = keras.models.load_model( opt.crnn )
    backBone.set_weights( backBone_.get_weights() )
    print( 'Loaded "%s"' % opt.crnn )

backBone.summary()
#  plot_model( backBone, to_file='model.png', show_shapes=True)


if( opt.adadelta ):
    optimizer = Adadelta()
elif( opt.adam ):
    optimizer = Adam( lr=opt.lr  )
else:
    optimizer = RMSprop( lr=opt.lr )

class DataGenerator( Sequence ):
    def __init__( self, txtFile, **kwargs):
        self.ds = TextDataset( txtFile, **kwargs )

    def __len__(self):
        return self.ds.__len__()

    def __getitem__( self, batchIndex ):
        unNormalized =  self.ds.getUnNormalized( batchIndex )
        images, labels = normalizeBatch( unNormalized, channel_axis=2 )
        labels, label_lengths  = converter.encodeStrListRaw( labels, labelWidth )
        inputs = {
                'the_images': images,
                'the_labels': np.array( labels ),
                'label_lengths': np.array( label_lengths ),
                }
        outputs = {'ctc': np.zeros([ batchSize ])}  # dummy data for dummy loss function
        return (inputs, outputs)

def ctc_lambda_func( args ):
    y_pred, labels, label_lengths = args
    y_pred_len = [ [ int(y_pred.shape[1]) ] ] * batchSize
    #  y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost( labels, K.softmax( y_pred ), y_pred_len, label_lengths )

labels = Input(name='the_labels', shape=[ labelWidth ], dtype='int32')
images = Input(name='the_images', shape=[ targetH, targetW, 1 ], dtype='float32')
label_lengths = Input(name='label_lengths', shape=[1], dtype='int32')

y_pred = backBone( images )
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')( [ y_pred, labels, label_lengths ])
fullModel = Model( inputs=[ images, labels, label_lengths ], outputs=loss_out )
#  plot_model(fullModel, to_file='model2.png', show_shapes=True)

fullModel.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer,  metrics=['accuracy'])


train_loader = DataGenerator( opt.traindata, batchSize=opt.batchSize, limit=opt.traindata_limit, cache=opt.traindata_cache )
test_loader = DataGenerator( opt.valdata, batchSize=opt.batchSize, limit=opt.valdata_limit, cache=opt.valdata_cache )

#  import pdb; pdb.set_trace();
#  import IPython as x; x.embed()

class WeightsSaver(Callback):
    def __init__(self):
        self.fname = opt.outfile + '_' + datetime.now().strftime('%d%m%Y_%H%M%S')
        self.i = 1
        self.j = 1
    def on_batch_end(self, epoch, logs={}):
        if(self.j%100 == 0):
            name = '%s_%i.h5' % (self.fname, self.i)
            backBone.save( name )
            print( 'Saved "%s"' % name)
            self.i = self.i + 1
        self.j = self.j + 1

#  from keras.callbacks import  TensorBoard
#  tensorboardLogs = TensorBoard( update_freq='batch' )

model_saver = WeightsSaver()
fullModel.fit_generator(generator=train_loader,
                    steps_per_epoch=int(opt.traindata_limit/batchSize),
                    epochs=opt.niter,
                    validation_data=test_loader,
                    validation_steps=int(opt.valdata_limit/batchSize),
                    initial_epoch=0, callbacks=[ model_saver ])
