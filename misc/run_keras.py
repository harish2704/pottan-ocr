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
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--outdir', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
opt = parser.parse_args()
print(opt)


from misc import keras_model
from keras.optimizers import SGD, RMSprop
from keras.layers import Input, Lambda
from keras import backend as K, Model
from keras.utils import Sequence
from pottan_ocr.utils import config, readJson
from pottan_ocr import string_converter as converter
from pottan_ocr.dataset import TextDataset, normalizeBatch
import numpy as np
from keras.utils import plot_model
from os import environ

if 'DEBUG' in environ:
    from tensorflow.python import debug as tf_debug
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)

targetW = config['trainImageWidth']
targetH = config['imageHeight']
batchSize = opt.batchSize
m = keras_model.KerasCrnn()

plot_model(m, to_file='model.png', show_shapes=True)
#  m.summary()
outputSize = m.layers[-1].output.shape[1].value
MG = '/home/hari/tmp/ocr-related/keras-js/demos/data/test2.jpg'
TRAINED_TORCH_MODEL =  '/home/hari/tmp/ocr-related/trained_models/netCRNN_07-25-04-45-02_0.pth'


#  sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
sgd = RMSprop( lr=0.001, epsilon=K.epsilon() )
labelWidth = 95

class DataGenerator( Sequence ):
    def __init__( self, txtFile, **kwargs):
        self.ds = TextDataset( txtFile, **kwargs )

    def __len__(self):
        return self.ds.__len__()

    def __getitem__( self, batchIndex ):
        unNormalized =  self.ds.getUnNormalized( batchIndex )
        images, labels = normalizeBatch( unNormalized, channel_axis=2 )
        labels, label_lengths  = converter.encodeStrListRaw( labels, labelWidth )
        #  print( labels )
        input_lengths = [ labelWidth ] * batchSize
        inputs = {
                'the_images': images,
                'the_labels': np.array( labels ),
                'label_lengths': np.array( label_lengths ),
                'input_lengths': np.array( input_lengths ),
                }
        outputs = {'ctc': np.zeros([ batchSize ])}  # dummy data for dummy loss function
        return (inputs, outputs)

def ctc_lambda_func( args ):
    y_pred, labels, y_pred_len, label_lengths = args
    return K.ctc_batch_cost( labels, K.softmax( y_pred ), y_pred_len, label_lengths )

labels = Input(name='the_labels', shape=[ labelWidth ], dtype='int32')
images = Input(name='the_images', shape=[ targetH, targetW, 1 ], dtype='float32')
label_lengths = Input(name='label_lengths', shape=[1], dtype='int32')
y_pred_len = Input(name='input_lengths', shape=[1], dtype='int32')

y_pred = m( images )
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')( [ y_pred, labels, y_pred_len, label_lengths ])
mm = Model( inputs=[ images, labels, label_lengths, y_pred_len ], outputs=loss_out )
plot_model(mm, to_file='model2.png', show_shapes=True)

mm.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd,  metrics=['accuracy'])


train_loader = DataGenerator( opt.traindata, batchSize=opt.batchSize, limit=opt.traindata_limit, cache=opt.traindata_cache )
test_loader = DataGenerator( opt.valdata, batchSize=opt.batchSize, limit=opt.valdata_limit, cache=opt.valdata_cache )

mm.fit_generator(generator=train_loader,
                    steps_per_epoch=int(opt.traindata_limit/batchSize),
                    epochs=opt.niter,
                    validation_data=test_loader,
                    validation_steps=int(opt.traindata_limit/batchSize),
                    initial_epoch=0)

#  def toArr( x ):
    #  return K.variable( np.array( x ))

#  sampleStr= readJson('/home/hari/tmp/ocr-related/pred_test/str.txt')
#  yP = np.load('/home/hari/tmp/ocr-related/pred_test/pred.npz')
#  yT, yTLen = converter.encodeStrListRaw( [ sampleStr ] , 81 )

#  yT = toArr( yT )
#  yTLen = toArr( [ yTLen ] )
#  yP = toArr( yP )
#  yPLen = toArr( [ [ yP.shape[1].value ] ] )



#  for i in ( yT, yP, yPLen, yTLen ):
    #  print( i.shape )
#  yP = K.softmax( yP )
#  cost = K.eval( K.ctc_batch_cost( yT, yP, yPLen, yTLen ) )
