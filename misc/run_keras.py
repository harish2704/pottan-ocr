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

if len( sys.argv ) > 1 and sys.argv[1] == '--use-plaidml':
    import plaidml.keras
    plaidml.keras.install_backend()

from misc import keras_model
from keras.optimizers import SGD
from keras import backend as K
from pottan_ocr import string_converter as converter
from pottan_ocr.dataset import TextDataset

batchSize = opt.batchSize
m = keras_model.KerasCrnn()

m.summary()

sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

def ctc_lambda_func( y_true, y_pred ):
    #  import ipdb; ipdb.set_trace()
    if y_true.shape[0].value:
        import ipdb; ipdb.set_trace()
        y_true_encoded, y_true_len  = converter.encode( y_true )
        y_pred_len = [y_pred.shape[1].value] * batchSize
        return K.ctc_batch_cost( y_true_encoded, y_pred, y_pred_len, y_true_len )
    return y_pred


m.compile(loss=ctc_lambda_func, optimizer=sgd)

train_loader = TextDataset( opt.traindata, batchSize=opt.batchSize, limit=opt.traindata_limit, cache=opt.traindata_cache )
test_loader = TextDataset( opt.valdata, batchSize=opt.batchSize, limit=opt.valdata_limit, cache=opt.valdata_cache )
import ipdb; ipdb.set_trace()
print( "\n\n Valdata len: %d" % len( test_loader ))

m.fit_generator(generator=train_loader[0],
                    steps_per_epoch=3,
                    epochs=2,
                    validation_data=test_loader[0],
                    validation_steps=2,
                    initial_epoch=0)
