#!/usr/bin/env python

from collections import namedtuple
import torch
import torch.nn as nn
from torch.autograd import Variable as Variable

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, MaxPooling2D, TimeDistributed, Input, Embedding, Bidirectional, LSTM
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Permute, Lambda, Reshape

from pottan_ocr.model import CRNN as CRNNT
from misc.keras_model import KerasCrnn as CRNNK
from pottan_ocr import string_converter as converter
from pottan_ocr import utils

dictToObj = lambda x: namedtuple('Struct', x.keys() )( *x.values() )
opt = dictToObj({
    'crnn': '/home/hari/Downloads/netCRNN_01-19-06-09-54_3.pth',
    'cuda': False,
    'nh': 64
    })


Tcrnn = CRNNT(32, 1, converter.totalGlyphs, opt.nh )
utils.loadTrainedModel( Tcrnn, opt )
Tcrnn.eval()

Kcrnn = CRNNK( imgH=32, nc=1, nclass=converter.totalGlyphs, nh=opt.nh )

layerMapping = {
        'cnn.batchnorm2': 'batchnorm2',
        'cnn.batchnorm4': 'batchnorm4',
        'cnn.batchnorm6': 'batchnorm6',
        'cnn.conv0': 'conv0',
        'cnn.conv1': 'conv1',
        'cnn.conv2': 'conv2',
        'cnn.conv3': 'conv3',
        'cnn.conv4': 'conv4',
        'cnn.conv5': 'conv5',
        'cnn.conv6': 'conv6',
        'rnn.0.embedding': 'time_distributed_1',
        'rnn.0.rnn' : 'bidirectional_1',
        'rnn.1.embedding' : 'time_distributed_2',
        'rnn.1.rnn': 'bidirectional_2'
        }

torchLayerMap = dict( Tcrnn.named_modules() )

def layerNameFromParamKey( paramKey ):
    return paramKey[:paramKey.rindex('.')]


def LSTM( tLayer, kLayer ):
    if( tLayer.bidirectional != True ):
        raise NotImplemented('Sorry')

    stateDict = dictToObj( tLayer.state_dict() )
    newKparams = [
           stateDict.weight_ih_l0.transpose(1,0),
           stateDict.weight_hh_l0.transpose(1,0),
           stateDict.bias_ih_l0                     + stateDict.bias_hh_l0,
           stateDict.weight_ih_l0_reverse.transpose(1,0),
           stateDict.weight_hh_l0_reverse.transpose(1,0),
           stateDict.bias_ih_l0_reverse             + stateDict.bias_hh_l0_reverse,
           ]
    kLayer.set_weights( newKparams )

def Linear( tLayer, kLayer ):
    stateDict = dictToObj( tLayer.state_dict() )
    newKparams = [
            stateDict.weight.transpose(1,0).numpy(),
            stateDict.bias.numpy(),
            ]
    kLayer.set_weights( newKparams )

def BatchNorm2d( tLayer, kLayer ):
    newKparams = [ i.numpy() for i in tLayer.state_dict().values() ]
    kLayer.set_weights( newKparams )

def Conv2d( tLayer, kLayer ):
    newKparams = [ i.numpy() for i in tLayer.state_dict().values() ]
    newKparams[0] = newKparams[0].transpose( 2,3,1,0 )
    kLayer.set_weights( newKparams )

transferFunctions = {
        'Conv2d': Conv2d,
        'LSTM': LSTM,
        'Linear': Linear,
        'BatchNorm2d': BatchNorm2d
        }

torchStateDict = Tcrnn.state_dict()
for torchLayerName in set( [ layerNameFromParamKey(i) for i in torchStateDict ]):
    torchLayer = torchLayerMap[ torchLayerName ]
    kerasLayerName = layerMapping[ torchLayerName ]
    try:
        kerasLayer = Kcrnn.get_layer( name=kerasLayerName )
    except Exception as e:
        print( e )
        continue
    print( 'Transfering %s ---> %s' %( torchLayerName, kerasLayerName ) )

    torchLayerType = type( torchLayer ).__name__
    transferFunctions[ torchLayerType ]( torchLayer, kerasLayer )



from pottan_ocr.ocr import loadImg
from pottan_ocr import utils
Tip = loadImg('/home/hari/tmp/ocr-related/keras-js/demos/data/test2.jpg').unsqueeze(0)

allTmods =[ [ i, name ] for name,i in Tcrnn.named_modules() if len(list( i.modules() )) == 1 ]
cnnTmods =allTmods[:21]
tin = Variable(Tip)
outs = []
for mod, name in cnnTmods:
    print( 'Running %s' % name )
    tin = mod( tin )
    outs.append( [tin, name ])

b, c, h, w = tin.size()
assert h == 1, "the height of output must be 1"
tin = tin.squeeze(2)
tin = tin.permute(2, 0, 1)  # [w, b, c]
for i in range(2):
    lstm, lstmName = allTmods[ 21 + (i*2) ]
    linear, linearName = allTmods[ 22 + (i*2) ]
    tin, _ = lstm( tin )
    outs.append( [tin, lstmName ])

    T, b, h = tin.size()
    tin = tin.view(T * b, h)

    tin = linear(tin)  # [T * b, nOut]
    tin = tin.view(T, b, -1)
    outs.append( [tin, linearName ])


jsonOut = [];
for data, name in outs:
    jsonOut.append({
        'name': name,
        'data': data.tolist(),
        'shape': data.shape
        })

import json
utils.writeFile( './tdebug.json', json.dumps( jsonOut ) )



#  TcnnBig = nn.Sequential( *list( list( Tcrnn.children() )[0].children() )[:] )
#  TcnnBig.eval()
#  KcnnBig = Sequential( Kcrnn.layers[:] )
TcnnBig = Tcrnn
KcnnBig = Kcrnn
ToutBig = TcnnBig( Variable(Tip) ).data
KoutBig = torch.from_numpy( KcnnBig.predict( Tip.permute(0,2,3,1).numpy() ) )

#  TcnnA = nn.Sequential( *list( list( Tcrnn.children() )[0].children() )[:7] )
#  KcnnA = Sequential( Kcrnn.layers[:10] )
#  ToutA = TcnnA( Variable(Tip) ).data.numpy()
#  KoutA = KcnnA.predict( Tip.numpy() )

#  TcnnB = nn.Sequential( *list( list( Tcrnn.children() )[0].children() )[7:8] )
#  def genModel( lay ):
    #  ip = Input( shape=lay.input_shape[1:] )
    #  op = lay( ip )
    #  model = Model( inputs=ip, outputs=op )
    #  return model
#  KcnnB = genModel( Kcrnn.layers[10] )
#  ToutB = TcnnB( Variable( torch.from_numpy( ToutA ) ) ).data.numpy()
#  KoutB = KcnnB.predict( KoutA )

diff = KoutBig - ToutBig.permute( 1,0,2 )
print( 'Max: ', diff.max() )
