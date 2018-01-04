import torch
import json
import cv2
import numpy as np
from torch.autograd import Variable
import gzip

FINAL_W=32
FINAL_H=32

def readFile( fname ):
    opener, mode = ( gzip.open, 'rt' ) if fname[-3:] == '.gz' else ( open, 'r' )
    with opener( fname, mode ) as f:
        return f.read()

def readJson( fname ):
    with open( fname, 'r' ) as f:
        return json.load( f )

def writeFile( fname, contents ):
    with open( fname, 'w' ) as f:
        f.write( contents )

def writeJson( fname, data ):
    with open( fname, 'w') as outfile:
        json.dump(data, outfile)

def fixedWidthImg( img ):
    ( currentH, currentW) = img.shape
    if( currentH > currentW ):
        newWidth = int( currentW * ( FINAL_H/currentH ) )
        newHeigh = FINAL_H
        offsetY=0
        offsetX = int( ( FINAL_W - newWidth )/2 )
    else:
        newHeigh = int( currentH * ( FINAL_W/currentW ) )
        newWidth = FINAL_W
        offsetX=0
        offsetY = int( ( FINAL_H - newHeigh )/2 )
    resizedImg = cv2.resize( img, ( newWidth, newHeigh ), interpolation=cv2.INTER_CUBIC )
    canvasImg = np.ones( ( FINAL_H, FINAL_W ), dtype=np.uint8)*255
    canvasImg[offsetY:offsetY+newHeigh, offsetX:offsetX+newWidth] = resizedImg
    return canvasImg




class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def loadTrainedModel( crnn, opt ):
    print('loading pretrained model from %s' % opt.crnn)
    if( opt.cuda ):
        stateDict = torch.load(opt.crnn )
    else:
        stateDict = torch.load(opt.crnn, map_location={'cuda:0': 'cpu'} )

    #  Handle the case of some old torch version. It will save the data as module.<xyz> . Handle it
    if( list( stateDict.keys() )[0][:7] == 'module.' ):
        for key in list(stateDict.keys()): 
            stateDict[ key[ 7:] ] = stateDict[key]
            del stateDict[ key ]
    crnn.load_state_dict( stateDict )
    print('Completed loading pre trained model')
