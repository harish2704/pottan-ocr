import json
import cv2
import numpy as np
from torch.autograd import Variable

FINAL_W=32
FINAL_H=32

def readFile( fname ):
    with open( fname, 'r') as f:
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


