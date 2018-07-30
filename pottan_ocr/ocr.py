#!/usr/bin/env python3

import multiprocessing
import sys
import argparse
from PIL import Image

from os.path import splitext
import torch
from torch.autograd import Variable
import numpy as np

from pottan_ocr import string_converter as converter
from pottan_ocr import utils
from pottan_ocr import model as crnn
from pottan_ocr.dataset import normaizeImg

imageHeight = utils.config['imageHeight']

def loadImg( fname ):
    img = Image.open( fname ).convert('L')
    origW, origH = img.size
    targetH = imageHeight
    targetW = int( origW * targetH/origH )
    img = img.resize( (targetW, targetH ), Image.BILINEAR )
    img = np.array( img )
    return normaizeImg( img  )

def evalModel( img_path, current, total ):
    global crnnModel
    print( 'Progress %d/%d' % ( current, total ), file=sys.stderr )
    image = loadImg( img_path )
    image = image.unsqueeze(0)
    image = Variable(image)
    preds = crnnModel(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    return converter.decode(preds.data, preds_size.data, raw=False)

def threadInitializer( opt ):
    global crnnModel
    crnnModel = crnn.CRNN( imageHeight, 1, converter.totalGlyphs, opt.nh )
    utils.loadTrainedModel( crnnModel, opt )
    crnnModel.eval()

def main( opt ):
    totalImages = len( opt.image_paths )
    pool = multiprocessing.Pool( multiprocessing.cpu_count()-1, initializer=threadInitializer, initargs=( opt, ) )
    results = [ pool.apply_async( evalModel, ( img_path, idx, totalImages ) ) for idx, img_path in enumerate( opt.image_paths ) ]
    for idx, result in enumerate( results ):
        img_path = opt.image_paths[ idx ]
        text = result.get()[0]
        if( opt.stdout ):
            print('%s:::%s' %( img_path, [ text ] ))
        else:
            fname = '%s.txt'% splitext( img_path )[0]
            utils.writeFile( fname, text )

def findNHFromFile( opt ):
    if( opt.cuda ):
        stateDict = torch.load(opt.crnn )
    else:
        stateDict = torch.load(opt.crnn, map_location={'cuda:0': 'cpu'} )
    nh = stateDict['module.rnn.0.embedding.bias'].shape[0]
    print( "Number of hidden layers ( --nh ) = %d" % nh )
    return nh


if( __name__ == '__main__' ):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_paths', help='Path image file', nargs="+" )
    parser.add_argument('--crnn', required=True, help="path to pre trained model state ( .pth file)")
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--stdout', action='store_true', help='Write output to stdout instead of saving it as <imgfile>.txt')
    opt = parser.parse_args()
    opt.nh = findNHFromFile( opt )
    main( opt )
