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

def loadImg( fname ):
    img = Image.open( fname ).convert('L')
    origW, origH = img.size
    targetH = 32
    targetW = int( origW * targetH/origH )
    img = img.resize( (targetW, targetH ), Image.BILINEAR )
    return normaizeImg( np.array( img ) )

def evalModel( img_path, current, total ):
    global crnnModel
    print( 'Progress %d/%d' %( current, total ), file=sys.stderr )
    image = loadImg( img_path )
    image = image.view(1, *image.size())
    image = Variable(image)
    preds = crnnModel(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    return converter.decode(preds.data, preds_size.data, raw=False)

def threadInitializer( opt ):
    global crnnModel
    crnnModel = crnn.CRNN(32, 1, converter.totalGlyphs, opt.nh )
    utils.loadTrainedModel( crnnModel, opt )
    crnnModel.eval()

def main( opt ):
    totalImages = len( opt.image_paths )
    pool = multiprocessing.Pool( multiprocessing.cpu_count(), initializer=threadInitializer, initargs=( opt, ) )
    results = [ pool.apply_async( evalModel, ( img_path, idx, totalImages ) ) for idx, img_path in enumerate( opt.image_paths ) ]
    for idx, result in enumerate( results ):
        img_path = opt.image_paths[ idx ]
        text = result.get()[0]
        if( opt.stdout ):
            print('%s:::%s' %( img_path, [ text ] ))
        else:
            fname = '%s.txt'% splitext( img_path )[0]
            utils.writeFile( fname, text )


if( __name__ == '__main__' ):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_paths', help='Path image file', nargs="+" )
    parser.add_argument('--crnn', required=True, help="path to pre trained model state ( .pth file)")
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--stdout', action='store_true', help='Write output to stdout instead of saving it as <imgfile>.txt')
    opt = parser.parse_args()
    main( opt )
