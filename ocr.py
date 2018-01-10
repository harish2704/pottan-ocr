#!/usr/bin/env python3

from PIL import Image

from os.path import splitext
import torch
from torch.autograd import Variable
import numpy as np
import utils
from dataset import normaizeImg
import argparse
import models.crnn as crnn
import string_converter as converter
import multiprocessing
import sys

def loadImg( fname ):
    img = Image.open( fname ).convert('L')
    origW, origH = img.size
    targetH = 32
    targetW = int( origW * targetH/origH )
    img = img.resize( (targetW, targetH ), Image.BILINEAR )
    img = np.expand_dims( img, axis=0 )
    return normaizeImg( img )

def evalModel( model, img_path, current, total ):
    print( 'Progress %d/%d' %( current, total ), file=sys.stderr )
    image = loadImg( img_path )
    image = image.view(1, *image.size())
    image = Variable(image)
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    return converter.decode(preds.data, preds_size.data, raw=False)

def main( opt ):
    model = crnn.CRNN(32, 1, converter.totalGlyphs, 256)
    utils.loadTrainedModel( model, opt )
    model.eval()
    totalImages = len( opt.image_paths )
    pool = multiprocessing.Pool( multiprocessing.cpu_count() )
    results = [ pool.apply_async( evalModel, ( model, img_path, idx, totalImages ) ) for idx, img_path in enumerate( opt.image_paths ) ]
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
    parser.add_argument('--stdout', action='store_true', help='Write output to stdout instead of saving it as <imgfile>.txt')
    opt = parser.parse_args()
    main( opt )
