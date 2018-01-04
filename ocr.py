#!/usr/bin/env python3

from PIL import Image

import torch
from torch.autograd import Variable
import numpy as np
import utils
from dataset import normaizeImg
import argparse
import models.crnn as crnn
import string_converter as converter

def loadImg( fname ):
    img = Image.open( fname ).convert('L')
    origW, origH = img.size
    targetH = 32
    targetW = int( origW * targetH/origH )
    img = img.resize( (targetW, targetH ), Image.BILINEAR )
    img = np.expand_dims( img, axis=0 )
    return normaizeImg( img )

def evalModel( model, image ):
    model.eval()
    image = image.view(1, *image.size())
    #  import ipdb; ipdb.set_trace()
    image = Variable(image)
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))

def main( opt ):
    img = loadImg( opt.image_path )
    model = crnn.CRNN(32, 1, converter.totalGlyphs, 256)
    utils.loadTrainedModel( model, opt )
    evalModel( model, img )

if( __name__ == '__main__' ):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path image file')
    parser.add_argument('--crnn', required=True, help="path to pre trained model state ( .pth file)")
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    opt = parser.parse_args()
    main( opt )
