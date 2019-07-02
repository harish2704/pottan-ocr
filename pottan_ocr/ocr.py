#!/usr/bin/env python

#  File name: pottan_ocr/ocr.py
#  Author: Harish.K<harish2704@gmail.com>
#  Copyright 2019 Harish.K<harish2704@gmail.com>
#  Date created: Sun Jun 30 2019 20:16:35 GMT+0530 (IST)
#  Date last modified: Sun Jun 30 2019 20:16:35 GMT+0530 (IST)
#  Python Version: 3.x

from PIL import Image
import argparse
import numpy as np
from keras import models
from os.path import splitext

from pottan_ocr.utils import config, readJson
from pottan_ocr.dataset import normaizeImg
from pottan_ocr.string_converter import decodeStr
from pottan_ocr import utils

imageHeight = config['imageHeight'] - 3

def loadImg( fname ):
    img = Image.open( fname ).convert('L')
    origW, origH = img.size
    targetH = imageHeight
    targetW = int( origW * targetH/origH )
    img = img.resize( (targetW, targetH ), Image.BILINEAR )
    # channel is set to 2 for the compatibility with old torch model.
    return normaizeImg( np.array( img ), 2 )

def main( opt ):
    model = models.load_model( opt.crnn )
    totalImages = len( opt.image_paths )
    images = [ loadImg( i ) for i in opt.image_paths ]
    maxWidth = max([i.shape[1] for i in images ])
    images = [ np.pad( i, [(1, 2), (0, maxWidth - i.shape[1] ), (0,0)], mode='constant', constant_values=1) for i in images ]
    images = np.array( images )
    out = model.predict( images )
    out = out.argmax(2)
    textResults = [ decodeStr( i, raw=False ) for i in out ]
    #  import pdb; pdb.set_trace();
    #  import IPython as x; x.embed()
    for img_path, text in zip( opt.image_paths, textResults ):
        if( opt.stdout ):
            print('%s:::%s' %( img_path, [ text ] ))
        else:
            fname = '%s.txt'% splitext( img_path )[0]
            utils.writeFile( fname, text )


if( __name__ == '__main__' ):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_paths', help='Path image file', nargs="+" )
    parser.add_argument('--crnn', required=True, help="path to pre trained model ( Keras saved model )")
    parser.add_argument('--stdout', action='store_true', help='Write output to stdout instead of saving it as <imgfile>.txt')
    opt = parser.parse_args()
    main( opt )
