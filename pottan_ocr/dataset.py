#!/usr/bin/env python3

from .utils import config, readLines, writeFile

import torch
from torch.utils.data import Dataset

import os
import math
from enum import Enum
from collections import Sequence
from random import choice, sample
from re import split
import numpy as np
from numpy.random import normal
from PIL import Image
import cairo
import gi
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo


#  Text alignment variations
VARIATIONS = Enum('AlignmentVariation', 'random alighn_bottom fit_height' )

fontList = config['fonts']
fontListFlat = []
for fnt, styles in fontList:
    for style in styles:
        fontDescStr = '%s %s 16' %( fnt, style )
        fontListFlat.append([ fontDescStr, VARIATIONS.random,       ])
        fontListFlat.append([ fontDescStr,  VARIATIONS.alighn_bottom ])
        fontListFlat.append([ fontDescStr,  VARIATIONS.fit_height ])

totalVariations = len(fontListFlat)


#  Pango FontDescription cache
fontDescCache = {};


#  List of available angles of rotation.
#  -1 --> Maximum possible negative rotation with in available free space
#  +1 --> Maximum possible positive rotation with in available free space
TWIST_CHOICES = [ -1, -0.95, -0.85, 0.85, 0.95, 1 ]

noiseSDChoices = [
        #  ( mean, standard-deviation ) of normal distribution
        (0,0),
        (10, 50),
        (12, 60),
        (15, 70),
        (20, 75)
        ]

targetW = 1024
#  let Canvas have some extra space than required so that, we can handle text overflow easly
canvasWidth = targetW*2
targetH = 32


##
#
# @param text - The text to be rendered
# @param font - A string representing Pango font description. Eg: 'FreeSans bold 18'
# @param variation - representing alighnment variation .
#
# @return numpy.array
def renderText( text, font, variation ):
    """Render a unicode text into 32xH image and return a numpy array"""

    surface = cairo.ImageSurface(cairo.FORMAT_A8, canvasWidth, targetH )
    context = cairo.Context(surface)
    pc = PangoCairo.create_context(context)
    layout = PangoCairo.create_layout(context)
    if( font in fontDescCache ):
        fontDesc = fontDescCache[font]
    else:
        fontDesc = fontDescCache[font] = Pango.font_description_from_string( font )
    layout.set_font_description(Pango.FontDescription( font ))
    layout.set_text( text, -1 );

    inkRect, _ = layout.get_pixel_extents()
    actualW = inkRect.width; actualH = inkRect.height

    if( variation == VARIATIONS.fit_height ):
        #  Fit-height is done by varying font size
        fontDesc = layout.get_font_description()
        fontDesc.set_size( int(fontDesc.get_size() * 0.9 * targetH/ actualH ) )
        layout.set_font_description( fontDesc )
        inkRect, _ = layout.get_pixel_extents()
        actualW = inkRect.width; actualH = inkRect.height
    elif( variation ==  VARIATIONS.random):
        # Random alignment means random rotation
        twist = choice( TWIST_CHOICES )
        context.rotate( twist * math.atan( ( targetH - actualH  )/ actualW  ) )
        if twist < 0:
            context.translate(0, targetH - actualH )
    elif( variation == VARIATIONS.alighn_bottom ):
        context.translate(0, targetH - actualH )



    PangoCairo.show_layout(context, layout)
    data = surface.get_data()

    if( actualW > targetW ):
        #  Resize image by shrinking the width of image
        data = np.frombuffer(data, dtype=np.uint8).reshape(( targetH, canvasWidth ))[:, :actualW + 10]
        data = Image.fromarray( data ).resize( ( targetW, targetH ), Image.BILINEAR )
        data = np.array( data )
    else:
        data = np.frombuffer(data, dtype=np.uint8).reshape(( targetH, canvasWidth ))[:, :targetW]

    data = np.invert( data ) # becuase pango will render white text on black

    # Add create a noise layer and merge with image
    data = np.clip( data - normal(  *choice(noiseSDChoices), data.shape ), 0, 255 ).astype(np.uint8)
    return data




def getTrainingTexts( txtFile ):
    lines = readLines( txtFile )
    lines = filter( None, lines )
    return list(set( lines ))


def normaizeImg( img ):
    img = torch.FloatTensor( img.astype('f') )
    img = ((img*2)/255 ) -1
    return img.unsqueeze(0)


def normalizeBatch( batch ):
    images, labels = zip(*batch)
    images = [ normaizeImg(image) for image in images]
    images = torch.stack( images )
    return images, labels



class TextDataset( Sequence ):

    def __init__( self, txtFile, limit=None, num_workers=2, cache=None, batchSize=32 ):
        self.txtFile = txtFile
        self.lines = getTrainingTexts( txtFile )
        self.bs = batchSize
        self.cache = cache
        maxItemCount = len( self.lines )*totalVariations
        self.itemCount =  maxItemCount if limit == None else limit
        self.batchCount = int( self.itemCount/batchSize )
        self.randomIds = sample( range( maxItemCount ), self.itemCount )
        if cache != None:
            os.system( 'mkdir -p "%s"' % cache )

    def getLabel( self, index ):
        return self.lines[ int( index / totalVariations ) ]

    def getFont( self, index ):
        return fontListFlat[ index % totalVariations ]

    def __len__(self):
        return self.batchCount

    def getSingleItem( self, index ):
        font, variation = self.getFont( index )
        label = self.getLabel( index )
        img = renderText( label, font, variation )
        return ( img, label)

    def __getitem__( self, batchIndex ):
        if( batchIndex >= self.batchCount ):
            raise StopIteration

        #  cacheImage = '%s/batch_%03d_%06d.pgm' %( self.cache, self.bs, batchIndex)
        cacheImage = '%s/batch_%03d_%06d.jpg' %( self.cache, self.bs, batchIndex)
        cacheLabels = '%s/batch_%03d_%06d.txt' %( self.cache, self.bs, batchIndex)
        if( self.cache and os.path.exists( cacheImage ) ):
            labels = readLines( cacheLabels )
            images = np.array_split( np.array( Image.open( cacheImage ) ), len( labels ) )
            out = list( zip( images, labels ) )
        else:
            startIndex = batchIndex*self.bs
            out = [ self.getSingleItem( i ) for i in self.randomIds[ startIndex: startIndex+self.bs ] ]

        #  write cache
        if( self.cache and not os.path.exists( cacheImage )):
            images, labels = zip( *out )
            images = Image.fromarray( np.concatenate( images, 0) )
            images.save( cacheImage, 'JPEG', quality=50 )
            writeFile( cacheLabels, '\n'.join( labels ) )
        return normalizeBatch( out )
