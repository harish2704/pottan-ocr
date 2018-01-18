#!/usr/bin/env python3

from .utils import config, readLines

import torch
from torch.utils.data import Dataset

import math
from enum import Enum
from collections import Sequence
from random import choice, sample
from re import split
import numpy as np
from numpy.random import randn
import cv2
import cairo
import gi
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo


VARIATIONS = Enum('AlignmentVariation', 'random alighn_bottom fit_height' )

fontList = config['fonts']
fontListFlat = []
for fnt, styles in fontList:
    for style in styles:
        fontDescStr = '%s %s 18' %( fnt, style )
        fontListFlat.append([ fontDescStr, VARIATIONS.random,       ])
        fontListFlat.append([ fontDescStr,  VARIATIONS.alighn_bottom ])
        fontListFlat.append([ fontDescStr,  VARIATIONS.fit_height ])

totalVariations = len(fontListFlat)
print( 'Total font variations = %d'% totalVariations )


#  Pango FontDescription cache
fontDescCache = {};


#  List of available angles of rotation.
#  -1 --> Maximum possible negative rotation with in available free space
#  +1 --> Maximum possible positive rotation with in available free space
TWIST_CHOICES = [ -1, -0.95, -0.85, 0.85, 0.95, 1 ]



##
#
# @param text - The text to be rendered
# @param font - A string representing Pango font description. Eg: 'FreeSans bold 18'
# @param variation - representing alighnment variation .
#
# @return numpy.array
def renderText( text, font, variation ):
    """Render a unicode text into 32xH image and return a numpy array"""
    targetW = 1024
    targetH = 32

    surface = cairo.ImageSurface(cairo.FORMAT_A8, targetW, targetH )
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
    #  import ipdb; ipdb.set_trace()

    if( variation == VARIATIONS.fit_height ):
        fontDesc = layout.get_font_description()
        fontDesc.set_size( int(fontDesc.get_size() * targetH/ actualH ) )
        layout.set_font_description( fontDesc )
        inkRect, _ = layout.get_pixel_extents()
        actualW = inkRect.width; actualH = inkRect.height

    if( actualW > targetW ):
        fontDesc = layout.get_font_description()
        fontDesc.set_size( int(fontDesc.get_size() * 0.9 * targetW/ actualW  ) )
        layout.set_font_description( fontDesc )
        inkRect, _ = layout.get_pixel_extents()
        actualW = inkRect.width; actualH = inkRect.height

    if( variation ==  VARIATIONS.random):
        twist = choice( TWIST_CHOICES )
        context.rotate( twist * math.atan( ( targetH - actualH  )/ actualW  ) )
        if twist < 0:
            context.translate(0, targetH - actualH )
    elif( variation == VARIATIONS.alighn_bottom ):
        context.translate(0, targetH - actualH )

    PangoCairo.show_layout(context, layout)
    data = surface.get_data()
    data = np.frombuffer(data, dtype=np.uint8).reshape(( targetH, targetW ))
    data = np.invert( data )
    data = np.clip( data - cv2.randn( np.zeros( data.shape, dtype=np.float ), *choice(noiseSDChoices) ), 0, 255 ).astype(np.uint8)
    data = np.expand_dims( data, axis=0 )
    return data




def getTrainingTexts( txtFile ):
    lines = readLines( txtFile )
    lines = filter( None, lines )
    return list(set( lines ))


#  TODO: Move the below code into torch tensor. ( To make use of GPU )
def normaizeImg( img ):
    img = torch.FloatTensor( img.astype('f') )
    img = ((img*2)/255 ) -1
    return img


def alignCollate( batch ):
    images, labels = zip(*batch)
    images = [ normaizeImg(image) for image in images]
    images = torch.cat([t.unsqueeze(0) for t in images], 0)
    return images, labels


noiseSDChoices = [
        #  ( mean, standard-deviation )
        (0,0),
        (10, 70),
        (12, 80),
        (15, 90),
        (20, 100)
        ]

class TextDataset( Sequence ):

    def __init__( self, txtFile, limit=None, num_workers=2, cache=None, batch_size=32 ):
        self.txtFile = txtFile
        self.lines = getTrainingTexts( txtFile )
        self.bs = batch_size
        self.cache = cache
        maxItemCount = len( self.lines )*totalVariations
        self.itemCount =  maxItemCount if limit == None else limit
        self.batchCount = int( self.itemCount/batch_size )
        self.randomIds = sample( range( maxItemCount ), self.itemCount )

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

        cacheImage = '%s/batch_%03d_%06d.pgm' %( self.cache, self.bs, index)
        cacheLabels = '%s/batch_%03d_%06d.txt' %( self.cache, self.bs, index)
        if( self.cache and os.path.exists( cacheImage ) ):
            images = Image.open( cacheImage )
            labels = readLines( cacheLabels )
            out = zip( images, labels )
        else:
            startIndex = batchIndex*self.bs
            out = [ self.getSingleItem( i ) for i in self.randomIds[ startIndex: startIndex+self.bs ] ]
        return alignCollate( out )
