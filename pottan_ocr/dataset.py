#!/usr/bin/env python3

from .utils import config, readLines, writeFile, showImg, normalizeBatch

import os
import math
from enum import Enum
from collections import Sequence
from random import choice, sample
from re import split
import numpy as np
from numpy import random
from PIL import Image
import cairo
import gi
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo

imageHeight = config['imageHeight']
defaultFontSize = config['defaultFontSize']
targetW = config['trainImageWidth']
#  let Canvas have some extra space than required so that, we can handle text overflow easly
canvasWidth = targetW*2


#  Text alignment variations
VARIATIONS = Enum('AlignmentVariation', 'random align_h fit_height' )

fontList = config['fonts']
fontListFlat = []
for fnt, styles, *customFontSize in fontList:
    for style in styles:
        fontSize = defaultFontSize if len( customFontSize ) == 0 else customFontSize[0]
        fontDescStr = '%s %s %s' %( fnt, style, fontSize )
        fontListFlat.append([ fontDescStr, VARIATIONS.random, ])
        fontListFlat.append([ fontDescStr, VARIATIONS.random, ])
        fontListFlat.append([ fontDescStr, VARIATIONS.align_h, ])
        fontListFlat.append([ fontDescStr,  VARIATIONS.fit_height ])

totalVariations = len(fontListFlat)

leftPaddingChoices = [
        '',
        ' ',
        ]


#  Pango FontDescription cache
fontDescCache = {};


#  List of available angles of rotation.
#  -1 --> Maximum possible negative rotation with in available free space
#  +1 --> Maximum possible positive rotation with in available free space
TWIST_CHOICES = [ 
        -1,
        -0.95,
        -0.90,
        -0.80,
        #  -0.20,
        #  0.00,
        #  0.20,
        0.80,
        0.90,
        0.95,
        1 ]

noiseSDChoices = [
        #  ( mean, standard-deviation ) of normal distribution
        (0,  0),
        (0, 10),
        (0, 20),
        (25, 30),
        (40, 35),
        (20, 25),
        (30, 31),
        #  (0, 40),
        #  (0, 50),
        #  (0, 60),
        #  (0, 70),
        #  (0, 80),
        #  (0, 90),
        #  (0,100),
        #  (0,   0),
        #  (10, 30),
        #  (20, 30),
        #  (30, 30),
        #  (30, 35),
        #  (40, 30),
        #  (50, 30),
        #  (60, 30),
        #  (70, 30),
        #  (80, 30),
        #  (90, 30),
        #  (100,30),
        #  (0,   0),
        #  (10, 50),
        #  (20, 50),
        #  (30, 50),
        #  (30, 55),
        #  (40, 50),
        #  (50, 50),
        #  (60, 50),
        #  (70, 50),
        #  (80, 50),
        #  (90, 50),
        #  (100,50),
        ]

bgChoices=[
        1,
        1,
        0.9,
        0.8,
        0.7,
        0.6,
        ]

##
#
# @param text - The text to be rendered
# @param font - A string representing Pango font description. Eg: 'FreeSans bold 18'
# @param variation - representing alighnment variation .
#
# @return numpy.array
#  i=0
#  stats = { 'tw': 0, 'bot': 0, 'top': 0, 'fit': 0 }
def renderText( text, font, variation ):
    """Render a unicode text into 32xH image and return a numpy array"""

    #  global stats
    text = choice( leftPaddingChoices ) + text
    #  text = choice( leftPaddingChoices ) + text[:-20] + font
    #  print("'"+text+"'")
    surface = cairo.ImageSurface(cairo.FORMAT_A8, canvasWidth, imageHeight )
    context = cairo.Context(surface)
    pc = PangoCairo.create_context(context)
    layout = PangoCairo.create_layout(context)
    if( font in fontDescCache ):
        fontDesc = fontDescCache[font]
    else:
        fontDesc = fontDescCache[font] = Pango.font_description_from_string( font )
    layout.set_font_description(Pango.FontDescription( font ))
    layout.set_text( text , -1 );

    inkRect, _ = layout.get_pixel_extents()
    actualW = inkRect.width; actualH = inkRect.height

    if( variation == VARIATIONS.fit_height ):
        #  stats['fit'] +=1
        #  Fit-height is done by varying font size
        fontDesc = layout.get_font_description()
        fontDesc.set_size( int(fontDesc.get_size() * 0.85 * imageHeight/ actualH ) )
        layout.set_font_description( fontDesc )
        inkRect, _ = layout.get_pixel_extents()
        actualW = inkRect.width; actualH = inkRect.height
    elif( variation ==  VARIATIONS.random):
        #  stats['tw'] +=1
        # Random alignment means random rotation
        twist = choice( TWIST_CHOICES )
        context.rotate( twist * math.atan( ( imageHeight - actualH  )/ actualW  ) )
        if twist < 0:
            context.translate(0, imageHeight - actualH )
    elif( variation == VARIATIONS.align_h ):
        availablePadding = imageHeight - actualH
        context.translate(0, int( availablePadding * random.random()))



    PangoCairo.show_layout(context, layout)
    data = surface.get_data()
    #  import ipdb; ipdb.set_trace()

    if(  (actualW > targetW ) or ( actualW < targetW * 0.7 )):
        #  print(' actualW > targetW ')
        #  Resize image by shrinking the width of image
        data = np.frombuffer(data, dtype=np.uint8).reshape(( imageHeight, canvasWidth ))[:, :actualW + 10]
        data = Image.fromarray( data ).resize( ( targetW, imageHeight ), Image.BILINEAR )
        data = np.array( data )
    else:
        data = np.frombuffer(data, dtype=np.uint8).reshape(( imageHeight, canvasWidth ))[:, :targetW]

    data = np.invert( data ) # becuase pango will render white text on black
    bg = choice( bgChoices )
    if( bg != 1 ):
        data = (data*bg).astype( np.uint8 )

    # Add create a noise layer and merge with image
    ncc = choice(noiseSDChoices)
    #  ncc = noiseSDChoices[ i % len( noiseSDChoices )  ]
    #  print( ncc )
    data = np.clip( data - random.normal(  *ncc, data.shape ), 0, 255 ).astype(np.uint8)
    #  import ipdb; ipdb.set_trace()
    #  data = np.clip( data - random.multivariate_normal( [ ncc[0] ], [[ ncc[1]]], data.shape ).squeeze(), 0, 255 ).astype(np.uint8)
    return data




def getTrainingTexts( txtFile ):
    lines = readLines( txtFile )
    lines = filter( None, lines )
    return list(set( lines ))



class TextDataset( Sequence ):

    def __init__( self, txtFile, limit=None, cache=None, batchSize=32, overwriteCache=False ):
        self.txtFile = txtFile
        self.lines = getTrainingTexts( txtFile )
        self.bs = batchSize
        self.cache = cache
        self.overwriteCache = overwriteCache
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
        unNormalized =  self.getUnNormalized( batchIndex )
        return normalizeBatch( unNormalized )

    def getUnNormalized( self, batchIndex ):
        if( batchIndex >= self.batchCount ):
            raise StopIteration

        #  cacheImage = '%s/batch_%03d_%06d.pgm' %( self.cache, self.bs, batchIndex)
        cacheImage = '%s/batch_%03d_%06d.jpg' %( self.cache, self.bs, batchIndex)
        cacheLabels = '%s/batch_%03d_%06d.txt' %( self.cache, self.bs, batchIndex)
        if( self.cache and os.path.exists( cacheImage ) and not self.overwriteCache ):
            labels = readLines( cacheLabels )
            images = np.array_split( np.array( Image.open( cacheImage ) ), len( labels ) )
            out = list( zip( images, labels ) )
        else:
            startIndex = batchIndex*self.bs
            out = [ self.getSingleItem( i ) for i in self.randomIds[ startIndex: startIndex+self.bs ] ]

        #  write cache
        if( self.cache and ( ( not os.path.exists( cacheImage ) ) or self.overwriteCache ) ):
            images, labels = zip( *out )
            images = Image.fromarray( np.concatenate( images, 0) )
            images.save( cacheImage, 'JPEG', quality=50 )
            writeFile( cacheLabels, '\n'.join( labels ) )
        return out

    def printStats( self ):
        print( stats )
