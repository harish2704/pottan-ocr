
import torch
from torch.utils.data import Dataset
from utils import readYaml, readFile

from random import choice
import re
import numpy as np
from numpy.random import randn
import cv2
import cairo
import gi
import math
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo


variationChoices = [
        'random',
        'align-top',
        'align-bottom',
        'fit-height'
        ]

fontList = readYaml('./fontlist.yaml')
fontListFlat = []
for fnt, styles in fontList:
    for style in styles:
        fontDescStr = '%s %s 18' %( fnt, style )
        fontListFlat.append([ fontDescStr,  choice( variationChoices )])

totalVariations = len(fontListFlat)
print( 'Total font variations = %d'% totalVariations )


fontDescCache = {};
twistChoices = [ -1, -0.85, -0.75, 0.75, 0.85, 1 ]

def renderText( text, font, variation ):
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

    actualW, actualH = layout.get_pixel_size()

    if( variation == 'fit-height' ):
        fontDesc = layout.get_font_description()
        fontDesc.set_size( int(fontDesc.get_size() * targetH/ actualH ) )
        layout.set_font_description( fontDesc )

    actualW, actualH = layout.get_pixel_size()
    if( actualW > targetW ):
        fontDesc = layout.get_font_description()
        fontDesc.set_size( int(fontDesc.get_size() * 0.9 * targetW/ actualW  ) )
        layout.set_font_description( fontDesc )
    actualW, actualH = layout.get_pixel_size()

    if( variation == 'random' ):
        twist = choice( twistChoices )
        context.rotate( twist * math.atan( ( targetH - actualH  )/ actualW  ) )
        if twist < 0:
            context.translate(0, targetH - actualH )
    elif( variation == 'align-bottom' ):
        context.translate(0, targetH - actualH )

    PangoCairo.show_layout(context, layout)
    data = surface.get_data()
    data = np.frombuffer(data, dtype=np.uint8).reshape(( targetH, targetW ))
    return np.invert( data )




def getTrainingTexts( txtFile ):
    lines = readFile( txtFile )
    lines = filter( None, re.split('[\r\n]', lines ) )
    return list(set( lines ))




def normaizeImg( img ):
    img = torch.FloatTensor( img.astype('f') )
    img = ((img*2)/255 ) -1
    return img


def alignCollate( batch ):
    images, labels = zip(*batch)
    images = [ normaizeImg(image) for image in images]
    images = torch.cat([t.unsqueeze(0) for t in images], 0)
    return images, labels


#  bgChoices = [ 0, 10, 30, 50 ]
noiseChoices = [ 0, 0, 5, 10, 0,15, 20, ]

class TextDataset(Dataset):

    def __init__(self, txtFile):
        self.txtFile = txtFile
        self.words = getTrainingTexts( txtFile )
        self.variationCount = totalVariations
        self.itemCount = len( self.words )*totalVariations

    def getFont( self, index ):
        return fontListFlat[ index % totalVariations ]

    def __len__(self):
        return self.itemCount

    def __getitem__(self, index):
        wordIdx = int( index / totalVariations )
        font, variation = fontListFlat[ index % totalVariations ]
        label = self.words[ wordIdx ]
        img = renderText( label, font, variation )
        #  import ipdb; ipdb.set_trace()
        img = img - cv2.randn( np.zeros( img.shape, dtype=np.uint8 ), 0, choice( noiseChoices ) )
        img = np.expand_dims( img, axis=0 )
        return ( img, label)
