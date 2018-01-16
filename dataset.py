
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
twistChoices = [ -1, -0.95, -0.85, 0.85, 0.95, 1 ]

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

    inkRect, _ = layout.get_pixel_extents()
    actualW = inkRect.width; actualH = inkRect.height
    #  import ipdb; ipdb.set_trace()

    if( variation == 'fit-height' ):
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


noiseSDChoices = [
        (0,0),
        (10, 70),
        (12, 80),
        (15, 90),
        (20, 100)
        ]

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
        #  from hari import imshowgr; import ipdb; ipdb.set_trace()
        img = np.clip( img - cv2.randn( np.zeros( img.shape, dtype=np.float ), *choice(noiseSDChoices) ), 0, 255 ).astype(np.uint8)
        img = np.expand_dims( img, axis=0 )
        return ( img, label)
