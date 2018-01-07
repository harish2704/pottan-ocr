
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
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo



fontList = readYaml('./fontlist.yaml')
fontListFlat = []
for fnt, styles in fontList:
    for style in styles:
        fontListFlat.append([ fnt, style ])

totalVariations = len(fontListFlat)
print( 'Total font variations = %d'% totalVariations )


fontDescCache = {};

def pangoRenderText( text, font, targetW, targetH, twist ):
    """
    twist - can be anything with in -1  to +1. +1 means maximum possible clockvise rotation, and -1 is maximum possible anticlockvise rotation
    """
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
    context.rotate( twist * targetH / actualW / 3 ) # found '3' is the best fit instead of '2' ( 2 from 2*pi )
    if twist < 0:
        context.translate(0, targetH - actualH )

    PangoCairo.show_layout(context, layout)
    data = surface.get_data()
    return np.frombuffer(data, dtype=np.uint8).reshape(( targetH, targetW ))


def extractWords( txtFile ):
    words = readFile( txtFile )
    words = filter( None, re.split('[\n\ ]', words ) )
    return list(set( words ))


twistChoices = [ i/4 for i in range(-4,4) ]

def renderText( word, font='AnjaliOldLipi', style='regular' ):
    img = pangoRenderText( word, '%s %s 16' % ( font, style ), 400, 32, choice( twistChoices ) )
    return np.invert( img )


def normaizeImg( img ):
    img = torch.FloatTensor( img.astype('f') )
    img = ((img*2)/255 ) -1
    return img


def alignCollate( batch ):
    images, labels = zip(*batch)
    images = [ normaizeImg(image) for image in images]
    images = torch.cat([t.unsqueeze(0) for t in images], 0)
    return images, labels


noiseChoices = [ i/20 for i in range(9) ]

class TextDataset(Dataset):

    def __init__(self, txtFile):
        self.txtFile = txtFile
        self.words = extractWords( txtFile )
        self.itemCount = len( self.words )*totalVariations

    def __len__(self):
        return self.itemCount

    def __getitem__(self, index):
        wordIdx = int( index / totalVariations )
        font, style = fontListFlat[ index % totalVariations ]
        label = self.words[ wordIdx ]
        img = renderText( label, font=font, style=style )
        gauss = randn( *img.shape )
        img = cv2.add( img, img * gauss * choice(noiseChoices), dtype=cv2.CV_8UC3)
        # Convert into 1xWxH
        img = np.expand_dims( img, axis=0 )
        return ( img, label)
