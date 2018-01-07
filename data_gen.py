#!/usr/bin/env python3

from utils import readFile, writeFile
from string_converter import encodeStr, decodeStr
import re
import numpy as np
import os
from PIL import Image
import array

import cairo
import gi
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo

fontDescCache = {};

def pangoRenderText( text, font, targetW, targetH, xoffset, yoffset, twist ):
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
    context.translate( xoffset, yoffset )
    PangoCairo.show_layout(context, layout)
    data = surface.get_data()
    return np.frombuffer(data, dtype=np.uint8).reshape(( targetH, targetW ))



zwjMapping = {
        'ല്‍': 'ൽ',
        'ന്‍': 'ൻ',
        'ണ്‍': 'ൺ',
        'ര്‍': 'ർ',
        'ക്‍': 'ൿ',
        }
zwnjChilluRe = re.compile( '(' + '|'.join(zwjMapping.keys()) + ')' )
zwnjRe = re.compile('[‌‍]')



def extractWords( txtFile ):
    words = readFile( txtFile )
    words = filter( None, re.split('[\n\ ]', words ) )
    return list(set( words ))


def renderText( word, font='AnjaliOldLipi', style='regular', yoffset=3 ):
    img = pangoRenderText( word, '%s %s 16' % ( font, style ), 400, 32, 0, yoffset, 0 )
    img = np.invert( img )
    # Convert into 1xWxH  
    return np.expand_dims( img, axis=0 )

#  Generate image as numpy array for a unicode text.
#  We are using fixed width because torch.DataLoader expect a fixed size array
# pangoRenderText function will fill the extra space with white/black color
def computeDataset( words, **kwargs ):
    imgs = []
    labels = []
    total = len( words )
    for idx, word in enumerate(words):
        if( idx % 10000 == 0 ):
            print( 'ComputeDataset %4.2f %%' % ( idx*100/total ) )
        imgs.append( renderText( word, **kwargs )  )
        labels.append( word )
    return imgs, labels




class DataGen:

    def __init__(self, infile, outfile ):
        self.WORD_LIST_FILE = infile
        self.DATA_FILE = outfile


    #  Pre-process input text file.
    #  Pre-processing includes the following steps
    #  * Convert ZWNJ based chillu to atomic chillu.
    #  * Remove/replace un-necessary chars
    #  * Then try encode each word. If it fails becuase of the presents of ZWJ char, then remove that ZWJ
    def preProcess( self ):
        txtFile = self.WORD_LIST_FILE
        words = readFile( txtFile )

        # Replace all zwnj chillus with atomic chillu
        words = zwnjChilluRe.sub( lambda g: zwjMapping[ g.group(0) ], words )

        words = filter( None, re.split('‌*[\n\ ]', words ) )
        words = list(set( words ))
        goodWords = []
        total = len( words )
        for idx, w in enumerate(words):
            if( idx % 10000 == 0 ):
                print( 'preProcess %4.2f %%' % ( idx*100/total ) )
            try:
                encodeStr( w )
                goodWords.append( w )
            except Exception as e:
                if( re.match( ".*u200[cd].* is not in list", e.args[0] ) ):
                    #  print('Replacing zwj in "%s"' % w )
                    goodWords.append( zwnjRe.sub('', w ) )
                else:
                    pass
                    #  print( 'Omiting "%s"' % w, e )
                    #  import ipdb; ipdb.set_trace()
                    #  raise e
        goodWords = list( filter( lambda x: len(x)< 27, goodWords ) )
        writeFile( txtFile, '\n'.join( goodWords ))




    # function for self testing.
    # Encode each wrod, then decode it back
    def testEncoding( self ):
        words = extractWords( self.WORD_LIST_FILE )
        for w in words:
            enc, encSize = encodeStr( w )
            dec  = decodeStr( enc )
            if( w != dec ):
                raise ValueError('Encoding failed "%s" != "%s"'% ( w, dec ) )



    def createDataset( self, opts ):
        words = extractWords( self.WORD_LIST_FILE )
        if( opts.format == 'numpy' ):
            imgs, labels = computeDataset( words, font=opts.font, style=opts.style )
            np.savez_compressed( self.DATA_FILE, list(zip( imgs, labels )) )
        else:
            os.system('mkdir -p %s' % self.DATA_FILE )
            for w in words:
                img = renderText( w, font=opts.font, style=opts.style )
                img = Image.fromarray( img[0] )
                img.save( '%s/%s.png' %( self.DATA_FILE, w ) )







def main( opt ):
    dg = DataGen( opt.input, opt.output )
    if( opt.preprocess ):
        print('Pre-processing' )
        dg.preProcess()

    if( opt.testencoding ):
        print('Testing encodability of  dataset' )
        dg.testEncoding()

    if( not opt.skip_creation ):
        print( 'Create dataset' )
        dg.createDataset( opt )
    print('Completed generating traindata for dataset\n\n' )



if( __name__ == '__main__' ):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', help='preprocess wordlist and save it back to disk')
    parser.add_argument('--testencoding', action='store_true', help='do encodability test on each workd in the wordlist')
    parser.add_argument('--skip-creation', action='store_true', help='Skip dataset creation')
    parser.add_argument('--input', help='input text file contains words')
    parser.add_argument('--output', help='output numpy data file')
    parser.add_argument('--font', default='AnjaliOldLipi', help='Name of the font')
    parser.add_argument('--style', default='regular', choices=['regular', 'bold', 'Italic', 'bold italic' ], help='font style')
    parser.add_argument('--format', choices=[ 'numpy', 'images' ], default='numpy', help='Format of output. Numpy array vs Directory of images' )
    parser.add_argument('--name', help='name of dataset. ( Ie, input=./data/<name>.txt , output=./data/<name>_data.npz )')
    opt = parser.parse_args()
    if( opt.name ):
        opt.input = './data/%s.txt' % opt.name
        opt.output = './data/%s_data' % opt.name
    elif( opt.input and opt.output ):
        pass
    else:
        parser.error("Either '--name' or both '--input' and '--output' need to be specified")

    main( opt )

