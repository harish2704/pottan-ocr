#!/usr/bin/env python3


import os
def ee():
    os._exit(0)

from os import makedirs
from utils import readJson, fixedWidthImg
from unicode_text_to_image_array.scribe import scribe_wrapper
from GlyphExtractor import detectContoursFromImg, markContoursImg
import cv2
import numpy as np
from matplotlib.pyplot import imshow

def filterInnerContours( conts ):
    outerOne = conts[ 0 ]
    for i in conts:
        if( i[2] > outerOne[2] ):
            outerOne = i

    #  print( 'Bigger box', outerOne )
    x1 = outerOne[0]
    y1 = outerOne[1]
    x2 = x1 + outerOne[2]
    y2 = y1 + outerOne[3]

    children = conts.copy();
    children.remove( outerOne )

    out = [ outerOne ];
    for child in children:
        (x,y,w,h) = child
        if( ( x < x1 ) or (x+w) > x2  or y < y1  or ( y+h ) > y2 ):
            out.append( child )
    return out

def show( im ):
    imshow( im, cmap='gray')

glyphs = readJson('./cache/glyph_labels.json')
fonts=[
        'AnjaliOldLipi',
        'Rachana'
        'Noto Serif Malayalam',
        'Meera',
        'Kalyani',
        'Noto Sans Malayalam UI',
        'RaghuMalayalam',
        'Lohit Malayalam',
        'Suruma',
        'Dyuthi',
        'Samyak'
        ]
styles = [
        'regular',
        'bold',
        'italic',
        'bold italic'
        ]



def renderGlyph( txt, lable, font='AnjaliOldLipi', style='regular' ):
    fontStyle = "%s %s 38"%( font, style )
    img = ( 255 - scribe_wrapper( txt, fontStyle, 120, 0, 0, 0 ) )
    img1 = cv2.cvtColor( img, cv2.COLOR_GRAY2BGR  )
    contours = detectContoursFromImg( img, eclipse=(2,2), rect=(1,1) )[3]
    contours = [ cv2.boundingRect( contour ) for contour in contours ];
    contours = filterInnerContours( contours )
    if( lable[0:2] == '-@' and len( contours ) > 1 ):
        if( lable == '-@e' or lable == '-@E' ):
            contours = contours[1:]
        else:
            contours = contours[:-1]
        contours = filterInnerContours( contours )

    if( len( contours ) > 1 ):
        #  cv2.imwrite('./test.png', img1 )
        #  markContoursImg( img1, contours, minWidth=5 )
        #  imshow( img1 )
        #  import ipdb;ipdb.set_trace()
        raise ValueError( "Found multiple glyphs for %s -> %s . Ignoring" % ( txt, lable ) )
    #  print( "Processing %s -> %s ." % ( txt, lable ) )
    ( x, y, w, h ) = contours[0]
    croppedImg = img[y:y+h, x:x+w ]
    return fixedWidthImg( croppedImg )

def main():
    for ( glyph, lable ) in glyphs:
        dirname = "./cache/generated/%s"% lable
        makedirs( dirname, exist_ok=True )
        for font in fonts:
            for style in styles:
                try:
                    img = renderGlyph( glyph, lable, font, style )
                    fname =  "%s/%s_%s_%s.pgm" %( dirname, lable, font, style )
                    cv2.imwrite( fname, img )
                except Exception as e:
                    print( e, glyph+ ' - ' +lable, font, style )

if( __name__ == '__main__' ):
    main()

#  ( txt, lable ) = glyphs[ 96 ]
#  renderGlyph( txt, lable )
