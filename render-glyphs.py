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
        if( w> 5 and h > 5 ):
            if( ( x < x1 ) or (x+w) > x2  or y < y1  or ( y+h ) > y2 ):
                out.append( child )
    return out


def show( im ):
    imshow( im, cmap='gray')

glyphs = readJson('./cache/glyph_labels.json')
fonts=[
        ('AnjaliOldLipi', ['regular', 'bold' ]),
        ('Chilanka', ['regular', 'bold', 'italic' ]),
        ('Dyuthi', ['regular', 'bold', 'italic', 'bold italic']),
        ('Kalyani', ['regular', 'bold', 'italic', 'bold italic']),
        ('Karumbi', ['regular', 'bold', 'italic', 'bold italic']),
        #  It is already too tick. Rm bold
        ('Keraleeyam', ['regular', 'italic' ]),
        ('Lohit Malayalam', ['regular', 'bold', 'italic', 'bold italic']),
        ('Manjari', ['regular', 'bold', 'italic', 'bold italic']),
        ('Manjari,Manjari Thin', ['regular', 'italic']),
        ('Meera', ['regular', 'bold', 'italic', 'bold italic']),
        ('ML-NILA01', ['regular', 'bold', 'italic', 'bold italic']),
        ('ML-NILA02', ['regular', 'bold', 'italic', 'bold italic']),
        ('ML-NILA03', ['regular', 'bold', 'italic', 'bold italic']),
        ('ML-NILA04', ['regular', 'bold', 'italic', 'bold italic']),
        ('ML-NILA05', ['regular', 'bold', 'italic', 'bold italic']),
        ('ML-NILA06', ['regular', 'bold', 'italic', 'bold italic']),
        ('ML-NILA07', ['regular', 'bold', 'italic', 'bold italic']),
        ('Noto Sans Malayalam', ['regular', 'bold', 'italic', 'bold italic']),
        ('Noto Sans Malayalam UI', ['regular', 'bold', 'italic', 'bold italic']),
        ('Noto Serif Malayalam', ['regular', 'bold', 'italic', 'bold italic']),
        ('Rachana', ['regular', 'bold', 'italic', 'bold italic']),
        ('RaghuMalayalam', ['regular', 'bold', 'italic', 'bold italic']),
        ('Samyak Malayalam', ['regular', 'bold', 'italic', 'bold italic']),
        ('Suruma', ['regular', 'bold', 'italic', 'bold italic']),
        #  It is already too tick. Rm bold
        ('Uroob', ['regular', 'italic' ])
        ]



def renderGlyph( txt, lable, font='AnjaliOldLipi', style='regular' ):
    #  if( lable[0] == '@' ):
        #  txt = 'ഉ' + txt
    fontStyle = "%s %s 48"%( font, style )
    img = ( 255 - scribe_wrapper( txt, fontStyle, 120, 10, 10, 0 ) )
    img1 = cv2.cvtColor( img, cv2.COLOR_GRAY2BGR  )
    contours = detectContoursFromImg( img, eclipse=(2,2), rect=(1,1) )[0]
    contours = [ cv2.boundingRect( contour ) for contour in contours ];
    if( lable[0] == '@' and len( contours ) > 1 ):
        if( lable == '@va' and ( font in [ 'ML-NILA04','ML-NILA05','ML-NILA06','ML-NILA07',  ]) ):
            raise ValueError('Skipping @va rendering for NILA fonts')
        #  if( lable == '@i' ):
            #  markContoursImg( img1, contours, minWidth=5 ); imshow( img1 ); import ipdb;ipdb.set_trace()
        contours.sort(key=lambda x:x[0]+x[2] )
        if( lable == '@e' or lable == '@E' ):
            contours = contours[0:1]
        else:
            contours = contours[-1:]
    else:
        contours = filterInnerContours( contours )

    if( len( contours ) > 1 ):
        #  markContoursImg( img1, contours, minWidth=5 ); imshow( img1 ); import ipdb;ipdb.set_trace()
        raise ValueError( "Found multiple glyphs for %s -> %s . Ignoring" % ( txt, lable ) )
    #  print( "Processing %s -> %s ." % ( txt, lable ) )
    ( x, y, w, h ) = contours[0]
    croppedImg = img[y:y+h, x:x+w ]
    return fixedWidthImg( croppedImg )

def main():
    for ( glyph, lable ) in glyphs:
        dirname = "./cache/generated/%s"% lable
        makedirs( dirname, exist_ok=True )
        for ( font, styles ) in fonts:
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
