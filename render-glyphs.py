#!/usr/bin/env python3


from os import makedirs
from utils import readJson
from unicode_text_to_image_array.scribe import scribe_wrapper
from GlyphExtractor import detectContoursFromImg, markContoursImg, filterInnerContours
import cv2
import numpy as np
from matplotlib.pyplot import imshow
def show( im ):
    imshow( im, cmap='gray')

glyphs = readJson('./cache/glyph_labels.json')
fonts=[
        'AnjaliOldLipi',
        #  'Rachana'
        #  'Noto Serif Malayalam',
        #  'Meera',
        #  'Kalyani',
        #  'Noto Sans Malayalam UI',
        #  'RaghuMalayalam',
        #  'Lohit Malayalam',
        #  'Suruma',
        #  'Dyuthi',
        #  'Samyak'
        ]
styles = [
        'regular',
        'bold',
        'italic',
        'bold italic'
        ]
FINAL_W=96
FINAL_H=96
frameImg = np.zeros( ( FINAL_H, FINAL_W ), dtype=np.uint8)



def renderGlyph( txt, lable, font='AnjaliOldLipi', style='regular' ):
    fontStyle = "%s %s 38"%( font, style )
    img = ( 255 - scribe_wrapper( txt, fontStyle, 120, 0, 0, 0 ) )
    img1 = cv2.cvtColor( img, cv2.COLOR_GRAY2BGR  )
    contours = detectContoursFromImg( img )[3]
    contours = [ cv2.boundingRect( contour ) for contour in contours ];
    contours = filterInnerContours( contours )
    if( lable[0] == '@' ):
        if( lable == '@e' or lable == '@E' ):
            contours = contours[:-1]
        else:
            contours = contours[1:]
        contours = filterInnerContours( contours )

    if( len( contours ) > 1 ):
        #  markContoursImg( img1, contours )
        #  show( img1 )
        #  filterInnerContours( contours )
        #  import ipdb;ipdb.set_trace()
        raise ValueError( "Found multiple glyphs for %s -> %s . Ignoring" % ( txt, lable ) )
    #  print( "Processing %s -> %s ." % ( txt, lable ) )
    ( x, y, w, h ) = contours[0]
    croppedImg = img[y:y+h, x:x+w ]
    ( currentH, currentW) = croppedImg.shape
    if( currentH > currentW ):
        newWidth = int( currentW * ( FINAL_H/currentH ) )
        newHeigh = FINAL_H
        offsetY=0
        offsetX = int( ( FINAL_W - newWidth )/2 )
    else:
        newHeigh = int( currentH * ( FINAL_W/currentW ) )
        newWidth = FINAL_W
        offsetX=0
        offsetY = int( ( FINAL_H - newHeigh )/2 )
    resizedImg = cv2.resize( croppedImg, ( newWidth, newHeigh ), interpolation=cv2.INTER_CUBIC )
    frameImg = np.ones( ( FINAL_H, FINAL_W ), dtype=np.uint8)*255
    frameImg[offsetY:offsetY+newHeigh, offsetX:offsetX+newWidth] = resizedImg
    return frameImg

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
                    print( e )

main()

#  ( txt, lable ) = glyphs[ 96 ]
#  renderGlyph( txt, lable )
