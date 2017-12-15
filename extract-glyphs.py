#!/usr/bin/env python3

from GlyphExtractor import GlyphExtractor
from utils import readJson
import os
from os import path
import cv2

import sys

def main( fname, ourDir ):
    gd = GlyphExtractor( fname )
    contours = gd.detectContours( rect=(2,2) )
    gd.extractContours( contours, ourDir )

def extractFromRendered( fname, outDir ):
    imageFileBaseName = path.basename( fname )
    lableMap = readJson( './cache/glyph_labels.json' )
    gd = GlyphExtractor( fname )
    contours = gd.detectContours( rect=(1,1) )

    os.makedirs( outDir, exist_ok=True )
    rgb = gd.rgb
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    totalHeight = rgb.shape[0]
    glyphLabels = readJson('./cache/glyph_labels.json')
    totalGlyphCount = len( glyphLabels )
    heightPerGlyph = totalHeight/totalGlyphCount

    glyphCountrourMap = [ [] for i in range( totalGlyphCount ) ]
    
    for idx, ( x, y, w, h ) in enumerate(contours):
        glyphId = int( y/heightPerGlyph )

        r = float(cv2.countNonZero( small[y:y+h, x:x+w])) / (w * h)
        #  if more than 90% area is filled, then it is a 'dash' symbol which is kept as placeholder for zwnj chihnas
        label = glyphLabels[ glyphId ][1]
        #  print ( r, label )
        if( r > 0.3 ):
            glyphCountrourMap[ glyphId ].append(( x, y, w, h,  label ))


    for contList in glyphCountrourMap:
        if( len( contList ) == 1 ):
            ( x, y, w, h, label ) = contList[0]
            if( w > 10 ):
                croppedImg = rgb[y:y+h, x:x+w ]
                cv2.imwrite( '%s/%s_%s.png'%( outDir, label, imageFileBaseName ), croppedImg )


if __name__ == "__main__":
    #  main( sys.argv[1],  sys.argv[2] )
    extractFromRendered( sys.argv[1],  sys.argv[2] )

