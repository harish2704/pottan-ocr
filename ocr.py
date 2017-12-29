#!/usr/bin/env python3

from GlyphExtractor import GlyphExtractor
from utils import readJson
import os
from os import path
import cv2
import numpy as np
import json

import sys
from detect_char import detectGlyph
from utils import fixedWidthImg, writeFile
from matplotlib.pyplot import imshow

def ocrControus( img, boxes,  minWidth=10 ):
    imgs = [];
    charBoxes = []
    for idx, ( x, y, w, h ) in enumerate( boxes ):
        if( w > minWidth ):
            croppedImg = img[y:y+h, x:x+w ]
            croppedImg = fixedWidthImg( croppedImg )
            imgs.append( croppedImg )
            charBoxes.append( boxes[idx] )
    return list( zip( detectGlyph( np.array(imgs) ), charBoxes ) )

def detectGlyphsXY( fname ):
    gd = GlyphExtractor( fname )
    boxes = gd.detectContours()
    return ocrControus( cv2.cvtColor( gd.rgb, cv2.COLOR_BGR2GRAY ), boxes );

def main( fname ):
    glyphList = detectGlyphsXY( fname )
    jsonResult = json.dumps( glyphList, ensure_ascii=False )
    writeFile( './results.js', "var ocrResult = %s ;" % jsonResult )


#  Usage: python3 ./ocr.py <path-to-image-files>
#  Then open test.html to see the results
#  import ipdb; ipdb.set_trace()

if( __name__ == '__main__' ):
    if( len( sys.argv ) != 2 or sys.argv[1] == '--help' ):
        print("Usage: \n./ocr.py <image-file> \n\nThen open test.html in your browser to see the result")
    else:
        main( sys.argv[1] )

