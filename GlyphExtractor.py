#!/usr/bin/env python3

import os
from os import path
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


def detectContoursFromImg( rgb,
            eclipse=(2,2),
            rect=(1,1),
            threshold=150,
            chain=cv2.CHAIN_APPROX_NONE,
            retr=cv2.RETR_EXTERNAL ):
    #  _, small = cv2.threshold( small, threshold, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, eclipse )
    grad = cv2.morphologyEx( rgb, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, threshold, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, rect )
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    #  imshow( connected )
    # using RETR_EXTERNAL instead of RETR_CCOMP
    _,contours, hierarchy = cv2.findContours(connected.copy(), retr, chain )
    return ( grad, bw, connected, contours )

def markContoursImg( rgb, contours, minWidth=11 ):
    #  mask = np.zeros( rgb.shape, dtype=np.uint8)
    for idx, ( x, y, w, h ) in enumerate(contours):
        #  mask[y:y+h, x:x+w] = 0
        #  cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        #  r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
        print ( x,y, w, h, 'Contour ')
        if w>minWidth:
            cv2.rectangle(rgb, (x, y), (x+w, y+h), (255, 0, 0), 1)

class GlyphExtractor(object):
    """Extract glyphs from a image"""
    def __init__(self, imageFile ):
        super(GlyphExtractor, self).__init__()
        self.imageFile  = imageFile 
        self.imageFileBaseName = path.basename( imageFile )
        self.rgb = cv2.imread( imageFile )

    def _detectContours( self, **kwargs ):
        rgb = cv2.cvtColor( self.rgb, cv2.COLOR_BGR2GRAY)
        ( grad, bw, connected, contours ) = detectContoursFromImg( rgb, **kwargs )
        contours = [ cv2.boundingRect( contour ) for contour in contours ];
        contours = filterInnerContours( contours )
        return ( grad, bw, connected, contours )
    def detectContours( self, **kwargs ):
        ( grad, bw, connected, contours ) = self._detectContours( kwargs )
        return contours

    def extractContours( self, contours, outdir='./glyp-extracted', minWidth=11 ):
        os.makedirs( outdir, exist_ok=True )
        rgb = self.rgb
        for idx, ( x, y, w, h ) in enumerate(contours):
            if( w > minWidth ):
                croppedImg = rgb[y:y+h, x:x+w ]
                cv2.imwrite( '%s/%s_%04d.png'%( outdir, self.imageFileBaseName, idx ), croppedImg )

    def markContours( self, contours, **kwargs ):
        return markContoursImg( self.rgb, contours, **kwargs )

