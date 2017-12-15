#!/usr/bin/env python3

import os
from os import path
import cv2
import numpy as np
from matplotlib.pyplot import imshow
from utils import fixedWidthImg


def detectContoursFromImg( rgb,
            eclipse=(2,2),
            rect=(1,1),
            threshold=150 ):
    #  _, small = cv2.threshold( small, threshold, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, eclipse )
    grad = cv2.morphologyEx( rgb, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, threshold, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, rect )
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    #  imshow( connected )
    _,contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )

    #  Filter top level contours only
    topC = []
    for idx, h in enumerate(hierarchy[0]):
        if( h[3] == -1 ):
            topC.append( contours[idx] )
    return ( grad, bw, connected, topC )

def markContoursImg( rgb, contours, minWidth=11 ):
    #  mask = np.zeros( rgb.shape, dtype=np.uint8)
    for idx, ( x, y, w, h ) in enumerate(contours):
        #  mask[y:y+h, x:x+w] = 0
        #  cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        #  r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
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
        return ( grad, bw, connected, contours )

    def detectContours( self, **kwargs ):
        ( grad, bw, connected, contours ) = self._detectContours( **kwargs )
        return contours

    def extractContours( self, contours, outdir='./cache/extracted', minWidth=11 ):
        os.makedirs( outdir, exist_ok=True )
        rgb = cv2.cvtColor( self.rgb, cv2.COLOR_BGR2GRAY)
        for idx, ( x, y, w, h ) in enumerate(contours):
            if( w > minWidth ):
                croppedImg = rgb[y:y+h, x:x+w ]
                cv2.imwrite( '%s/%s_%04d.png'%( outdir, self.imageFileBaseName, idx ), fixedWidthImg( croppedImg ) )

    def markContours( self, contours, **kwargs ):
        return markContoursImg( self.rgb, contours, **kwargs )

