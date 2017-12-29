import json
import cv2
import numpy as np

FINAL_W=32
FINAL_H=32

def readFile( fname ):
    with open( fname, 'r') as f:
        return f.read()

def readJson( fname ):
    with open( fname, 'r' ) as f:
        return json.load( f )

def writeFile( fname, contents ):
    with open( fname, 'w' ) as f:
        f.write( contents )

def writeJson( fname, data ):
    with open( fname, 'w') as outfile:
        json.dump(data, outfile)

def fixedWidthImg( img ):
    ( currentH, currentW) = img.shape
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
    resizedImg = cv2.resize( img, ( newWidth, newHeigh ), interpolation=cv2.INTER_CUBIC )
    canvasImg = np.ones( ( FINAL_H, FINAL_W ), dtype=np.uint8)*255
    canvasImg[offsetY:offsetY+newHeigh, offsetX:offsetX+newWidth] = resizedImg
    return canvasImg
