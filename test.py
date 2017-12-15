from GlyphExtractor import GlyphExtractor
import cv2
from matplotlib.pyplot import imshow
from utils import readJson

def run(
        image='/home/hari/tmp/ocr-related/hari-utils/cache/rr/yy-027.pgm',
        rect=(1,1),
        eclipse=(2,2),
        threshold=150,
        ):
    gd = GlyphExtractor( image )
    #  cv2.fastNlMeansDenoising( gd.rgb )
    ( grad, bw, connected, conts ) = gd._detectContours( rect=rect, eclipse=eclipse, threshold=threshold )
    gd.markContours( conts )
    return ( grad, bw, connected, gd.rgb )

