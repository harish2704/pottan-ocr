
from matplotlib.pyplot import imshow as imshow_

def imshow( img ):
    imshow_( img, cmap='gray')

def gallery(array, ncols=3):
    array = array.numpy()
    nindex, height, width  = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width )
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols))
    return result
