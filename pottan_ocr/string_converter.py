
import re
from torch import IntTensor
from .utils import config

glyphList = config['glyphs']
glyphList.sort( key=lambda x: len(x), reverse=True);

#  escape chars like ", ; ) ( ] [ ? *" for regex
glyphSearchRe =  '|'.join( [ re.escape(i) for i in glyphList ] )

glyphSearchRe = re.compile( '(%s)' % glyphSearchRe)

# Empty string stands for blank
glyphList.insert(0, '')
totalGlyphs = len( glyphList )

def encodeStr( word ):
    glyphs = filter( None, glyphSearchRe.split( word ) )
    out = [ glyphList.index( g ) for g in  glyphs ]
    return out, len( out )

def encodeStrList( items ):
    txt, l = list( zip( *[ encodeStr(i) for i in items ] ) )
    return IntTensor( sum(txt, []) ), IntTensor( l )
encode = encodeStrList



def decodeStr( strEnc, raw=True ):
    glyphs = [ ( glyphList[ g ] if raw or strEnc[ idx-1 ] != g else '') for idx,g in enumerate(strEnc) ]
    return ''.join( glyphs )

def decodeStrList( mergedPres, sizeArr, raw=True ):
    i=0
    out =  []
    for size in sizeArr:
        end = i+size
        out.append( decodeStr( mergedPres[i:end], raw ) )
        i = end
    return out
decode = decodeStrList
