import json

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
