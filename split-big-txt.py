#!/usr/bin/env python

import sys
fname = sys.argv[1]
suffix = sys.argv[2]

fd = open( fname, 'r' )
contents = fd.read()
fd.close()

contents = contents.split( '\n')
i=1
for line in contents:
    if line:
        fd = open( suffix + '%03d.txt' % i, 'w')
        fd.write( line )
        fd.close()
        i=i+1
