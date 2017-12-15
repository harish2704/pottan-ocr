#!/usr/bin/env node

var fs = require('fs');

var glyphList = fs.readFileSync( process.argv[2], 'utf-8' ).split('\n').filter(Boolean);
var gyphLabelList = fs.readFileSync( process.argv[3], 'utf-8' ).split('\n');
var labelOverride = {
'0': '0',
'1': '1',
'2': '2',
'3': '3',
'4': '4',
'5': '5',
'6': '6',
'7': '7',
'8': '8',
'9': '9',

'൦': '_ml0',
'൧': '_ml1',
'൨': '_ml2',
'൩': '_ml3',
'൪': '_ml4',
'൫': '_ml5',
'൬': '_ml6',
'൭': '_ml7',
'൮': '_ml8',
'൯': '_ml9',
};

out = [];

glyphList.forEach(function( glyph, i ){
  label = labelOverride[glyph] || gyphLabelList[i];
  out.push( [ glyph, label ] );
  // fs.writeFileSync( `./ml-glyphs/text/${ label }.txt`, glyph );
});

fs.writeFileSync( process.argv[4], JSON.stringify(out, null, 1) );
