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

'൦': 'ml0',
'൧': 'ml1',
'൨': 'ml2',
'൩': 'ml3',
'൪': 'ml4',
'൫': 'ml5',
'൬': 'ml6',
'൭': 'ml7',
'൮': 'ml8',
'൯': 'ml9',

'ം': '@m',
'ഃ': '@H',
};

out = [];

glyphList.forEach(function( glyph, i ){
  label = labelOverride[glyph] || gyphLabelList[i];
  out.push( [ glyph, label ] );
  // fs.writeFileSync( `./ml-glyphs/text/${ label }.txt`, glyph );
});

fs.writeFileSync( process.argv[4], JSON.stringify(out, null, 1) );
