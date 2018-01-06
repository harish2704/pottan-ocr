/*
 * collect-wiki-data.js
 * Created: Mon Jan 01 2018 14:28:15 GMT+0530 (IST)
 * Copyright 2018 Harish.K<harish2704@gmail.com>
 */

var bigXml = require('big-xml');
var infile = process.argv[2];
if( !infile || ( infile === '--help' ) ){
  console.log('Usage: node ./collect-wiki-data.js <wiki_dump.xml>');
  process.exit(-1);
}

var reader = bigXml.createReader( infile, /^(text)$/ );

var i =0;
var preProcess1 = new RegExp( '[\\[\\]\(\)\{\}\'"\-=\.\*_]{2,}', 'gm' );
var nonMl = new RegExp('[^\\[\\]\-_\?\;!\:\'\"\.\{\}0-9\u0d00-\u0d7f\u200C\u200D\ ]+', 'gm' );
var out = [];
reader.on('record', function(record) {
  // console.log( i );
  // if( i >= 0 ){
  out.push( record.text );
  // }
  i++;
});

reader.on('end', function(){
  out.forEach(function( str ){
    if( str ){
      str = str.replace( preProcess1, '' );
      str = str.replace( nonMl, ' ' );
      console.log( str );
    }
  });
  console.error('Completed... ' + i + ' Pages' );
});

reader.on('error', function( err ){
  console.error('Err', err );
});

