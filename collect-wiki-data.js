/*
 * collect-wiki-data.js
 * Created: Mon Jan 01 2018 14:28:15 GMT+0530 (IST)
 * Copyright 2018 Harish.K<harish2704@gmail.com>
 */

var fs = require('fs');
var Transform = require('stream').Transform;
var infile = process.argv[2];

var symbolsTobeIncluded = [
' ',
  '!',
  '"',
  '#',
  '$',
  '%',
  '&',
  "'",
  '(',
  ')',
  '*',
  '+',
  ',',
  '-',
  '.',
  '/',
  ':',
  ';',
  '<',
  '=',
  '>',
  '?',
  '@',
  '[',
  '\\',
  ']',
  '^',
  '_',
  '`',
  '{',
  '|',
  '}',
  '~',
].join('');

// Escape special chars used in regex
symbolsTobeIncluded = symbolsTobeIncluded.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');

if( !infile || ( infile === '--help' ) ){
  console.log('Usage: node ./collect-wiki-data.js <wiki_dump.xml>');
  process.exit(-1);
}

var preProcessPatterns = [
'<[a-zA-Z\ \-_\"\']+>',
'<\/[a-zA-Z]+>',
'\s{2,}',
'&[^\s]*;',
'\\\>',
'[\\[\\]\\(\\)\\{\\}\'"\-=\.\*_]{2,}',
];


// Replace all non zwj/zwnj
// Ref http://thottingal.in/blog/2017/05/27/a-formal-grammar-for-malayalam-syllables/
var unwantedZwj = new RegExp( '\([^്]|^\)[‌‍]+' , 'g' );

var zwjMapping = {
  'ല്‍': 'ൽ',
  'ന്‍': 'ൻ',
  'ണ്‍': 'ൺ',
  'ര്‍': 'ർ',
  'ക്‍': 'ൿ',
};
var zwjRegex = '('+Object.keys(zwjMapping).join('|')+')';
zwjRegex = RegExp( zwjRegex, 'g');

var preProcessPatterns = '('+ preProcessPatterns.join('|')+')';
preProcessPatterns = new RegExp( preProcessPatterns, 'g');

var matchingPattern = new RegExp('[0-9\u0d00-\u0d7f\u200C\u200D][0-9\u0d00-\u0d7f\u200C\u200D'+ symbolsTobeIncluded +']{2,}', 'g' );
var whiteSpace = new RegExp('\\s+', 'g' );

var parser = new Transform();
parser.lastReadedLine = '';
parser._transform = function(data, encoding, done) {
  data = data.toString();
  data = data.replace( preProcessPatterns, '')
  data = data.replace( unwantedZwj, '$1');
  data = data.replace( zwjRegex, function( a ){
    return zwjMapping[a] || '';
  });
  var match = data.match( matchingPattern );
  if( !match ){
    done();
    return;
  }

  // Convert long sentance into array of words
  match = [].concat.apply( [], match.map(v=> v.split( whiteSpace ) ) );

  var i=0,
    sentance=this.lastReadedLine,
    word,
    total = match.length;

  // Generate lines with maximum length N
  while( i< total ){
    word = match[ i ];
    // if( word === ''){ debugger; }

    
    if( ( sentance.length + word.length ) > 55 ){
      this.push( sentance + '\n' );

      // if this is a single long word. Ignore it
      sentance = word.length > 55 ?  '' : word ;
    } else if(word){
      sentance = sentance + ' ' + word
    }
    i++
  }
  this.lastReadedLine = sentance;
  done();
};

reader = fs.createReadStream( infile )
  .pipe( parser )
  .pipe( process.stdout )
reader.on('error', function(e){
  console.error( e );
});

