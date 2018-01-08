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
  '‌{2,}',
  '‍{2,}',
  '\s‍',
'ൽ‍',
'ൻ‍',
'ൺ‍',
'ർ‍',
];

var zwjMapping = {
  'ല്‍': 'ൽ',
  'ന്‍': 'ൻ',
  'ണ്‍': 'ൺ',
  'ര്‍': 'ർ',
  'ക്‍': 'ൿ',
  '‌+$': '',
  '‌\|': ' |',
  '‌\.': ' .',
  '‌ഷ': 'ഷ',
  'ാ‌':    'ാ',
};
var zwjRegex = '('+Object.keys(zwjMapping).join('|')+')';
zwjRegex = RegExp( zwjRegex, 'g');

var preProcessPatterns = '('+ preProcessPatterns.join('|')+')';
preProcessPatterns = new RegExp( preProcessPatterns, 'g');
var matchingPattern = new RegExp('[\u0d00-\u0d7f\u200C\u200D][\u0d00-\u0d7f\u200C\u200D'+ symbolsTobeIncluded +']{2,}', 'g' );
var whiteSpace = new RegExp('\\s+', 'g' );

var parser = new Transform();
parser.lastReadedLine = '';
parser._transform = function(data, encoding, done) {
  data = data.toString();
  data = data.replace( preProcessPatterns, '')
  data = data.replace( zwjRegex, function(a, b, c ){
    return zwjMapping[a] || '';
  });
  var match = data.toString().match( matchingPattern );
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

  // Generate lines with maximum length 90
  while( i< total ){
    word = match[ i ];
    if( ( sentance.length + word.length ) > 90 ){
      if( sentance === "അല്മോദിബ്ളാഥയീമിൽ പാളയമിറങ്ങി. അല്മോദിബ്ളാഥയീമിൽനിന്നു പുറപ്പെട്ടു നെബോവിന്നു കിഴക്കു"){
        console.log( word );
        debugger;
      }
      this.push( sentance + '\n' );
      sentance = word;
    } else if(word){
      // if( sentance[ sentance.length-1] === ' ' ){ debugger; }
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

