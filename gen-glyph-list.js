#!/usr/bin/env node

var fs = require('fs');
var _l = function charList( str ){
  return str.split('-');
};

function crossProd( a, b ){
  var twoDimList = b.map( b1 => a.map( a1 => a1+b1 ) );
  return Array.prototype.concat.apply( [], twoDimList );
}

var vyanjanasHasDouble = _l('ക-ഗ-ങ-ച-ജ-ഞ-ട-ഡ-ണ-ത-ദ-ന-പ-ബ-മ' + '-യ-ല-വ-ശ-സ-ള-റ' );
var misc = _l( 'ങ്ക-ണ്ട-ക്ഷ-ഷ്ട-ത്മ-ന്ധ-ഞ്ച-ല്പ-ഹ്ന-ജ്ഞ-ശ്ച-ന്ത-ണ്ഡ-സ്ഥ-ബ്ദ-മ്പ-ന്റ').concat(
  crossProd( _l('ക-ഗ-ങ-ച-ജ-ഞ-ട-ഡ-ണ-ത-ദ-ധ-ന-പ-ബ-ഭ-മ' + '-യ-ല-വ-ശ-സ-ള-റ' ), ["്ല"] ),
  crossProd( _l('ക-ന-പ-ബ-മ' + '-സ' + '-ണ്ട-ന്ധ-ന്ത-ണ്ഡ-മ്പ' ), ["്ര"] ),
  crossProd( _l('ക-പ' + '-യ-ല-സ' ), ["്ത"] ),
  crossProd( _l('ക-ഗ-ത-ധ-പ-ബ-മ' + '-യ-ശ-സ' ), ["്ന"] ),
);
var chihnnasZWJ = _l('ു-ൂ');

var allGlyphs = {
  swaras:  _l('അ-ആ-ഇ-ഉ-ഋ-ൠ-ഌ-ൡ-എ-ഏ-ഒ'),
  vyanjanas: _l('ക-ഖ-ഗ-ഘ-ങ-ച-ഛ-ജ-ഝ-ഞ-ട-ഠ-ഡ-ഢ-ണ-ത-ഥ-ദ-ധ-ന-പ-ഫ-ബ-ഭ-മ' + '-യ-ര-ല-വ-ശ-ഷ-സ-ഹ-ള-ഴ-റ'),
  chillu:  _l('ൾ-ൽ-ൻ-ർ-ൺ'),
  chihnnas:   ["്", "ാ", "ി", "ീ", "ു", "ൂ", "ൃ",
    'ോ', 'ൊ' ,'ഈ', 'ൈ' , 'ഊ', 'ഓ', 'ഐ',  'ൌ', 'ൎ', 'ഔ',
    // "ൢ", "ൣ",
    "ൄ", "െ", "േ", "ൗ",
    "ം", "ഃ"
     ],
    numbers: _l('൦-൧-൨-൩-൪-൫-൬-൭-൮-൯' )
    // ( '-൰-൱-൲-൳-൴-൵'),
  };


var chandrakakala = allGlyphs.chihnnas[ 0 ];

function genDouble( gl ){
  return gl + chandrakakala + gl;
};
allGlyphs.hardVyanjanas = vyanjanasHasDouble.map( genDouble );

var chihnnasToBeExcluded = ["ങ്ല", "ച്ല", "ജ്ല", "ഞ്ല", "ട്ല", "ഡ്ല", "ണ്ല", "ദ്ല", "ധ്ല", "ന്ല", "ഭ്ല", "യ്ല", "ള്ല", "റ്ല", "ബ്ത", "ത്ന", "ല്ന", "വ്ന", "ദ്ന", "മ്ത"];

allGlyphs.withChinnas = crossProd(
  allGlyphs.vyanjanas.concat( misc, allGlyphs.hardVyanjanas ).filter( v=> chihnnasToBeExcluded.indexOf(v) === -1 ),
  chihnnasZWJ );


allItems = [].concat(
  allGlyphs.swaras,
  allGlyphs.vyanjanas,
  allGlyphs.chillu,
  allGlyphs.chihnnas,
  allGlyphs.hardVyanjanas,
  misc,
  allGlyphs.withChinnas,
  allGlyphs.numbers,
  _l('0-1-2-3-4-5-6-7-8-9'),
);

var misc = [
'്വ',
// '‍്‍വ',
  '.',
  '"',
  "'",
  '(', ')',
  '?',
  '!',
  '+',
  '-',
  '_',
  ',',
  ';',
  ':',
  '[',']',
  '/',
  '\\',
  'ഽ', 
  'ൿ',
  '൹' ,
  '{','}',
  '<', '>',
  '=',
  ',',
  '&',
  '#',
  ];

fs.writeFileSync( './data/glyphs.json', JSON.stringify( allItems.concat(misc), null, 1 ) );

