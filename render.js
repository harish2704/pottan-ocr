/*
 * render.js
 * Created: Sun Dec 24 2017 03:50:28 GMT+0530 (IST)
 * Copyright 2017 Harish.K<harish2704@gmail.com>
 */


var container = document.getElementById('output');
var vSorted = ocrResult.sort( function( a, b ){
  return a[1][1] - b[1][1];
})
var vDelta = vSorted.map(function( v, i ){
  return v[1][1]- vSorted[i+1]
})
container.innerHTML = ocrResult.map(function( [ ch, box ]){
  return '<span style=" top: '+ box[1]+'px; left: '+box[0]+'px;">'+ ch +'</span>';
}).join('\n');

