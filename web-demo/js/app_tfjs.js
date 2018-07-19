/*
 * test.js
 * Created: Sun Jun 17 2018 23:09:55 GMT+0530 (IST)
 * Copyright 2018 Harish.K<harish2704@gmail.com>
 */

var tf = require('@tensorflow/tfjs');
var jquery = require('jquery');
window.$ = jquery;
window.tf = tf;


tf.setBackend('webgl');
require('cropper');
var glyphsList = require('../data/glyphs.json');
glyphsList.unshift('');
var IMG_HEIGHT = 32;

function initModel(){
  tf.loadModel('../tfjs_exported/model.json')
    .then(function(model){
      window.model = model;
    });
}

initModel();

$(function() {
  var img = $('#for-cropper');
  var output = $('#output');
  var tmpScaleCanvas = document.createElement('canvas');
  tmpScaleCanvas.width = 2048;
  tmpScaleCanvas.height = IMG_HEIGHT;
  var tmpScaleCanvasCtx = tmpScaleCanvas.getContext('2d');
  var imgCropper = img.cropper();

  function initUi() {

    $(".input-file").before(
      function() {
        if ( !$(this).prev().hasClass('input-ghost') ) {
          var element = $("<input type='file' class='input-ghost' accept=\"image/*\" style='visibility:hidden; height:0'>");
          element.attr("name",$(this).attr("name"));
          element.change(function( e ){
            var imgUrl = URL.createObjectURL(e.target.files[0]);
            imgCropper.data('cropper').replace( imgUrl );
            element.next(element).find('input').val((element.val()).split('\\').pop());
          });
          $(this).find("button.btn-choose").click(function(){
            element.click();
          });
          $(this).find('input').css("cursor","pointer");
          $(this).find('input').mousedown(function() {
            $(this).parents('.input-file').prev().click();
            return false;
          });
          return element;
        }
      }
    );

    $('#rotate-buttons > button').click(function(){
      imgCropper.data('cropper').rotate( $(this).data().degree )
    });
  }




  function decodeStr( strEnc, raw=false ){
    strEnc = strEnc.argMax(2)
    var numPredictions = strEnc.shape[1]
    var out = [];
    var val;
    for (var i = 0; i < numPredictions; i ++) {
      val = strEnc.get(0,i);
      if(  val !== strEnc.get(0, i-1 ) ){
        out.push( glyphsList[val] )
      }
    }
    return out.join('');
  }

  initUi();
  $('#run-btn').click(function(){
    if( !imgCropper ){
      return ;
    }
    var croppedCanvas = imgCropper.data('cropper').getCroppedCanvas();

    var scaleFactor = IMG_HEIGHT/croppedCanvas.height;
    var origWidth = croppedCanvas.width;
    var newWidth = parseInt( scaleFactor * origWidth );
    tmpScaleCanvasCtx.drawImage( croppedCanvas, 0,0, newWidth, IMG_HEIGHT );
    var imageDataScaled = tmpScaleCanvasCtx.getImageData(0,0, newWidth, IMG_HEIGHT );

    var imgData = tf.fromPixels( imageDataScaled, 4 );
    imgData = imgData.slice([0,0,1], [ -1, -1, 1 ]).expandDims().toFloat()
    // imgData = tf.sub( imgData, tf.scalar(127.5) );
    imgData = tf.div( imgData, tf.scalar(255) );
    tf.toPixels(
      imgData.squeeze()
        .add( tf.scalar(1) )
        .div( tf.scalar(2) ),
      $('#dbg')[0]
    );


    var outputData = model.predict( imgData )
    var out = decodeStr( outputData );
    output.text( out );
  });
});

