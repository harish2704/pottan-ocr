/*
 * js/app.js
 * Created: Thu Jan 25 2018 01:17:09 GMT+0530 (IST)
 * Copyright 2018 Harish.K<harish2704@gmail.com>
 */
var jquery = require('jquery');
var tf = require('@tensorflow/tfjs');
var glyphsList = require('../data/glyphs.json');

window.tf = tf;
window.$ = jquery;

glyphsList.push('');
var IMG_HEIGHT = 32;

require('cropper');

tf.enableProdMode ()


$(function() {
  var img = $('#for-cropper');
  var output = $('#output');
  var imgCropper = img.cropper().data('cropper');
  var debugCheckbox = $('#cb_debug');

  function initUi() {

    document.onpaste = function(event){
      var items = (event.clipboardData || event.originalEvent.clipboardData).items;
      for (var index in items) {
        var item = items[index];
        if (item.kind === 'file') {
          var blob = item.getAsFile();
          var reader = new FileReader();
          reader.onload = function(event){
            imgCropper.replace( event.target.result );
          };
          reader.readAsDataURL(blob);
        }
      }
    };

    img.on('ready', function(){
      imgCropper.setCropBoxData({left: 208.5, top: 49, width: 211, height: 41});
    });

    $('#cb_backend').on('change', function( ev ){
      tf.setBackend( event.target.checked ? 'webgl' :'cpu' );
    });

    $(".input-file").before(
      function() {
        if ( !$(this).prev().hasClass('input-ghost') ) {
          var element = $("<input type='file' class='input-ghost' accept=\"image/*\" style='visibility:hidden; height:0'>");
          element.attr("name",$(this).attr("name"));
          element.change(function( e ){
            var imgUrl = URL.createObjectURL(e.target.files[0]);
            imgCropper.replace( imgUrl );
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
      imgCropper.rotate( $(this).data().degree )
    });
  }


  function decodeStr( strEnc, raw=false ){
     return strEnc.map((v,i) => {
      return ( raw || ( strEnc[i-1] !== v )) ? glyphsList[ v ] : '';
    }).join('');
  }

  tf.loadLayersModel('./data/pottan_min/model.json').then( model =>{
    window.model = model;
    IMG_HEIGHT = model.layers[0].batchInputShape[1];
    // This is hack to trace outputs of each layer after execution
    model.layers.forEach(function( l ){
      l.__orig_call = l.call;
      l.call = function( inputs, args ){
        var out = this.__orig_call( inputs, args );
        this.__hari_lastout = tf.keep( out.clone() );
        return out;
      }
    });
  })

  initUi();
  $('#run-btn').click(function(){
    if( !imgCropper ){
      return ;
    }
    var croppedCanvas = imgCropper.getCroppedCanvas();
    var scaleFactor = IMG_HEIGHT/croppedCanvas.height;
    var origWidth = croppedCanvas.width,
      newWidth = parseInt( scaleFactor * origWidth );
    var data = tf.browser.fromPixels( croppedCanvas );

    // pick any one of the color channel. ( Here it is Green )
    data = data.gather([0], [2])
    data = tf.image.resizeBilinear( data, [ IMG_HEIGHT, newWidth ] );

    // Normalize the values . Fit between -1 & 1
    data = data.sub( 127.5 );
    data = data.div( 127.5 );

    var start = new Date();
    var outputData = model.predict( data.expandDims() );
    var predictions = outputData.argMax(2).squeeze();
    tf.print( predictions,1 );
    var out = decodeStr( Array.from( predictions.dataSync()) );
    output.text( out );
    if( debugCheckbox.prop("checked") ){
      renderLayerOutputs( model );
    }
  });
});

async function renderLayerOutputs( model ){
  var tfvis = require('@tensorflow/tfjs-vis');
  window.tfvis = tfvis;
  console.log(`Total num layers: ${model.layers.length}`);
  for (var i = 0; i < model.layers.length; i ++) {
    var layer = model.layers[i];

    var surface = tfvis.visor().surface({ name: 'Layer outputs', tab: layer.name });
    var out = layer.__hari_lastout.squeeze();
    // Normalize the data with in the range 0 - 1.0
    out = out.sub( out.min() );
    out = out.div( out.max());

    if( out.shape.length > 2 ){
      await renderImage( surface.drawArea, out.transpose( [2,0,1] ).reshape( [ out.shape[1], out.shape[0]*out.shape[2] ].reverse() ) );
    } else {
      await renderImage( surface.drawArea, out.transpose([1,0]) );
    }
  }
}

async function renderImage(container, tensor ) {
  const canvas = document.createElement('canvas');
  canvas.width = tensor.shape[0];
  canvas.height = tensor.shape[1];
  canvas.style = `margin: 2px;height: ${tensor.shape[0]*4}px; width: ${tensor.shape[1]*4}px;image-rendering: pixelated;`;
  container.appendChild(canvas);
  return await tf.browser.toPixels( tensor, canvas);
}
