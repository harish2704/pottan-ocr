/*
 * ഓം ബ്രഹ്മാർപ്പണം 
 * app.js
 * Created: Mon Mar 09 2020 02:16:15 GMT+0530 (GMT+05:30)
 * Copyright 2020 Harish.K<harish2704@gmail.com>
 */


$(function() {
  var canvas = $('#for-cropper')[0];
  var fileInput = $(".input-file");
  var currentFile;

  function loadXHR(url) {

    return new Promise(function(resolve, reject) {
      try {
        var xhr = new XMLHttpRequest();
        xhr.open("GET", url);
        xhr.responseType = "blob";
        xhr.onerror = function() {reject("Network error.")};
        xhr.onload = function() {
          if (xhr.status === 200) {resolve(xhr.response)}
          else {reject("Loading error:" + xhr.statusText)}
        };
        xhr.send();
      }
      catch(err) {reject(err.message)}
    });
  }

  function setImage( file ){
    currentFile = file;
    var ctx = canvas.getContext('2d');
    var img = new Image;
    img.onload = function() {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    }
    img.src = URL.createObjectURL( file );
  }

  function drawBoxes( data ){
    var lines = data.lines;
    lines.forEach(function(line){
      var ctx = canvas.getContext('2d');
      var width = line[2] - line[0];
      var height = line[3] - line[1];
      ctx.beginPath();
      ctx.lineWidth = "4";
      ctx.strokeStyle = "red";
      // ctx.rotate( item.rot );
      ctx.rect( line[0], line[1], width, height);
      ctx.stroke();
      // ctx.rotate(0-item.rot);
    })
  }


  function initUi() {
    document.onpaste = function(event){
      var items = (event.clipboardData || event.originalEvent.clipboardData).items;
      for (var index in items) {
        var item = items[index];
        if (item.kind === 'file') {
          var blob = item.getAsFile();
          setImage( blob )
        }
      }
    };

    fileInput.before(
      function() {
        if ( !$(this).prev().hasClass('input-ghost') ) {
          var element = $("<input type='file' class='input-ghost' accept=\"image/*\" style='visibility:hidden; height:0'>");
          element.attr("name",$(this).attr("name"));
          element.change(function( e ){
            setImage(e.target.files[0]);
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

    $('#run-btn').click(function(){
      var data = new FormData();
      data.append('image', currentFile );
      $.ajax({
        type: "POST",
        url: "http://localhost:5544/ocr",
        success: function (data) {
          console.log( 'success', data );
          drawBoxes( data );
          $('#output').text( data.text.join('\n') )
        },
        error: function (error) {
          console.log( 'error', data );
        },
        data: data,
        processData: false,
        contentType: false
      });
    })

    loadXHR('./data/sample.png')
    .then(function(blob) {
      setImage(blob);
    });
  }



  initUi();
});
