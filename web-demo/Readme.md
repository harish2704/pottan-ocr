### What is this?

This is a static client side web-application which can run trained models of OCR inside a web browser. See [Demo](http://harish2704.github.io/pottan-demo/)

### How it is working ?

* It is using [ Keras-js ]( https://github.com/transcranial/keras-js ) to run trained models.
* Originally, this recognition engine uses a [Model written in PyTorch](../pottan_ocr/model.py) for training and recognition. It is converted to Keras model by using following steps.
  - Written a equivalent model in [Keras](https://keras.io/). It can be found [here](../misc/keras_model.py)
  - Weights from trained PyTorch model is transfered to Keras model using [this script](../tools/torch_to_keras.py).
  - Now, the prepared the Keras model as per [KerasJs documentation](https://transcranial.github.io/keras-js-docs/) for loading it inside web.

### What are the limitations ?

* Since this is just a core engine of OCR, There are limitations for providing a complete demo. They are
  - This OCR engine depends on its previous layers for layout analysis & text line separation.
  - In short, it can only accept a single line of image data.
  - It is not easy to port layout analysis tool written in Python+PIL+OpencCv to port into client side web. So, user have to manually select a single line of image from by using image crop GUI available in the demo page.
* Even though KerasJs can run on both CPU & GPU, CPU version seems to broken and found not working properly. So it is disabled. In short, this can only run in web-browser with WebGL-2 support.


#### Why put effort to make a web front-end which do not provide any useful result other than simple demo ?

* In future, it can be converted as a platform were users can
  - Evaluate the current state of OCR recognition,
  - If they found any correction, then they can submit the correct form through web interface. It will be added to training dataset after some spam removal steps.
  - By this way, we can continuously improve the accuracy of the engine.

