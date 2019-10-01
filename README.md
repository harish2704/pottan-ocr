[![Join the chat at https://gitter.im/pottan-ocr/Lobby](https://badges.gitter.im/pottan-ocr/Lobby.svg)](https://gitter.im/pottan-ocr/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# pottan-ocr

A stupid OCR for malayalam language. It can be Easily configured to process any other languages with complex scripts

## Web Demo of individual line recognition
https://harish2704.github.io/pottan-demo/

## Screenshot of complete page OCR
![Screenshot](https://i.imgur.com/CqeBYox.png)

## Installation

#### Clone the project
```
git clone https://github.com/harish2704/pottan-ocr
cd pottan-ocr
```

#### Run the installer bash script to complete the installation.

* For Debian
  ```bash
  env DISTRO=debian ./tools/install-dependencies.sh
  ```
* For Fedora
  ```bash
  env DISTRO=fedora ./tools/install-dependencies.sh
  ```
* For OpenSUSE
  ```bash
  env DISTRO=opensuse ./tools/install-dependencies.sh
  ```
* For Ubuntu
  ```bash
  ./tools/install-dependencies.sh
  ```

By default, the installer will install dependencies which is necessary to run the OCR. For training the OCR, pass the string `for_training` as first argument to installer.
```bash
  ./tools/install-dependencies.sh for_training
```


## Usage

1. Download [latest pre-trained model][latest_model] file from [pottan-ocr-data][data_repo] repository
  ```bash
  wget 'https://github.com/harish2704/pottan-ocr-data/raw/master/netCRNN_01-19-06-09-54_3.pth' -O ./misc/netCRNN_01-19-06-09-54_3.pth
  ```
2. Create configuration file
  ```bash
  cp ./config.yaml.sample ./config.yaml
  ```
3. Run the OCR using any PNG/JPEG image
  ```bash
  ./bin/pottan ocr ./misc/netCRNN_01-19-06-09-54_3.pth <path_to_image.png>
  ```

For more details, see the `--help` of `bin/pottan` and its subcommands

```
Usage:
./pottan <command> [ arguments ]

List of available commands ( See '--help' of individual command for more details ):

    extractWikiDump - Extract words from wiki xml dump ( most most of the text corpus ). Output is written to stdout.

    datagen         - Prepare training data from data/train.txt & data/validate.txt. ( Depreciated. used only for manual varification of training data )

    train           - Run the training

    ocr             - Run charector recognition with a pre-trained model and image file
```

## OCR usage

```
./bin/pottan ocr <trained_model.h5> <iamge_path> [ pottan_ocr_output.html ]
```


## Training

* For training, we need to install `warp-ctc` and it's pytorch bindings. See `./tools/install-dependencies.sh` for the detailed installation steps
* Training is done using synthetic images generated on the fly using text corpus. For this to work, we should have enough fonts installed in the system. In short, fonts listed in the `./config.yaml.sample` should be available in the output of command `fc-list :lang=ml`
  - It is also possible to write the generated images to disk. sub-command `datagen` does exactly this. When running training, if the images already found to exists in the cache directory( eg: point cache directory to generated images directory ), it will be used for the training instead of generating new images. This idea is used to reduce CPU load during production training sessions
* Also it is recommended to have a GPU for training the OCR.

The currently, models are trained on [Floydhub][floyd_hub_page]. Following details are available there
* Exact command-line options
* raw text data used and generate synthetic images used for training
* logs , timing & progress of each training sessions.


## Getting involved
* Join public Gitter chat room ( See badge on the top ) or Public Matrix chat room `#pottan-ocr:matrix.org` ( https://riot.im/app/#/room/#pottan-ocr:matrix.org ).
* Status, progress & pending tasks can be seen @ https://github.com/harish2704/pottan-ocr/projects/1


## Credits
* Authors of http://arxiv.org/abs/1507.05717
* [jieru mei](https://github.com/meijieru) who created pytorch implementation for above mentioned model. Repo https://github.com/meijieru/crnn.pytorch. The model used in Pottan-OCR is taken from this project.
* [Tom]( https://github.com/tmbdev ) and the contributes of Ocropy project ( https://github.com/tmbdev/ocropy ) which is the back-bone of pottan-ocr.
  - Code-base of pottan-ocr can do only one thing. Just convert a single line of image into single line of text.
  - Everything else including layout detection, line segmentation, output generation etc are handled by **Ocropy**.
  - pottan-ocr just works as core engine by replacing default engine **Tesseract OCR** used in the **Ocropy**
* Pytorch https://pytorch.org/
* [Leon Chen]( https://github.com/transcranial ) and the Team behind [ KerasJS ](https://github.com/transcranial/keras-js).
  - KerasJS is used to create the Web-based demo application of the OCR.
  - KerasJS does its job very well by running Keras Models in browsers with WebGL2 acceleration.
  - It also have great features such as visualizing each stages of process , which is not explored yet.

## Thanks
* [Swathanthra Malayalam Computing](https://smc.org.in/) group members for evaluating and providing suggestions.
* Stackoverflow user "Yu-Yang" for answering [ my question ](https://stackoverflow.com/questions/48361376/converting-state-parameters-of-pytorch-lstm-to-keras-lstm)

[latest_model]: https://github.com/harish2704/pottan-ocr-data/raw/master/netCRNN_01-19-06-09-54_3.pth
[data_repo]: https://github.com/harish2704/pottan-ocr-data
[floyd_hub_page]: https://www.floydhub.com/harish2704/projects/pottan-ocr/3
