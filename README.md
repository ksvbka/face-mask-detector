# Face Mask Detection

![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)

Detecting face mask with OpenCV and TensorFlow. Using simple CNN or model provided by TensorFlow as MobileNetV2, VGG16, Xception.

![Demo](doc/8.jpg)

## Data

Raw data collected from kaggle and script crawl_image.py, split to 'Mask' and 'Non Mask' class.

Using build_data.py to extract faces from raw dataset and resize to 64x64.

## Installation

Clone the repo

```
git clone git@github.com:ksvbka/face-mask-detector.git
```
cd to project folder and create virtual env

```
virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
```

Download raw dataset and execute script build_dataset.py to preprare dataset for training
```
cd data
bash download_data.sh
cd -
python3 build_dataset.py --data-dir data/64x64_dataset --output-dir data/dataset_raw/
```
## Training

Execute train.py script and pass  network architecture type dataset dir and epochs to it.
Default network type is MobileNetV2.
```
python3 train.py --net-type MobileNetV2 --data-dir data/64x64_dataset --epochs 20
```
View tensorboard
```
tensorboard --logdir logs --bind_all
```
## Testing

```
python3 mask_detect_image.py -i demo_image/2.jpg
```
## Demo

![Demo](doc/1.jpg)
![Demo](doc/2.jpg)
![Demo](doc/3.jpg)
![Demo](doc/4.jpg)
![Demo](doc/5.jpg)
![Demo](doc/6.jpg)
![Demo](doc/8.jpg)
![Demo](doc/9.jpg)
![Demo](doc/10.jpg)

