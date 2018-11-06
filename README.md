# Deep Transfer Learning with PyTorch
An easy way to test out ANY PyTorch model using transfer learning and a flowers dataset. This was part of the final project for the Udacity AI Nanodegree course.

## Installation
* Python 3.6
* Clone this repo and run `pip install -r requirements.txt`

## Usage train.py
Train a new network on a data set with train.py

*Note:* If possible use a GPU for training or be prepared to wait a long time.
Network training progress will be printed to the command line.

Basic usage: `python train.py data_directory`
* Prints out training loss, validation loss, and validation accuracy as the network trains
* Help: `python train.py -h`

#### Options:
* Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
* Choose architecture: `python train.py data_dir --arch "vgg13"`
* Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
* Use GPU for training: `python train.py data_dir --gpu`

## Usage predict.py
* Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
* Help: `python predict.py -h`

Basic usage: python predict.py /path/to/image checkpoint
#### Options:
* Return top K most likely classes: `python predict.py input checkpoint --top_k 3`
* Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
* Use GPU for inference: `python predict.py input checkpoint --gpu`
