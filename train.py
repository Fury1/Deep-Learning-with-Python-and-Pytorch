import os.path
import argparse
import functions

from sys import exit


# Create an argparse object with the required arguments
parser = argparse.ArgumentParser(description='''Easily train a Neural Network classifier of your choice on a flowers dataset
                                                with transfer learning from PyTorch models''',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data_directory', help='directory of the data to be used')
parser.add_argument('--save_dir', default=os.path.abspath(os.path.dirname(__file__)), help='directory to save checkpoints to')

parser.add_argument('--arch',
                    default='resnet152',
                    choices=['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
                             'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                             'squeezenet1_0', 'squeezenet1_1', 'densenet121', 'densenet169', 'densenet161', 'densenet201',
                             'inception_v3'],
                    help='PyTorch architecture to use for transfer learning (See torchvision.models in PyTorch docs)')

parser.add_argument('--learning_rate', default=0.001, type=float, help='set the learning rate hyperparameter')
parser.add_argument('--batch_size', default=64, type=int, help='set the training batch size')
parser.add_argument('--hidden_units', default=500, type=int, help='set the number of hidden units')
parser.add_argument('--epochs', default=4, type=int, help='set the number of epochs to train for')
parser.add_argument('--gpu', action='store_true', help='train on a GPU')

# Get the args
args = parser.parse_args()

# Get some data ready for a PyTorch model
train_data, val_data, test_data, class_to_idx  = functions.import_data(args.data_directory,
                                                                       args.batch_size,
                                                                       args.arch)

# Present the user settings for review
print()
print("Data Directory: {}".format(args.data_directory))
print("Save Directory: {}".format(args.save_dir))
print("Use GPU: {}".format(args.gpu))
print("PyTorch Architecture: {}".format(args.arch))
print()
print("<" + "-" * 10, "Hyperparameters", "-" * 10 + ">" )
print("Hidden Units: {}".format(args.hidden_units))
print("Learning Rate: {}".format(args.learning_rate))
print("Epochs: {}".format(args.epochs))
print("Batch Size: {}".format(args.batch_size))
print()

# See if the user would like to continue based on the settings
while True:
    inp = input("Do the settings above look correct? [y/n]: ")
    if inp.lower() == 'y':
        break
    else:
        exit("Adjust the settings and retry again, exiting.")

# Get the class-->name mappings
cat_names = functions.get_category_names('cat_to_name.json')

# Build a transfer learning model and optimizer with user defined settings
model, optimizer = functions.build_model(args.arch, args.hidden_units, args.learning_rate)

# Train the model
model = functions.train(model, train_data, val_data, optimizer, args.epochs, args.gpu)

# Save the model
functions.save(model, args.arch, class_to_idx, cat_names, args.save_dir)