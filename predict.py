import argparse
import functions


# Create an argparse object with the required arguments
parser = argparse.ArgumentParser(description='''Make a flower prediction from a PyTorch checkpoint''',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('image_path', default=None, help='path to image for prediction')
parser.add_argument('checkpoint', default=None, help='path to PyTorch checkpoint for loading a model')
parser.add_argument('--top_k', default=1, type=int, help='return top K most likely classes')
parser.add_argument('--category_names', default=None, help='json file mapping of categories to real names')
parser.add_argument('--gpu', action='store_true', help='predict on a GPU')

# Get the args
args = parser.parse_args()

# Load up a model checkpoint
model, checkpoint = functions.load_checkpoint(args.checkpoint, args.gpu)

# Preprocess a image to be used for the prediction in the model
image = functions.process_image(args.image_path, model)

# Make a prediction
functions.predict(args.image_path, model, args.gpu, args.top_k, args.category_names)