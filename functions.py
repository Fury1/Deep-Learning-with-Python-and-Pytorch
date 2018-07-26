# Helper functions for train.py and predict.py

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

import json
import torch
import os.path
import numpy as np


# <---train.py start--->
def import_data(data_dir, batch_size, arch):
    """
    Loads a dataset from a directory and performs preproccessing, normalization,
    and training data augmentation.

    The data directory must have sub folders "train", "valid, and "test" containing
    the necessary classes/data. See PyTorch datasets.ImageFolder for more info.

    Args:
        data_dir: Directory path of model data (string)
        batch_size: Training batch size (int)
        arch: PyTorch architecture to use for transfer learning from torchvision.models (string)

    Returns:
        Model ready training, validation, test, and class to index mappings data
        as a tuple of PyTorch Dataloader objects
        IE: (train, val, test, class_to_indexes)
    """

    # PyTorch model normalization and standard deviation values
    # https://pytorch.org/docs/stable/torchvision/models.html
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    # Determine the min image size for diffrent classifier models
    if arch == 'inception_v3': # Inception needs 299px despite documentation listing 224px
        model_size = 299
    else:
        model_size = 224

    # Directory information split
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Perform data augmentation in addition to resizing and normalizing training data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomResizedCrop(model_size),
                                           transforms.ToTensor(), # <-- Needed to convert PIL images to pytorch tensors
                                           transforms.Normalize(mean=mean, std=std)])

    # Just resize and normalize validation and testing data
    test_val_transforms = transforms.Compose([transforms.Resize((model_size, model_size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std)])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(valid_dir, transform=test_val_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_val_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_data, val_data, test_data, train_dataset.class_to_idx

def get_category_names(file):
    """
    Gets category names from a json file.

    Args:
        file: file containing category class:name mappings (json file)

    Returns:
        Python dictionary of class:names
        IE: {'1':'name'}
    """

    # Open the file and return a python dict
    with open(file, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

def build_model(arch, hidden_units, learn_rate):
    """
    Builds a PyTorch transfer learning model as specified.

    Args:
        arch: PyTorch architecture to be used for transfer learning (string)
        hidden_units: number of hidden units for the fully connected layers (int)
        learn_rate: the learning rate to be used with the model for training (float)

    Returns:
        A PyTorch transfer learning model with a new user defined classifer ready for training and
        a equally prepared optimzer as tuple
        IE: (model, optimizer)
    """

    # Download the architecture selected
    model = getattr(models, arch)(pretrained=True)

    # Freeze weights to reuse with new classifier for transfer learning
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Identify the classifier so it can be replaced with the user's specification
    possible_classifier_matches = []
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.modules.linear.Linear) or isinstance(layer, torch.nn.modules.container.Sequential):
            possible_classifier_matches.append((name, layer))

    # Get the last match to get the name of the classifier for replacement
    classifier = possible_classifier_matches[-1]
    c_name = classifier[0]
    c_layer = classifier [1]

    # Get the classifiers input dimensions to so a new model can be built
    # Find the first linear layer if its a sequential module, otherwise just get the linear layer input features
    if isinstance(c_layer, torch.nn.modules.container.Sequential):
        for layer in c_layer:
            try:
                in_features = layer.in_features
                break
            except AttributeError:
                pass
    else:
        in_features = getattr(model, c_name).in_features

    # Define a new classifier using the inputed parameters
    new_classifier = torch.nn.Sequential(OrderedDict([
                          ('fc1', torch.nn.Linear(in_features, hidden_units)),
                          ('relu1', torch.nn.ReLU()),
                          ('dropout1', torch.nn.Dropout(p=0.5)),
                          ('fc2', torch.nn.Linear(hidden_units, 102))
                     ]))

    # Replace the classifier with the user defined
    setattr(model, c_name, new_classifier)

    # Now that the model is prepared set up an optimizer with the new classifier configured
    # Only pass the optimizer the NEW classifier weights
    # IE: "model.NEW_LAYERS.parameters()"
    # model.parameters() returns all trainable pytorch parameters like weights and biases
    optimizer = torch.optim.Adam(getattr(model, c_name).parameters(), lr=learn_rate)

    return model, optimizer

def validation(model, data, criterion, device):
    """
    Computes the accuracy and loss of a model.

    Args:
        model: pytorch model
        data: validation or testing data (Dataloader object)
        criterion: loss function being used
        device: device used for pytorch training ('cuda:0' or 'cpu')

    Returns:
        Model loss and model accuracy as a tuple
        IE: (loss, accuracy)
    """

    # Keep track of the validation loss and accuracy
    val_loss = 0
    val_accuracy = 0

    # Loop through the data in batches
    for images, labels in data:
        # Send the training data to the designated device for computation
        images, labels = images.to(device), labels.to(device)

        # Forward pass, when not in training mode aux_logits is not returned
        outputs = model.forward(images)

        # Get the probabilites from the logits
        probs = torch.nn.functional.softmax(outputs, dim=1) # Get the probabilities for the output logits like the criterion
        predictions = probs.max(dim=1) # Get the max value indexes across the probabilities vectors (batch_size, vector_of_probabilities)

        # Check the accuracy of the predictions against labels
        # Equality is a byte tensor that needs to be converted to a float tensor
        equality = labels.data == predictions[1]
        val_accuracy += equality.type(torch.FloatTensor).mean()

        # Calculate the error
        loss = criterion(outputs, labels)
        val_loss += loss.item() # Get a scaler value from pytorch tensor

    return val_loss, val_accuracy

def train(model, train_data, val_data, optimizer, epochs, gpu):
    """
    Trains a PyTorch model.

    Args:
        model: pytorch model
        train_data: training data (Dataloader object)
        val_data: validation data
        optimizer: pytorch optimizer to be used with the model
        epochs: number of training loops (int)
        gpu: train on gpu (bool)

    Returns:
        A trained version of the model passed in
    """

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()  # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class

    # Train on a gpu or cpu depending on settings
    device = torch.device("cuda:0" if gpu==True else "cpu")
    model.to(device)

    print("Starting training...")

    # Loop over training data
    for epoch in range(epochs - 1):
        # Make sure the model is in training mode
        model.train()
        # Keep track of the loss between training batches
        running_loss = 0

        # Loop through the training data in batches
        for images, labels in train_data:
            # Send the training data to the designated device for computation
            images, labels = images.to(device), labels.to(device)

            # Zero out gradients for the next training pass
            optimizer.zero_grad()

            # Forward pass
            try:
                outputs  = model.forward(images)
            except ValueError:
                # Inception V3 is probably being used, use inceptions logits and not aux_logits,
                # Inception is unique in this regard
                outputs, _ = model.forward(images)

            # Calculate the error
            loss = criterion(outputs, labels)
            running_loss += loss.item() # Get scaler value from pytorch tensor

            # Backprop the error and calculate the gradients for each layer
            loss.backward()

            # Update the weights to adjust for the error based on the gradients
            optimizer.step()

        # Evaluate the model to check progress
        with torch.no_grad(): # Turn off gradients to speed up inference
            # Change model to evaluation mode for inference
            model.eval()
            # Evaluate the model accuracy after adjusting the weights
            val_loss, val_accuracy = validation(model, val_data, criterion, device)

        # Print results per epoch
        print("Epoch: {0}/{1} | Training Error: {2:.2f} | Validation Error: {3:.2f} | Validation Accuracy: {4:.2f}%".format(epoch + 1,
                                                                                                                              epochs,
                                                                                                                              running_loss,
                                                                                                                              val_loss,
                                                                                                                              val_accuracy/len(val_data)*100))

    print("Training complete!")
    return model

def save(model, arch, class_to_idx, cat_to_name, save_dir):
    """
    Saves a trained model and associated information into a checkpoint that can
    be loaded and used later.

    Args:
        model: trained pytorch model
        arch: PyTorch architecture to be used for transfer learning (string)
        class_to_idx: class to index mapping via ImageFolderDataset.class_to_idx
        cat_to_name: class:name dict mappings
        save_dir: directory to save the checkpoint (string)

    Returns:
        None
    """

    print("Saving model")

    # Save the mapping of classes to indices
    model.class_to_idx = class_to_idx
    model.name = arch
    # Create a checkpoint with useful information about the model
    checkpoint = {'transfer_learning_model': model.name,
                  'model': model,
                  'class_to_idx': model.class_to_idx,
                  'classes': cat_to_name,
                  'pytorch_version': '0.4.0'}

    # Save the checkpoint in the project directory
    torch.save(checkpoint, os.path.join(save_dir, model.name + '_checkpoint.pth'))
    print("Model saved!")

# <---train.py end--->

# <---predict.py start--->
def load_checkpoint(filepath, gpu):
    """
    Loads a PyTorch checkpoint on to the desired compute device,
    as long as the compute device is available.
    IE: CUDA enabled GPUs

    Args:
        filepath: path to a PyTorch checkpoint (string)
        gpu: use gpu (bool)

    Returns:
        Pytorch model and the checkpoint dict as a tuple
    """

    # https://pytorch.org/docs/stable/torch.html?highlight=torch%20load#torch.load
    # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage.cuda(0) if gpu and torch.cuda.is_available() else storage)
    model = checkpoint['model']

    return model, checkpoint

def process_image(image_path, model):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Pytorch tensor.

    Args:
        image_path: path to an image
        model: pytorch model to be used

    Returns:
        PyTorch tensor ready to be fed into a model
    """

    # Determine the min image size for diffrent classifier models
    if model.name == 'inception_v3': # Inception needs 299px despite documentation listing 224px
        # Minimum image size required by the network (Inception V3 needs 299px instead of 224px)
        shortest_side = (326, 326)
        min_image_size = 299
    else:
        shortest_side = (256, 256)
        min_image_size = 224


    im = Image.open(image_path)
    im.thumbnail(shortest_side) # Resize while maintaining aspect ratio

    # Find the center of the image and crop based on width and height
    width, height = im.size

    # Find the cartesian coordinates for cropping center
    left = (width - min_image_size)//2
    top = (height - min_image_size)//2
    right = (width - min_image_size)//2 + min_image_size
    bottom = (height - min_image_size)//2 + min_image_size

    # Crop center of the image to (229px x 229px)
    im = im.crop((left, top, right, bottom))

    # Convert to numpy array to normalize
    np_image = np.array(im)
    # print("Original pixel mean: {}".format(np_image.mean()))

    # Scale the image RGB values from (0 - 255) --> (0.0 - 1.0)
    np_image = np_image / 255
    # print("Rescaled pixel mean: {}".format(np_image.mean()))
    # print("-" *40)

    # PyTorch model normalization and standard deviation values
    # https://pytorch.org/docs/stable/torchvision/models.html
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds # Normalize
    # print("Normalized pixel mean: {}".format(np_image.mean()))
    # print("-" *40)

    # Transpose the positions of the array to D, H, W like pytorch tensors
    # print("Old shape: {}".format(np_image.shape))
    np_image = np_image.transpose(2, 0, 1)
    # print("New shape: {}".format(np_image.shape))

    # Convert to pytorch tensor
    torch_tensor_image = torch.from_numpy(np_image)

    # Cast to FloatTensor from DoubleTensor to match weight dtype for predictions
    torch_tensor_image = torch_tensor_image.type(torch.FloatTensor)

    return torch_tensor_image

def predict(image_path, model, gpu, topk, cat_to_name=None):
    """
    Predict the class (or classes) of an image using a trained model.

    Args:
        image_path: path to an image for inference (string)
        model: trained model to be used (PyTorch model)
        gpu: use gpu (bool)
        topk: number of classes/ labels to output (int)
        cat_to_name: file containing class to label mappings (json file)

    Returns:
        None
    """

    # Make sure the model is in evaluation mode, send to process device
    model.eval()
    device = torch.device("cuda:0" if gpu==True else "cpu")
    model.to(device)

    # Preprocess the image for the network, convert to pytorch tensor, send to process device
    image = process_image(image_path, model)
    image.unsqueeze_(0) # Add the "batch_size" at position 0 in the tensor, IE: (1, D, H, W), this is required for single images
    image = image.to(device)

    # Turn off gradients and make a forward pass
    with torch.no_grad():
        outputs = model.forward(image)

    # Get the probabilities with the corresponding indexes
    probs = torch.nn.functional.softmax(outputs, dim=1)
    probs, idxs = probs.topk(topk)

    # Invert the class_to_index dictionary to use the topk indexes to look up dataset class numbers
    # from the ImageFolder class/index mappings
    # IE: {class_number:index_value} --> {index_value:class_number}
    idx_to_class = dict(map(reversed, model.class_to_idx.items()))

    # If topk is greater then 1 we have a list
    if topk > 1:
        # Map the indexes to the correct classes and make a python list
        classes = [idx_to_class[idx] for idx in idxs.squeeze_().tolist()]
        # Convert from pytorch tensor to python list
        probs = probs.squeeze_().tolist()
    # If topk is less then 2 we are no longer working with a list, just a single value
    elif topk < 2:
        classes = idx_to_class[idxs.squeeze_().item()]
        probs = probs.squeeze_().item()

    # Return the real name labels instead of classes
    if cat_to_name is not None and topk > 1:
        names = get_category_names(cat_to_name)
        classes = [names[i] for i in classes]
    elif cat_to_name is not None and topk < 2:
        names = get_category_names(cat_to_name)
        classes = names[classes]

    print("Probs: {} Classes: {}".format(probs, classes))
# <---predict.py end--->