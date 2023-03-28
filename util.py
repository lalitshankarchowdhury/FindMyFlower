import warnings
from torch.utils import data
from torchvision import datasets, transforms, models
from torch import nn, optim
import os
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np

OUTPUT_FEATURES = 102

# Filter PyTorch warnings
warnings.filterwarnings("ignore")


def load_data(path="flowers/"):
    train_dir = os.path.join(path, "train")
    valid_dir = os.path.join(path, "valid")
    test_dir = os.path.join(path, "test")
    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["test"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["valid"]),
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        "train": data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
        "valid": data.DataLoader(image_datasets["valid"], batch_size=64),
        "test": data.DataLoader(image_datasets["test"], batch_size=64),
    }

    return image_datasets["valid"], dataloaders


def setup_model(
    architecture="vgg16",
    dropout=0.5,
    learning_rate=0.001,
    hidden_units=4096,
    epochs=5,
    gpu=False,
):
    model = getattr(models, architecture)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Get the number of input features for a specific model
    input_features = next(model.classifier.parameters()).size()[1]

    # Create custom classifier
    classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, OUTPUT_FEATURES),
        nn.LogSoftmax(dim=1),
    )

    # Replace the pretrained classifier with custom classifier
    model.classifier = classifier

    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    # If GPU is enabled
    if gpu and torch.cuda.is_available():
        model.cuda()

    return model, criterion, optimizer


def train_network(
    train_loader,
    valid_loader,
    model,
    criterion,
    optimizer,
    epochs=5,
    print_every=10,
    gpu=False,
):
    # Port model weights to appropriate device
    if gpu and torch.cuda.is_available():
        model.cuda()

    # Train the classifier layers
    n_batch = 0
    running_loss = 0
    print_every = 10

    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            n_batch += 1

            # Move inputs and labels to the GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model.features(inputs)
            outputs = outputs.view(inputs.size(0), -1)
            outputs = model.classifier(outputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print training loss and accuracy
            if n_batch % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        # Move inputs and labels to the GPU if available
                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = model.features(inputs)
                        outputs = outputs.view(inputs.size(0), -1)
                        outputs = model.classifier(outputs)
                        batch_loss = criterion(outputs, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Training loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(valid_loader)*100:.3f}%"
                )
                running_loss = 0
                model.train()


def test_accuracy(test_loader, model, criterion, gpu=False):
    test_loss = 0
    accuracy = 0
    model.classifier.eval()

    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass and calculate test loss
            outputs = model.features(inputs)
            outputs = outputs.view(inputs.size(0), -1)
            outputs = model.classifier(outputs)
            batch_loss = criterion(outputs, labels)
            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(
        f"Test loss: {test_loss/len(test_loader):.3f}.. "
        f"Test accuracy: {accuracy/len(test_loader)*100:.3f}%"
    )


def save_checkpoint(
    model,
    optimizer,
    class_to_idx,
    path="checkpoint.pth",
    architecture="vgg16",
    hidden_units=4096,
    dropout=0.5,
    learning_rate=0.001,
    epochs=5,
):
    model.class_to_idx = class_to_idx

    # Get the number of input features for a specific model
    input_features = next(model.classifier.parameters()).size()[1]

    checkpoint = {
        "architecture": architecture,
        "input_size": input_features,
        "output_size": OUTPUT_FEATURES,
        "hidden_units": hidden_units,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "classifier": model.classifier,
        "number_of_epochs": epochs,
        "state_dict": model.state_dict(),
        "class_to_idx": model.class_to_idx,
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path="checkpoint.pth", gpu=False):
    device = "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)

    architecture = checkpoint["architecture"]
    dropout = checkpoint["dropout"]
    learning_rate = checkpoint["learning_rate"]
    hidden_units = checkpoint["hidden_units"]
    epochs = checkpoint["number_of_epochs"]

    model, _, _ = setup_model(
        architecture=architecture,
        dropout=dropout,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        epochs=epochs,
        gpu=gpu,
    )
    model.input_size = checkpoint["input_size"]
    model.output_size = checkpoint["output_size"]
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    model.optimizer = checkpoint["optimizer"]

    return model


def process_image(im):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    # Process a PIL image for use in a PyTorch model

    resize = 256
    crop_size = 224
    width, height = im.size
    shortest_side = min(width, height)

    # calculate the scale factor to resize the image
    scale_factor = resize / shortest_side

    # calculate the new width and height
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # resize the image using the thumbnail method
    im.thumbnail((new_width, new_height))

    # crop image
    left = (new_width - crop_size) / 2
    upper = (new_height - crop_size) / 2
    right = left + crop_size
    lower = upper + crop_size
    im = im.crop((left, upper, right, lower))

    # color channels
    im = np.array(im)
    im = im / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    im = np.transpose(im, (2, 0, 1))

    return im


def predict(image_path, model, topk=5, gpu=False):
    """Predict the class (or classes) of an image using a trained deep learning model."""

    device = "cuda:0" if gpu and torch.cuda.is_available() else "cpu"

    # Port model weights to appropriate device
    if gpu and torch.cuda.is_available():
        model.cuda()

    # Load and pre-process image
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image).to(device).float()
    image = image.unsqueeze(0)

    # Predict the class probabilities
    with torch.no_grad():
        output = model.forward(image)

    probabilities = F.softmax(output, dim=1)

    # Get the top k probabilities and their classes
    top_p, top_classes = probabilities.topk(topk, dim=1)

    # Convert class index to class label
    idx_to_class = {idx: cls for cls, idx in model.class_to_idx.items()}
    top_labels = [idx_to_class[idx.item()] for idx in top_classes[0]]

    return top_p[0].tolist(), top_labels
