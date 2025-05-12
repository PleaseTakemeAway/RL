# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from utils import ClockDataset, TwoHeadResNet
import glob 

cudnn.benchmark = True
plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = r'C:\Users\UNIST\Documents\GitHub\RL\data\images'
image_datasets = {x: ClockDataset(image_paths= glob.glob(os.path.join(data_dir, x, '*.png')),
                                transform = data_transforms[x]) for x in ['train', 'val']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                shuffle=True, num_workers=4)
                for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    import time
    from tempfile import TemporaryDirectory
    import os
    import copy

    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects_hour = 0
                running_corrects_minute = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    hour_labels = labels[:, 0]
                    minute_labels = labels[:, 1]

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        hour_logits, minute_logits = model(inputs)
                        hour_logits = torch.argmax(hour_logits, dim=1)
                        minute_logits = torch.argmax(minute_logits, dim=1)
                        loss_hour = criterion(hour_logits, hour_labels)
                        loss_minute = criterion(minute_logits, minute_labels)
                        loss = loss_hour + loss_minute

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    hour_preds = torch.argmax(hour_logits, dim=1)
                    minute_preds = torch.argmax(minute_logits, dim=1)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects_hour += torch.sum(hour_preds == hour_labels.data)
                    running_corrects_minute += torch.sum(minute_preds == minute_labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc_hour = running_corrects_hour.double() / dataset_sizes[phase]
                epoch_acc_minute = running_corrects_minute.double() / dataset_sizes[phase]

                # 평균 정확도
                epoch_avg_acc = (epoch_acc_hour + epoch_acc_minute) / 2

                print(f'{phase} Loss: {epoch_loss:.4f} '
                      f'Hour Acc: {epoch_acc_hour:.4f} '
                      f'Minute Acc: {epoch_acc_minute:.4f} '
                      f'Avg Acc: {epoch_avg_acc:.4f}')

                if phase == 'val' and epoch_avg_acc > best_acc:
                    best_acc = epoch_avg_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Avg Acc: {best_acc:.4f}')

        model.load_state_dict(torch.load(best_model_params_path))
    return model


# Visualizing few Images 
def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {preds}')
        imshow(img.cpu().data[0])

        model.train(mode=was_training)

# class_names = image_datasets['train'].__getitem__()

# train_image_paths = glob.glob(r"C:\Users\UNIST\Documents\GitHub\RL\data\images\train\*.png")
# train_dataset = ClockDataset(train_image_paths, transform= data_transforms)

# val_image_paths = glob.glob(r"C:\Users\UNIST\Documents\GitHub\RL\data\images\val\*.png")
# val_dataset = ClockDataset(val_image_paths, transform= data_transforms)

# dataloaders = {
#     'train' : DataLoader(train_dataset, batch_size=32, shuffle=True),
#     'val' : DataLoader(val_dataset, batch_size=32)
# }
# We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


if __name__ == '__main__':
    # Get a batch of training data
    inputs, labels = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, "images")

    # Modeling the ConvNet
    model = TwoHeadResNet().to(device)


    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    # Train and Evaluate
    model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, device,
                        num_epochs=25)

    visualize_model(model_ft)

    # ConvNet as Fixed Feature Extractor 
    model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    # Train and Evaluate
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                            exp_lr_scheduler, num_epochs=25)

    visualize_model(model_conv)

    plt.ioff()
    plt.show()

    # Inference on custom Images
    visualize_model_predictions(
        model_conv,
        img_path=r'C:\Users\UNIST\Documents\GitHub\RL\data\images\val\img_80011_10_51_translated.png'
    )

    plt.ioff()
    plt.show()