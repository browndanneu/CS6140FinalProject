#!/usr/bin/env python

import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn

from datetime import datetime

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(3, 5, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(24600, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.cnn_layers(x)
        output = output.view(output.size(0), -1)
        output = self.feed_forward(output)
        return output


def main():
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    grey_flattened_trn = datasets.ImageFolder('train/', transform)
    grey_flattened_tst = datasets.ImageFolder('test/', transform)
    grey_flattened_valid = datasets.ImageFolder('valid/', transform)

    trn_size = 800
    val_tst_size = 100
    trn_batch_size = 100
    num_epochs = 400
    learning_rate = .1

    trn_dataloader = DataLoader(dataset=grey_flattened_trn, batch_size=trn_batch_size)
    tst_dataloader = DataLoader(dataset=grey_flattened_tst, batch_size=len(grey_flattened_tst))
    val_dataloader = DataLoader(dataset=grey_flattened_valid, batch_size=len(grey_flattened_valid))

    model = net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    min_valid_loss = np.inf
    min_valid_epoch = 0  # epoch that gave us min_valid_loss
    min_train_loss = 0
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    final_epoch_changed = 0

    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")
    print('Starting Training :)', start_time)
    # Training/Validation Loop
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        train_count = 0
        model.train()
        for i, (x_trn, y_trn) in enumerate(trn_dataloader):
            outputs = model(x_trn)
            optimizer.zero_grad()
            loss = criterion(outputs, y_trn)
            _, predicted = torch.max(outputs.data, 1)
            train_count += np.count_nonzero(predicted == y_trn)
            loss.backward()
            optimizer.step()
            train_loss += (loss.item() * trn_batch_size)
        train_loss = train_loss / trn_size
        train_losses.append(train_loss)
        train_acc = train_count / trn_size
        train_accs.append(train_acc)

        for i, (x_val, y_val) in enumerate(val_dataloader):
            model.eval()
            outputs = model(x_val)
            loss = criterion(outputs, y_val)
            _, predicted = torch.max(outputs.data, 1)
            valid_acc = np.count_nonzero(predicted == y_val) / val_tst_size
            valid_accs.append(valid_acc)
            valid_loss = loss.item()
            valid_losses.append(valid_loss)

        if epoch % 10 == 0:
            print("Epoch: %d, train_loss: %1.5f, valid_loss: %1.5f, train_acc: %1.5f, valid_acc %1.5f" %
                  (epoch, train_loss, valid_loss, train_acc, valid_acc))

        if min_valid_loss > valid_loss:
            print("%d Validation Loss Decreased(%1.5f--->%1.5f), Saving the model" % (epoch, min_valid_loss, valid_loss))
            min_valid_loss = valid_loss
            min_valid_epoch = epoch - 1  # epochs are 1 indexed
            torch.save(model.state_dict(), 'saved_model_cnn.pth')
            final_epoch_changed = epoch
    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")
    print('Done Training :)', end_time)

    # Load best model
    final_model = net()
    final_model.load_state_dict(torch.load('saved_model_cnn.pth'))
    final_model.eval()

    for i, (x_tst, y_tst) in enumerate(tst_dataloader):
        model.eval()
        test_outputs = model(x_tst)
        _, predicted = torch.max(test_outputs.data, 1)
        test_acc = np.count_nonzero(predicted == y_tst) / val_tst_size
        loss = criterion(test_outputs, y_tst)
        test_loss = loss.item()

    print('Final Losses: \t Train: %1.5f, Valid: %1.5f, Test %1.5f' % (min_train_loss, min_valid_loss, test_loss))
    print('Final Accuracy: \t Train: %1.5f, Valid: %1.5f, Test %1.5f' %
          (train_accs[min_valid_epoch], valid_accs[min_valid_epoch], test_acc))


    plt.clf()
    plt.plot(range(num_epochs), train_losses, label='Train Losses')
    plt.plot(range(num_epochs), valid_losses, label='Validation Losses')
    plt.legend()
    plt.savefig('loss-plot-cnn.png')

    plt.clf()
    plt.plot(range(num_epochs), train_accs, label='Train Accuracies')
    plt.plot(range(num_epochs), valid_accs, label='Validation Accuracies')
    plt.legend()
    plt.savefig('acc-plot-cnn.png')

    print('final epoch model changed', final_epoch_changed)


if __name__ == '__main__':
    main()

