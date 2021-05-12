import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib as mpl
import os
import subprocess
import datetime
import argparse
from visualize_utils import make_meshgrid, predict_proba_om_mesh, plot_predictions
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
from generate_data import get_dataset
from network import Network
from sklearn.model_selection import train_test_split
from csv_dataset import CSVDataset
from numpy_dataset import NumpyDataset
mpl.rcParams['animation.ffmpeg_path'] = r'ffmpeg'

class Trainer:
    def __init__(self, num_epochs, learnig_rate, model, criterion, optimizer):
        self.num_epochs = num_epochs
        self.learnig_rate = learnig_rate
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_dataloader):
        frames = []
        fig = plt.figure()
        j=0
        for epoch in range(self.num_epochs):
            # forward
            for i, (features, labels) in enumerate(train_dataloader):
                y_predicted = self.model(features)
                loss = self.criterion(y_predicted, labels)
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (epoch >= 0):

                    train_dataset = train_dataloader.dataset

                    X_train, X_test, y_train, y_test = get_data_from_datasets(train_dataset,
                                                                              train_dataset)

                    xx, yy = make_meshgrid(X_train, X_test, y_train, y_test)
                    Z = predict_proba_om_mesh_tensor(self, xx, yy)

                    plot_title = "frames/frame_{}.png".format(str(j))
                    plot_predictions(xx, yy, Z, X_train=X_train, X_test=X_test,
                                           y_train=y_train, y_test=y_test,
                                           plot_name=plot_title)
                    frames.append(plot_title)
                j+=1
        if os.path.exists('ffmpeg'):
            print("Собираем видеоролик")
            subprocess.call([
                './ffmpeg', '-framerate', '8', '-pattern_type', 'glob', '-i', 'frames/*.png', '-pix_fmt', 'yuv420p',
                f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.mp4'
            ])
            for frame in frames:
                os.remove(frame)
        else:
            print("Отсутствует ffmpeg разместите его в корневой директории, для того чтобы делать видеоролики")

    def get_acc(self, test_dataloader):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0

            for features, labels in test_dataloader:
                y_predicted = self.model(features)
                # max returns (value ,index)
                _, predicted = torch.max(y_predicted.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')

    def predict(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.long)
        self.model.eval()

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)

                _, predicted = torch.max(output_batch.data, 1)
                print(predicted)

                all_outputs = torch.cat((all_outputs, predicted), 0)

        return all_outputs

    def predict_proba(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.long)

        self.model.eval()

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)
                all_outputs = torch.cat((all_outputs, output_batch), 0)

        return all_outputs

    def predict_proba_tensor(self, test_dataloader):
        self.model.eval()

        with torch.no_grad():
            output = self.model(test_dataloader)

        return output


def get_data_from_datasets(train_dataset, test_dataset):
    X_train = train_dataset.X_train.astype(np.float32)
    X_test = test_dataset.X_train.astype(np.float32)

    y_train = train_dataset.y_train.astype(int)
    y_test = test_dataset.y_train.astype(int)

    return X_train, X_test, y_train, y_test


def predict_proba_om_mesh_tensor(clf, xx, yy):
    q = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    Z = clf.predict_proba_tensor(q)[:, 1]
    Z = Z.reshape(xx.shape)
    return Z


def train_network(file_path, params):
    with open(params, 'r') as params_input:
        train_parameters = json.load(params_input)
    if file_path is not None:
        print("Генерируем датасет")
        train_dataset = CSVDataset(file_path)
        test_dataset = CSVDataset(file_path)
    else:
        # Генерируем себе датасет
        X, y = get_dataset(train_parameters['random_state'], train_parameters['num_samples'],
                           train_parameters['generate_type'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
        train_dataset = NumpyDataset(X_train, y_train)
        test_dataset = NumpyDataset(X_test, y_test)
        # train_dataset = Moons(n_samples=5000, shuffle=True, noise=0.1, random_state=0)
        # test_dataset = Moons(n_samples=1000, shuffle=True, noise=0.1, random_state=2)

    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    learnig_rate = train_parameters['learning_rate']
    num_epochs = train_parameters['num_epochs']
    model = Network(2, train_parameters['network_layers'], 3, train_parameters['activation_functions'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learnig_rate)
    trainer = Trainer(num_epochs, learnig_rate, model, criterion, optimizer)
    trainer.train(train_dataloader)
    trainer.get_acc(test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train neural network with your data and visualize it')
    parser.add_argument('--file', default=None, type=str, help='filepath to csv dataset')

    args = parser.parse_args()
    train_network(args.file, 'settings.json')
