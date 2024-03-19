import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from .lib_common import *

class NeuralNetworkLSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers, outputSize):
        super(NeuralNetworkLSTM, self).__init__()
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.lstm = nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True)
        self.fc_1 = nn.Linear(hiddenSize, outputSize)
        # self.fc_2 = nn.Linear(outputSize, outputSize)
        # self.activation = nn.Sigmoid() # Для Binary Cross Entropy
        self.activation = nn.Softmax() # Для Multi Cross Entropy
        # self.activation = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(x.device)
        c0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.activation(out)
        out = self.fc_1(out[:, -1, :])
        # out = self.activation(out)
        # out = self.fc_2(out)
        return out

    def train_model(self, XData, yData, numEpochs=100):
        XTrain, XTest, yTrain, yTest = train_test_split(XData, yData, test_size=0.2, shuffle=True,
                                                        random_state=42)
        # Преобразуем numpy массивы в тензоры pytorch
        XTrain = torch.tensor(XTrain, dtype=torch.float32)
        yTrain = torch.tensor(yTrain, dtype=torch.float32)
        XTest = torch.tensor(XTest, dtype=torch.float32)
        yTest = torch.tensor(yTest, dtype=torch.float32)

        train_dataset = torch.utils.data.TensorDataset(XTrain, yTrain)
        trainLoader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        #criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        self.train()  # переводим модель в режим обучения
        for epoch in range(numEpochs):
            for inputs, targets in trainLoader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{numEpochs}], Loss: {loss.item():.4f}')

        # Оцениваем нейросеть на тестовых данных
        with torch.no_grad():  # отключаем вычисление градиентов
            outputs = self(XTest)  # получаем выходные данные модели на тестовых данных
            outputs = torch.where(outputs > 0.5, 1, 0)  # преобразуем вероятности в 1 и 0 с помощью порога 0.5
            t_p = 0
            t_n = 0
            f_p = 0
            f_n = 0
            for i in range(0, len(XTest)):
                if outputs[i] == yTest[i]:
                    if (outputs[i] == 0):
                        t_p += 1
                    else:
                        t_n += 1
                else:
                    if (yTest[i] == 0):
                        f_n += 1
                    else:
                        f_p += 1
            print(f't_p: {t_p}, t_n: {t_n}, f_p: {f_p}, f_n: {f_n}')

    def predict_model(self, inputData):
        self.eval()  # переводим модель в режим предсказания
        with torch.no_grad():
            resultData = torch.tensor(inputData, dtype=torch.float32)
            outputs = self(resultData)
            prediction = torch.where(outputs > 0.5, 1, 0)
        return prediction

    def save_model(self, file_path='nn_weights_2.pth'):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path='nn_weights.pth'):
        self.load_state_dict(torch.load(file_path))

# Функция для запуска нейросети и тренировки, если модель изменена
def main():
    pathTrain = 'E:\\AlphaProjects\\simulinkModel\\trainData.xlsx'
    # all_train_data = pd.read_excel(pathTrain) # считываем новый excel файл
    # all_train_data = all_train_data.to_numpy()
    # X = all_train_data[:, :4]
    # X = X.astype(float)
    # y = all_train_data[:, 5]
    # y = y.astype(float)
    XTrain, yTrain = prepare_data(pathTrain, 4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn = NeuralNetworkLSTM(inputSize=4, hiddenSize=64, numLayers=1, outputSize=1).to(device)
    configFromFile = ConfigJSON('E://AlphaProjects//pythonProgramAI//venv//Include//config.json')
    allDicts = configFromFile.get_dicts()
    nnDict = allDicts['nnData']
    scalarDict = allDicts['scalarData']
    nn.load_state_dict(torch.load(nnDict['nnName']))

main()