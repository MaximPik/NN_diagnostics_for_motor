import numpy as np
import pandas as pd
import json  # Работа с JSON файлами
#import torch
from .lib_nn import *

class matFile:
    def __init__(self, path):
        self.data = {}
        self.path = path
        self._get()

    # add data to the file
    def add(self, dataName, newData):
        # Добавление нового ключа и массива данных/add new key and data array
        self.data[dataName] = newData

    # delete data from the file
    def delete(self, dataName):
        del self.data[dataName]

    # change data in the file
    def change(self, dataName, newData):
        if dataName in self.data:
            self.data[dataName] = newData

    # save data in the file
    def save(self):
        # Сохранение массива в файл .mat
        scipy.io.savemat(self.path, self.data)

    # get data from the file
    def _get(self):
        self.data = scipy.io.loadmat(self.path)

    def printAll(self):
        print(self.data)

    def printKeys(self):
        print(self.data.keys())

    def print(self, dataName):
        print(self.data[dataName])

    def writeToExcel(self, pathToExcel):
        tempData = {}
        for key in self.data.keys():
            if key != '__header__' and key != '__version__' and key != '__globals__':
                tempData[key] = np.array(self.data[key])

        maxLen = 0
        for key in tempData.keys():
            if maxLen < len(tempData[key]):
                maxLen = len(tempData[key])

        for key in tempData.keys():
            tempData[key] = np.append(tempData[key], np.zeros((maxLen - len(tempData[key]), 1)))

        # Создадим DataFrame из очищенного словаря/create a DataFrame from the cleaned dictionary
        df = pd.DataFrame(tempData)
        # Запишем DataFrame в файл Excel/write the DataFrame to an Excel file
        df.to_excel(pathToExcel, index=False)


# Подготовка данных: преобразование в массивы нужной размерности и реализация Standardscaler
def prepare_data(pathFile, intervalOut, columnName = None):
    data = pd.read_excel(pathFile)
    if columnName != None:
        data[columnName] = data[columnName].apply(lambda x: 1 if x == '+' else 0)
    input_sequences = [data.iloc[i:i + intervalOut, :4].values for i in range(0, len(data) - (intervalOut - 1))]
    output_sequences = data.iloc[(intervalOut - 1):, 5].values  # выходные данные для обучения
    # Составляем массив минимальных и максимальных значений
    maxValues = np.max(input_sequences, axis=0)[0]  # axis=0 для каждого столбца
    minValues = np.min(input_sequences, axis=0)[0]
    # Реализуем StandardScaler [-1;1]
    for i, sequence in enumerate(input_sequences):
        for j in range(sequence.shape[1]):  # проходимся по каждому столбцу
            minVal = minValues[j]
            maxVal = maxValues[j]
            # применяем формулу
            input_sequences[i][:, j] = (input_sequences[i][:, j] - minVal) / (maxVal - minVal) * 2 - 1
    return input_sequences, output_sequences

class ConfigJSON():
    def __init__(self, path):
        self.path = path

    def get_dicts(self):
        # Открываем файл для чтения
        with open(self.path, 'r') as json_file:
            # Загружаем словарь из файла
            return json.load(json_file)

def setup():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn = NeuralNetworkLSTM(inputSize=4, hiddenSize=64, numLayers=1, outputSize=1).to(device)
    configFromFile = ConfigJSON('E://AlphaProjects//pythonProgramAI//venv//Include//config.json')
    allDicts = configFromFile.get_dicts()
    nnDict = allDicts['nnData']
    scalarDict = allDicts['scalarData']
    nn.load_state_dict(torch.load(nnDict['nnName']))
    return nn, scalarDict

