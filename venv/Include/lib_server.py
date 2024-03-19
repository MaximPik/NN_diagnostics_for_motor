import struct
import subprocess
import socket
import select
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import tkinter as tk

class Server():
    def __init__(self, host, port, classNN,arrMinVal, arrMaxVal, axs, canvas, listObj):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        self.timeout = 10  # таймаут в секундах
        self.allData = None
        client_socket, address = self.server_socket.accept()
        #print(f"Client connected: {address}")
        self.listObj = listObj
        self.listObj.insert(tk.END, f"Client connected: {address}")
        clientThread = threading.Thread(target=self.handle_client, args=(client_socket, classNN, arrMinVal, arrMaxVal, axs, canvas,))
        clientThread.start()
        # Запуск таймера
        self.timer = threading.Timer(self.timeout, self.server_close)
        self.timer.start()

    def handle_client(self, sock, classNN, arrMinVal, arrMaxVal, axs, canvas):
        for ax in axs:
            ax.clear()  # Очистка каждого подграфика при новом подключении
        dataArray = []
        self.allData = []
        while True:
            data = sock.recv(4 * 8)
            if data:
                values = struct.unpack('dddd', data)
                #print(f"Received values: {values}")
                dataArray.append(values)
                if len(dataArray) == 4:
                    # Составляем массив минимальных и максимальных значений
                    XDataArray = np.array(dataArray)
                    maxValues = np.array(arrMaxVal)
                    minValues = np.array(arrMinVal)
                    # Реализуем StandardScaler [-1;1]
                    for j in range(XDataArray.shape[1]):  # проходимся по каждому столбцу
                        minVal = minValues[j]
                        maxVal = maxValues[j]
                        # применяем формулу
                        XDataArray[:, j] = (XDataArray[:, j] - minVal) / (maxVal - minVal) * 2 - 1

                    dataArray = dataArray[1:]
                    print(f"Received values: {XDataArray}")
                    prediction = classNN.predict_model([XDataArray])
                    self.allData.append(values + (prediction.item(),))
                    print(f'Result: {prediction}')
                else:
                    self.allData.append(values + (1,))
                for iter in range(len(self.allData[0])):
                    axs[iter].plot(np.arange(len(self.allData)), [val[iter] for val in self.allData], color='r')
                canvas.draw()
                plt.pause(0.1)
                #time.sleep(0.5)
                # Перезапуск таймера
                if self.timer:
                    self.timer.cancel()
                self.timer = threading.Timer(self.timeout, self.server_close)
                self.timer.start()

    def server_close(self):
        #print("Connection timed out, closing server")
        self.listObj.insert(tk.END, f"Connection timed out, closing server")
        self.server_socket.close()

    def _kill(self, port):
        try:
            # Find the PID using the port
            result = subprocess.check_output(f"netstat -aon | findstr :{port}", shell=True, text=True,
                                             stderr=subprocess.STDOUT)
            lines = result.strip().split('\n')
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 4 and parts[1].endswith(f':{port}'):
                    pid = parts[-1]
                    print(f"Killing process on port {port} with PID: {pid}")
                    # Kill the process using its PID
                    subprocess.check_output(f"taskkill /F /PID {pid}", shell=True, text=True)
                    print(f"Process {pid} has been terminated.")
        except subprocess.CalledProcessError as e:
            print(f"Error finding or killing process on port {port}.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")