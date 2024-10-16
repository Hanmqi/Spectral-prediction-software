from PySide6.QtWidgets import (QWidget, QHBoxLayout, QSizePolicy, QFileDialog,
                               QSpacerItem, QLineEdit, QStackedWidget, QTableWidgetItem,)

import pandas as pd
import numpy as np
import read

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QProgressDialog, QMessageBox, QTableWidget
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Dropout, Linear, MSELoss
from torch import optim, tensor, save, load, cuda, no_grad
from torch import device as torch_device
from torchmetrics.functional import r2_score
from scipy.signal import spectrogram
from PySide6.QtWidgets import QVBoxLayout, QLabel, QPushButton
import matplotlib.pyplot as plt


class CNN_2D(Module):
    def __init__(self):
        super(CNN_2D, self).__init__()
        self.conv = Sequential(
            Conv2d(1, 16, (1, 20), padding=2),
            ReLU(),
            Conv2d(16, 16, (3, 50), padding=2),
            ReLU(),
            Conv2d(16, 16, (3, 50), padding=2),
            ReLU(),
            MaxPool2d((2, 8), 2),
            Dropout(),
        )
        self.fc = Sequential(
            Linear(3672*2, 1000),
            ReLU(),
            Linear(1000, 200),
            ReLU(),
            Linear(200, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

def preprocess_data(raw_data_diff, ree_data, element, select_wl=[0, 2500]):
    """
    对原始数据进行预处理，根据指定的数据集类型进行滤波、包络线去除、吸光度、一阶导数或二阶导数处理。

    Args:
    - raw_data_path: str，原始光谱数据文件路径
    - ree_data_path: str，REE 数据文件路径

    Returns:
    - X: pd.DataFrame，处理后的特征数据
    - y: pd.Series，目标数据
    - feature_names: np.array，特征列的名称
    """



    # 绘制频谱图
    X = []
    X_WL = [int(i) for i in raw_data_diff.columns]
    X_WL = [i for i in X_WL if select_wl[0] <= i <= select_wl[1]]

    sampling_freq = 1  # VNIR-SWIR光谱的采样频率 (observations/nm)

    for idx in range(len(raw_data_diff.index)):
        # 光谱
        spectrum = raw_data_diff.iloc[idx, :].values.astype(np.float32)
        spectrum = spectrum[[i for i in range(len(raw_data_diff.columns)) if select_wl[0] <= int(raw_data_diff.columns[i]) <= select_wl[1]]]
        # 生成谱图
        frequencies, times, Sxx = spectrogram(
            spectrum,
            fs=sampling_freq,  # 采样频率
            window='hann',  # 窗口函数
            nperseg=20,  # 每个段的长度
            noverlap=10,  # 重叠的点数
            scaling='spectrum'
        )
        # 取对数
        Sxx_log = np.log(Sxx + 1e-10)
        X.append(Sxx_log)

    # 合并
    X = np.array(X)
    # X = (X - np.min(X)) / (np.max(X) - np.min(X))
    y = ree_data[element]

    return X, y, X_WL


def split_data(X, y):
    y = y.values.astype(np.float32)

    # 使用其他数据进行划分训练集和验证集
    index_sort = np.argsort(y)
    X = X[index_sort]
    y = y[index_sort]
    index = np.arange(len(y))
    index_val = index[::3]
    index_train = np.delete(index, index_val)
    X_train = X[index_train]
    y_train = y[index_train]
    X_val = X[index_val]
    y_val = y[index_val]

    return X_train, y_train, X_val, y_val


class MplCanvas(FigureCanvasQTAgg):
    # 创建一个matplotlib图形，将其嵌入到PyQt5窗口中
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

    def tight_layout(self):
        self.figure.tight_layout()

    def clear_figure(self):
        self.axes.clear()
        self.draw()


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.spectrum = pd.DataFrame()  # 光谱数据
        self.calibration_spectrum = pd.DataFrame()
        self.testing_spectrum = pd.DataFrame()
        self.calibration_ree = pd.DataFrame()
        self.testing_ree = pd.DataFrame()
        self.stack = QStackedWidget(self)  # 创建堆叠小部件

        # 定义三个不同的页面
        self.home_page = self.create_home_page()
        self.data_division_page = self.create_data_division_page()
        self.model_training_page = self.create_model_training_page()
        self.prediction_page = self.create_prediction_page()

        # 将页面添加到堆叠部件中
        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.data_division_page)
        self.stack.addWidget(self.model_training_page)
        self.stack.addWidget(self.prediction_page)

        # 设置默认显示的页面
        layout = QVBoxLayout(self)
        layout.addWidget(self.stack)
        self.setLayout(layout)

    def init_ui(self):
        # 创建布局
        self.layout = QVBoxLayout(self)

        # 加弹簧来调整布局（可选，视需求调整）
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 创建文字
        self.label = QLabel("REE prediction model")
        self.label.setAlignment(Qt.AlignCenter)  # 文字居中对齐
        self.label.setFont(QFont("Times New Roman", 40))  # 设置字号
        self.label.setStyleSheet("color: black;")  # 设置文字颜色为红色
        # 添加控件到布局中
        self.layout.addWidget(self.label)

        # 创建文字
        self.label = QLabel("Copyright belongs to Guangzhou Institute of Geochemistry, Chinese Academy of Sciences")
        self.label.setAlignment(Qt.AlignCenter)  # 文字居中对齐
        self.label.setFont(QFont("Times New Roman", 15))  # 设置字号
        self.label.setStyleSheet("color: black;")  # 设置文字颜色为红色
        # 添加控件到布局中
        self.layout.addWidget(self.label)

        # 创建文字
        self.label = QLabel("Email:2015893776@qq.com")
        self.label.setAlignment(Qt.AlignCenter)  # 文字居中对齐
        self.label.setFont(QFont("Times New Roman", 15))  # 设置字号
        self.label.setStyleSheet("color: black;")  # 设置文字颜色为红色
        # 添加控件到布局中
        self.layout.addWidget(self.label)

        # 加弹簧来调整布局（可选，视需求调整）
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 设置窗口大小
        self.setGeometry(100, 100, 600, 400)
        self.setWindowTitle("REE Prediction Model")

    def create_home_page(self):
        # 创建主界面
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        label = QLabel("REE prediction model")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Times New Roman", 40))
        layout.addWidget(label)

        label = QLabel("Copyright belongs to Guangzhou Institute of Geochemistry, Chinese Academy of Sciences")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Times New Roman", 15))
        layout.addWidget(label)

        label = QLabel("Author:HanMengQi     Email: 2015893776@qq.com")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Times New Roman", 15))
        layout.addWidget(label)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return page

    def create_data_division_page(self):
        # 创建“Data Division”界面
        page = QWidget()
        main_layout = QHBoxLayout(page)  # 主布局
        left_layout = QVBoxLayout()  # 左侧布局
        right_layout = QVBoxLayout()  # 左侧布局
        # 左侧布局
        self.page1_table_widget = QTableWidget()
        label = QLabel("All Spectrum data")
        label.setFont(QFont("Times New Roman", 15))
        left_layout.addWidget(label)
        left_layout.addWidget(self.page1_table_widget)

        # 按钮字体和尺寸样式
        button_font = QFont("Times New Roman", 12)

        # 右侧布局
        # 加一个空白间距
        right_layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Maximum))
        # button
        self.readall_button = QPushButton("Read all spectrum")
        self.readall_button.clicked.connect(self.readall_button_clicked)
        self.readall_button.setFont(button_font)
        right_layout.addWidget(self.readall_button)
        # edit
        mini_layout = QHBoxLayout()
        self.band_label_lineedit = QLineEdit()  # 添加输入框
        self.band_label_lineedit.setPlaceholderText("Please enter the informative band")
        self.band_label_lineedit.setFont(QFont("Times New Roman", 12))
        position_label = QLabel("Informative band")  # 添加标签
        position_label.setFont(QFont("Times New Roman", 12))
        mini_layout.addWidget(position_label)
        mini_layout.addWidget(self.band_label_lineedit)
        right_layout.addLayout(mini_layout)
        # button
        self.division_button = QPushButton("Data Division")
        self.division_button.clicked.connect(self.division_button_clicked)
        self.division_button.setFont(button_font)
        right_layout.addWidget(self.division_button)
        # button
        mini_layout = QHBoxLayout()
        self.save_calibration_spectrum_button = QPushButton("Save calibration spectrum")
        self.save_calibration_spectrum_button.clicked.connect(self.save_calibration_spectrum)
        self.save_calibration_spectrum_button.setFont(button_font)
        mini_layout.addWidget(self.save_calibration_spectrum_button)
        self.save_testing_spectrum_button = QPushButton("Save testing spectrum")
        self.save_testing_spectrum_button.clicked.connect(self.save_testing_spectrum)
        self.save_testing_spectrum_button.setFont(button_font)
        mini_layout.addWidget(self.save_testing_spectrum_button)
        right_layout.addLayout(mini_layout)
        right_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        main_layout.setStretch(0, 3)
        main_layout.setStretch(1, 1)

        return page

    def create_model_training_page(self):
        # 创建“Model Training”界面
        page = QWidget()
        main_layout = QHBoxLayout(page)  # 主布局
        left_layout = QHBoxLayout()  # 左侧布局
        right_layout = QVBoxLayout()  # 左侧布局

        # 左侧布局
        left_layout1 = QVBoxLayout()
        left_layout2 = QVBoxLayout()
        # 1
        self.page2_table_widget1 = QTableWidget()
        label = QLabel("Calibration Spectrum")
        label.setFont(QFont("Times New Roman", 15))
        left_layout1.addWidget(label)
        left_layout1.addWidget(self.page2_table_widget1)
        # 2
        self.page2_table_widget2 = QTableWidget()
        label = QLabel("Calibration REE")
        label.setFont(QFont("Times New Roman", 15))
        left_layout2.addWidget(label)
        left_layout2.addWidget(self.page2_table_widget2)
        left_layout.addLayout(left_layout1)
        left_layout.addLayout(left_layout2)

        # 按钮字体和尺寸样式
        button_font = QFont("Times New Roman", 12)

        # 右侧布局
        # 加一个空白间距
        right_layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Maximum))
        # button
        self.readCalSpec_button = QPushButton("Read calibration spectrum")
        self.readCalSpec_button.clicked.connect(self.readCalSpec_button_clicked)
        self.readCalSpec_button.setFont(button_font)
        right_layout.addWidget(self.readCalSpec_button)
        # button
        self.readCalREE_button = QPushButton("Read calibration REE")
        self.readCalREE_button.clicked.connect(self.readCalREE_button_clicked)
        self.readCalREE_button.setFont(button_font)
        right_layout.addWidget(self.readCalREE_button)
        # edit
        mini_layout = QHBoxLayout()
        self.trainelement_label_lineedit = QLineEdit()  # 添加输入框
        label = QLabel("Train element")  # 添加标签
        self.trainelement_label_lineedit.setPlaceholderText("Please enter the element to be trained")
        self.trainelement_label_lineedit.setFont(QFont("Times New Roman", 12))
        label.setFont(QFont("Times New Roman", 12))
        mini_layout.addWidget(label)
        mini_layout.addWidget(self.trainelement_label_lineedit)
        right_layout.addLayout(mini_layout)
        # button
        self.train_button = QPushButton("Model Training")
        self.train_button.clicked.connect(self.train_button_clicked)
        self.train_button.setFont(button_font)
        right_layout.addWidget(self.train_button)
        right_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        # button
        self.save_model_button = QPushButton("Save model")
        self.save_model_button.clicked.connect(self.save_model_button_clicked)
        self.save_model_button.setFont(button_font)
        right_layout.addWidget(self.save_model_button)
        # button
        self.show_r2_button = QPushButton("Show R2")
        self.show_r2_button.clicked.connect(self.show_r2_button_clicked)
        self.show_r2_button.setFont(button_font)
        right_layout.addWidget(self.show_r2_button)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        main_layout.setStretch(0, 3)
        main_layout.setStretch(1, 1)


        return page

    def create_prediction_page(self):
        # 创建“Prediction”界面
        page = QWidget()
        main_layout = QHBoxLayout(page)  # 主布局
        left_layout = QHBoxLayout()  # 左侧布局
        right_layout = QVBoxLayout()  # 左侧布局

        # 左侧布局
        left_layout1 = QVBoxLayout()
        left_layout2 = QVBoxLayout()
        # 1
        self.page3_table_widget1 = QTableWidget()
        label = QLabel("Testing Spectrum")
        label.setFont(QFont("Times New Roman", 15))
        left_layout1.addWidget(label)
        left_layout1.addWidget(self.page3_table_widget1)
        # 2
        self.page3_table_widget2 = QTableWidget()
        label = QLabel("Model predicted REE")
        label.setFont(QFont("Times New Roman", 15))
        left_layout2.addWidget(label)
        left_layout2.addWidget(self.page3_table_widget2)
        left_layout.addLayout(left_layout1)
        left_layout.addLayout(left_layout2)

        # 按钮字体和尺寸样式
        button_font = QFont("Times New Roman", 12)

        # 右侧布局
        # 加一个空白间距
        right_layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Maximum))
        # button
        self.readTestSpec_button = QPushButton("Read testing spectrum")
        self.readTestSpec_button.clicked.connect(self.readTestSpec_button_clicked)
        self.readTestSpec_button.setFont(button_font)
        right_layout.addWidget(self.readTestSpec_button)
        # button
        self.readmodel_button = QPushButton("Read model")
        self.readmodel_button.clicked.connect(self.readmodel_button_clicked)
        self.readmodel_button.setFont(button_font)
        right_layout.addWidget(self.readmodel_button)
        # button
        self.predict_button = QPushButton("Prediction")
        self.predict_button.clicked.connect(self.predict_button_clicked)
        self.predict_button.setFont(button_font)
        right_layout.addWidget(self.predict_button)
        # button
        self.save_prediction_button = QPushButton("Save prediction")
        self.save_prediction_button.setFont(button_font)
        right_layout.addWidget(self.save_prediction_button)
        self.save_prediction_button.clicked.connect(self.save_prediction_button_clicked)
        right_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        main_layout.setStretch(0, 3)
        main_layout.setStretch(1, 1)

        return page

    def show_data_division(self):
        # 切换到 Data Division 界面
        self.stack.setCurrentWidget(self.data_division_page)

    def show_model_training(self):
        # 切换到 Model Training 界面
        self.stack.setCurrentWidget(self.model_training_page)

    def show_prediction(self):
        # 切换到 Prediction 界面
        self.stack.setCurrentWidget(self.prediction_page)

    def readall_button_clicked(self):
        self.spectrum = pd.DataFrame()
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", ".", "All Files (*);;CSV Files (*.csv)")

        if not file_paths:
            return  # 如果没有选择文件，直接返回

        if len(file_paths) == 1:
            if file_paths[0].endswith('.csv'):
                self.spectrum = read.read_csv(file_paths[0])
        else:
            print('文件格式不一致')
            # 跳出提示弹窗
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("The file format is inconsistent!")
            msg_box.exec()
            return

        if self.spectrum.empty:
            print('没有读取到数据')
            return

        # 使用QTableWidget显示数据
        self.populate_table(self.page1_table_widget, self.spectrum)

    def division_button_clicked(self):
        if self.band_label_lineedit.text():
            info_band = self.band_label_lineedit.text()
            info_band = int(info_band)
            info_band_values = self.spectrum.loc[:, info_band]
            info_band_values = info_band_values.sort_values()

            calibration_index = []
            testing_index = []
            for i in range(0, len(info_band_values), 10):
                testing_index.append(info_band_values.index[i:i + 7].values)
                calibration_index.append(info_band_values.index[i + 7:i + 10].values)
            calibration_index = pd.Index(np.concatenate(calibration_index))
            testing_index = pd.Index(np.concatenate(testing_index))
            self.calibration_spectrum = self.spectrum.loc[calibration_index, :]
            self.testing_spectrum = self.spectrum.loc[testing_index, :]
            if len(self.testing_spectrum) > 0:
                # 弹窗 分割成功
                msg_box = QMessageBox()
                msg_box.setText("Division Successful!")
                msg_box.exec()
                return
        else:
            # 弹窗
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("Informative band is empty!")
            msg_box.exec()
            return

    def save_calibration_spectrum(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save calibration spectrum", ".", "CSV Files (*.csv)")
        if not file_path:
            return

        self.calibration_spectrum.to_csv(file_path)

    def save_testing_spectrum(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save testing spectrum", ".", "CSV Files (*.csv)")
        if not file_path:
            return

        self.testing_spectrum.to_csv(file_path)

    def readCalSpec_button_clicked(self):
        self.calibration_spectrum = pd.DataFrame()
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", ".", "All Files (*);;CSV Files (*.csv)")

        if not file_paths:
            return  # 如果没有选择文件，直接返回

        if len(file_paths) == 1:
            if file_paths[0].endswith('.csv'):
                self.calibration_spectrum = read.read_csv(file_paths[0])
        else:
            print('文件格式不一致')
            # 跳出提示弹窗
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("The file format is inconsistent!")
            msg_box.exec()
            return

        if self.calibration_spectrum.empty:
            print('没有读取到数据')
            return

        self.populate_table(self.page2_table_widget1, self.calibration_spectrum)

    def readCalREE_button_clicked(self):
        self.calibration_ree = pd.DataFrame()
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", ".", "All Files (*);;CSV Files (*.csv)")

        if not file_paths:
            return  # 如果没有选择文件，直接返回

        if len(file_paths) == 1:
            if file_paths[0].endswith('.csv'):
                self.calibration_ree = pd.read_csv(file_paths[0], index_col=0)
                print(self.calibration_ree)
        else:
            print('文件格式不一致')
            # 跳出提示弹窗
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("The file format is inconsistent!")
            msg_box.exec()
            return

        if self.calibration_ree.empty:
            print('没有读取到数据')
            return

        self.populate_table(self.page2_table_widget2, self.calibration_ree)

    def train_button_clicked(self):
        if self.trainelement_label_lineedit.text():
            self.element = self.trainelement_label_lineedit.text()
            self.element = str(self.element)

            if self.calibration_spectrum.empty or self.calibration_ree.empty:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle("Warning")
                msg_box.setText("The data is incomplete!")
                msg_box.exec()
                return

            X, y, X_WL = preprocess_data(self.calibration_spectrum, self.calibration_ree, self.element, select_wl=[0, 2600])
            X_train, y_train, X_val, y_val = split_data(X, y)
            X_train = tensor(X_train).unsqueeze(1)
            y_train = tensor(y_train)
            X_val = tensor(X_val).unsqueeze(1)
            y_val = tensor(y_val)
            device = torch_device('cuda' if cuda.is_available() else 'cpu')
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            X_val = X_val.to(device)
            y_val = y_val.to(device)

            num_epochs = 2000

            # 弹窗进度条
            progress_dialog = QProgressDialog("Model training...", "Cancel", 0, num_epochs-1, self)
            progress_dialog.setWindowTitle("Training Progress")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setValue(0)

            # 设置模型
            net = CNN_2D().to(device)
            loss = MSELoss()  # 损失函数使用交叉熵
            optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 优化器使用Adam, 学习率为0.001

            # 训练
            losses = []
            best_val_R2 = 0
            for epoch in range(num_epochs):
                net.train()
                optimizer.zero_grad()

                y_train_pred = net(X_train)
                train_loss = loss(y_train_pred, y_train.reshape(y_train_pred.shape))
                train_loss.backward()
                optimizer.step()

                net.eval()  # 切换到评估模式
                with no_grad():
                    y_val_pred = net(X_val)
                    train_r2 = r2_score(y_train_pred.squeeze(), y_train)
                    val_r2 = r2_score(y_val_pred.squeeze(), y_val)
                    if val_r2 > best_val_R2:
                        best_val_R2 = val_r2
                        self.best_model = net
                losses.append([train_loss.item(), train_r2.item(), val_r2.item()])

                # 更新进度条
                progress_dialog.setValue(epoch)

                if progress_dialog.wasCanceled():
                    break

            # save loss
            self.losses_df = pd.DataFrame(losses, columns=['train loss', 'train R2', 'val R2'])
        else:
            # 弹窗
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("Train element is empty!")
            msg_box.exec()
            return

    def save_model_button_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save model", ".", "pth Files (*.pth)")
        if not file_path:
            return

        save(self.best_model, file_path)

    def show_r2_button_clicked(self):
        if hasattr(self, 'losses_df'):
            plot_loss = True
            if plot_loss:
                val = self.losses_df.loc[:, 'val R2']
                train = self.losses_df.loc[:, 'train R2']
                plt.plot(val, label='val R2')
                plt.plot(train, label='train R2')
                plt.xlim(0, len(self.losses_df))
                plt.xlabel('Epoch')
                plt.ylabel('R2')
                val_best = self.losses_df['val R2'].max()
                val_best_index = self.losses_df['val R2'].idxmax()
                train_best = self.losses_df['train R2'][val_best_index]
                plt.title(f'Best val R2: {val_best:.4f}, train R2: {train_best:.4f}')
                plt.show()
        else:
            # 弹窗
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("The model has not been trained!")
            msg_box.exec()
            return

    def readTestSpec_button_clicked(self):
        self.testing_spectrum = pd.DataFrame()
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", ".", "All Files (*);;CSV Files (*.csv)")

        if not file_paths:
            return

        if len(file_paths) == 1:
            if file_paths[0].endswith('.csv'):
                self.testing_spectrum = read.read_csv(file_paths[0])
        else:
            print('文件格式不一致')
            # 跳出提示弹窗
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("The file format is inconsistent!")
            msg_box.exec()
            return

        if self.testing_spectrum.empty:
            print('没有读取到数据')
            return

        self.populate_table(self.page3_table_widget1, self.testing_spectrum)

    def readmodel_button_clicked(self):
        self.model = None
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", ".", "pth Files (*.pth)")
        if not file_path:
            return

        self.model = load(file_path)

    def predict_button_clicked(self):
        if self.model is None:
            # 弹窗
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("The model has not been loaded!")
            msg_box.exec()
            return

        if self.testing_spectrum.empty:
            # 弹窗
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText("The testing spectrum is empty!")
            msg_box.exec()
            return

        X = []
        X_WL = [int(i) for i in self.testing_spectrum.columns]
        X_WL = [i for i in X_WL if 0 <= i <= 2600]

        sampling_freq = 1  # VNIR-SWIR光谱的采样频率 (observations/nm)

        for idx in range(len(self.testing_spectrum.index)):
            # 光谱
            spectrum = self.testing_spectrum.iloc[idx, :].values.astype(np.float32)
            spectrum = spectrum[[i for i in range(len(self.testing_spectrum.columns)) if 0 <= int(self.testing_spectrum.columns[i]) <= 2600]]
            # 生成谱图
            frequencies, times, Sxx = spectrogram(
                spectrum,
                fs=sampling_freq,  # 采样频率
                window='hann',  # 窗口函数
                nperseg=20,  # 每个段的长度
                noverlap=10,  # 重叠的点数
                scaling='spectrum'
            )
            # 取对数
            Sxx_log = np.log(Sxx + 1e-10)
            X.append(Sxx_log)

        # 合并
        X = np.array(X)
        X = tensor(X).unsqueeze(1)
        y_pred = self.model(X)
        y_pred = y_pred.cpu().detach().numpy().squeeze()
        self.testing_ree = pd.DataFrame(y_pred, columns=['Predicted concentration'])
        self.testing_ree.index = self.testing_spectrum.index
        self.populate_table(self.page3_table_widget2, self.testing_ree)

    def save_prediction_button_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save prediction", ".", "CSV Files (*.csv)")
        if not file_path:
            return

        self.testing_ree.to_csv(file_path)

    def populate_table(self, table, data):
        # 设置行数和列数
        table.setRowCount(data.shape[0])
        table.setColumnCount(data.shape[1])

        # 设置表头
        table.setHorizontalHeaderLabels([str(col) for col in data.columns])
        table.setVerticalHeaderLabels([str(idx) for idx in data.index])

        # 填充数据
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                item = QTableWidgetItem(str(data.iat[row, col]))
                table.setItem(row, col, item)

        # 自动调整列宽以显示完整内容
        table.resizeColumnsToContents()

        # 调整列宽自适应最后一列
        table.horizontalHeader().setStretchLastSection(True)


