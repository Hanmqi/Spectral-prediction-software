import sys
import pandas as pd

from PySide6.QtCore import QDateTime, QTimeZone
from PySide6.QtWidgets import QApplication
from my_main_widget import Widget
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow


class MainWindow(QMainWindow):
    def __init__(self, widget):
        QMainWindow.__init__(self)
        self.setWindowTitle("REE prediction model")
        self.setCentralWidget(widget)
        # 菜单栏
        self.menu = self.menuBar()

        ## 添加菜单栏
        data_division_action = QAction("Data Division", self)
        data_division_action.triggered.connect(stacked_widget.show_data_division)
        self.menu.addAction(data_division_action)

        model_training_action = QAction("Model training", self)
        model_training_action.triggered.connect(stacked_widget.show_model_training)
        self.menu.addAction(model_training_action)

        prediction_action = QAction("Prediction", self)
        prediction_action.triggered.connect(stacked_widget.show_prediction)
        self.menu.addAction(prediction_action)

        # 窗口大小
        geometry = self.screen().availableGeometry()
        self.setMinimumSize(int(geometry.width() * 0.5), int(geometry.height() * 0.5))
        # self.setFixedSize(geometry.width() * 0.8, geometry.height() * 0.7)


def transform_date(utc, timezone=None):
    utc_fmt = "yyyy-MM-ddTHH:mm:ss.zzzZ"
    new_date = QDateTime().fromString(utc, utc_fmt)
    if timezone:
        new_date.setTimeZone(timezone)
    return new_date


def read_data(fname):
    # Read the CSV content
    df = pd.read_csv(fname)

    # Remove wrong magnitudes
    df = df.drop(df[df.mag < 0].index)
    magnitudes = df["mag"]

    # My local timezone
    timezone = QTimeZone(b"Europe/Berlin")

    # Get timestamp transformed to our timezone
    times = df["time"].apply(lambda x: transform_date(x, timezone))

    return times, magnitudes


if __name__ == "__main__":

    # Qt Application
    app = QApplication(sys.argv)  # 创建应用程序
    stacked_widget = Widget()  # 使用自定义Widget，包含多个界面
    main_window = MainWindow(stacked_widget)
    main_window.show()

    sys.exit(app.exec())