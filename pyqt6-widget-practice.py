# import sys

# from PyQt6.QtCore import Qt
# from PyQt6.QtWidgets import (
#     QApplication,
#     QCheckBox,
#     QComboBox,
#     QDateEdit,
#     QDateTimeEdit,
#     QDial,
#     QDoubleSpinBox,
#     QFontComboBox,
#     QLabel,
#     QLCDNumber,
#     QLineEdit,
#     QMainWindow,
#     QProgressBar,
#     QPushButton,
#     QRadioButton,
#     QSlider,
#     QSpinBox,
#     QTimeEdit,
#     QVBoxLayout,
#     QWidget,
# )

# # Subclass QMainWindow to customize your application's main window
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("Widgets App")

#         layout = QVBoxLayout()
#         widgets = [
#             QCheckBox,
#             QComboBox,
#             QDateEdit,
#             QDateTimeEdit,
#             QDial,
#             QDoubleSpinBox,
#             QFontComboBox,
#             QLCDNumber,
#             QLabel,
#             QLineEdit,
#             QProgressBar,
#             QPushButton,
#             QRadioButton,
#             QSlider,
#             QSpinBox,
#             QTimeEdit,
#         ]

#         for w in widgets:
#             layout.addWidget(w())

#         widget = QWidget()
#         widget.setLayout(layout)

#         # Set the central widget of the Window. Widget will expand
#         # to take up all the space in the window by default.
#         self.setCentralWidget(widget)

# app = QApplication(sys.argv)
# window = MainWindow()
# window.show()
# app.exec()

# import sys

# from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QStackedLayout, QStackedWidget

# from layout_colorwidget import Color


# class MainWindow(QMainWindow):

#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Practice App")

#         layout = QHBoxLayout()

#         layout.addWidget(Color("red"))
#         layout.addWidget(Color("yellow"))
#         layout.addWidget(Color("sky blue"))
#         layout.addWidget(Color("indigo"))

#         widget = QWidget()
#         widget.setLayout(layout)
#         self.setCentralWidget(widget)

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("My App")

#         layout1 = QHBoxLayout()
#         layout2 = QVBoxLayout()
#         layout3 = QVBoxLayout()

#         layout2.addWidget(Color("red"))
#         layout2.addWidget(Color("yellow"))
#         layout2.addWidget(Color("purple"))

#         layout1.setContentsMargins(0,0,0,0)
#         layout1.setSpacing(20)

#         layout1.addLayout(layout2)

#         layout1.addWidget(Color("green"))

#         layout3.addWidget(Color("red"))
#         layout3.addWidget(Color("purple"))

#         layout1.addLayout(layout3)

#         widget = QWidget()
#         widget.setLayout(layout1)
#         self.setCentralWidget(widget)

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("My App")

#         layout = QGridLayout()

#         layout.addWidget(Color("red"), 0, 3)
#         layout.addWidget(Color("green"), 1, 1)
#         layout.addWidget(Color("orange"), 2, 2)
#         layout.addWidget(Color("blue"), 3, 0)

#         widget = QWidget()
#         widget.setLayout(layout)
#         self.setCentralWidget(widget)

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("My App")

#         layout = QStackedLayout()

#         layout.addWidget(Color("red"))
#         layout.addWidget(Color("green"))
#         layout.addWidget(Color("blue"))
#         layout.addWidget(Color("yellow"))

#         layout.setCurrentIndex(3)

#         widget = QWidget()
#         widget.setLayout(layout)
#         self.setCentralWidget(widget)

import sys

from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from layout_colorwidget import Color


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My App")

        pagelayout = QVBoxLayout()
        button_layout = QHBoxLayout()
        self.stacklayout = QStackedLayout()

        pagelayout.addLayout(button_layout)
        pagelayout.addLayout(self.stacklayout)

        btn = QPushButton("red")
        btn.pressed.connect(self.activate_tab_1)
        button_layout.addWidget(btn)
        self.stacklayout.addWidget(Color("red"))

        btn = QPushButton("green")
        btn.pressed.connect(self.activate_tab_2)
        button_layout.addWidget(btn)
        self.stacklayout.addWidget(Color("green"))

        btn = QPushButton("yellow")
        btn.pressed.connect(self.activate_tab_3)
        button_layout.addWidget(btn)
        self.stacklayout.addWidget(Color("yellow"))

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

    def activate_tab_1(self):
        self.stacklayout.setCurrentIndex(0)

    def activate_tab_2(self):
        self.stacklayout.setCurrentIndex(1)

    def activate_tab_3(self):
        self.stacklayout.setCurrentIndex(2)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()

