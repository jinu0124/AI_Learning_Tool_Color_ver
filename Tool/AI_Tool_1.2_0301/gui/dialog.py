from PyQt5.QtWidgets import QDialog
from gui.form import Ui_ConfigOpt, Ui_Val

class ConfigDialog(QDialog, Ui_ConfigOpt):
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.parent = parent

        self.resnet = 'resnet50'
        self.gpu = 1
        self.use = 'GPU'
        self.images = 1
        self.epoch = 20
        self.mask = 'On'
        self.batch_size = 1
        self.learning_rate = 0.001
        self.lineEdit.setText(str(self.learning_rate))
        self.layers = 'heads'
        self.step = 100
        self.ratio = 90

        view_ratio = "    Train : " + str(self.ratio) + "% | Val : " + str((100 - int(self.ratio))) + "%"
        self.listWidget_2.addItem(str(view_ratio))
        self.listWidget.addItem(str(self.batch_size))

        self.init()

        self.spinBox.valueChanged.connect(self.changevalue)
        self.spinBox_3.valueChanged.connect(self.changevalue)
        self.spinBox_4.valueChanged.connect(self.changevalue)
        self.spinBox_5.valueChanged.connect(self.changevalue)
        self.lineEdit.textChanged.connect(self.changevalue)
        self.checkBox.clicked.connect((lambda x: [self.exclusive('resnet50')]))
        self.checkBox_2.clicked.connect((lambda x: [self.exclusive('resnet101')]))
        self.checkBox_3.clicked.connect((lambda x: [self.exclusive('GPU')]))
        self.checkBox_4.clicked.connect((lambda x: [self.exclusive('CPU')]))
        self.pushButton.clicked.connect(self.mask_opt)
        self.horizontalSlider_2.valueChanged.connect(self.changevalue)
        self.comboBox.currentIndexChanged.connect(self.changevalue)

    def init(self):
        self.comboBox.addItem('heads')
        self.comboBox.addItem('4+')
        self.comboBox.addItem('3+(HighEnd)')
        self.comboBox.addItem('2+(HighEnd)')
        self.comboBox.addItem('all(HighEnd)')

        self.checkBox_3.setChecked(True) # GPU
        self.checkBox.setChecked(True) # resnet50

        self.spinBox.setValue(20)
        self.spinBox.setMinimum(1)
        self.spinBox_3.setMinimum(1)
        self.spinBox_4.setMinimum(1)
        self.spinBox_5.setMinimum(5)
        self.spinBox_5.setMaximum(1200)
        self.spinBox_3.setValue(1)
        self.spinBox_4.setValue(1)
        self.spinBox_5.setValue(100)
        self.horizontalSlider_2.setRange(15, 100)  # 범위 (min, max)

    def mask_opt(self):
        if self.pushButton.text() == 'On':
            self.mask = 'Off'
            self.pushButton.setText(self.mask)
        else:
            self.mask = 'On'
            self.pushButton.setText(self.mask)

    def exclusive(self, button):
        if button is 'resnet50':
            self.checkBox.setChecked(True)
            self.checkBox_2.setChecked(False)
        elif button is 'resnet101':
            self.checkBox.setChecked(False)
            self.checkBox_2.setChecked(True)
        elif button is 'GPU':
            self.checkBox_3.setChecked(True)
            self.checkBox_4.setChecked(False)
        else:
            self.checkBox_3.setChecked(False)
            self.checkBox_4.setChecked(True)
        self.changevalue()

    def changevalue(self):
        if self.checkBox.isChecked():
            self.resnet = 'resnet50'
        elif self.checkBox_2.isChecked():
            self.resnet = 'resnet101'

        if self.checkBox_3.isChecked():
            self.use = 'GPU'
            self.spinBox_3.setEnabled(True)
            self.spinBox_4.setEnabled(True)
        else:
            self.use = 'CPU'
            self.spinBox_3.setEnabled(False)
            self.spinBox_4.setEnabled(False)
            self.spinBox_3.setValue(1)
            self.spinBox_4.setValue(1)

        self.epoch = self.spinBox.value()
        self.gpu = self.spinBox_3.value()
        self.images = self.spinBox_4.value()
        self.learning_rate = self.lineEdit.text()
        self.layers = self.comboBox.currentText()
        self.step = self.spinBox_5.value()
        self.ratio = self.horizontalSlider_2.value()
        self.listWidget_2.clear()
        view_ratio = "    Train : "+str(self.ratio)+"% | Val : "+str((100-int(self.ratio)))+"%"
        self.listWidget_2.addItem(str(view_ratio))

        batchS = self.gpu * self.images
        self.batch_size = batchS
        self.listWidget.clear()
        self.listWidget.addItem(str(self.batch_size))

    def accepted_button(self):
        self.accept()

    def rejected_button(self):
        self.reject()

    def showModal(self):
        return super().exec_()

#******************* ********************** ************************ ********************
# Validation Configuration
class ValDialog(QDialog, Ui_Val):
    def __init__(self, parent, class_label=None, h5=180000000):
        super().__init__()
        self.setupUi(self)
        self.parent = parent
        self.h5 = h5
        self.class_label = class_label

        self.init_()

        self.checkBox.clicked.connect(lambda x: [self.value_change(value='CPU')])
        self.checkBox_2.clicked.connect(lambda x: [self.value_change(value='GPU')])
        self.checkBox_4.clicked.connect(lambda x: [self.value_change(value='resnet50')])
        self.checkBox_3.clicked.connect(lambda x: [self.value_change(value='resnet101')])
        self.spinBox.valueChanged.connect(lambda x: [self.value_change(value='spinBox')])
        self.pushButton.clicked.connect(self.splash_func)
        self.pushButton_2.clicked.connect(self.add_class)
        self.pushButton_3.clicked.connect(self.del_class)

    def init_(self):
        self.spinBox.setMinimum(50)
        self.spinBox.setMaximum(99)
        self.spinBox.setValue(90)
        self.checkBox_2.setChecked(True)
        self.pushButton.setText('ON')

        self.val_use = 'GPU'
        if self.h5 > 200000000:
            self.val_backbone = 'resnet101'
            self.checkBox_3.setChecked(True)
        else:
            self.val_backbone = 'resnet50'
            self.checkBox_4.setChecked(True)
        self.splash = 'ON'
        self.detection_rate = 90
        self.val_class = []
        for label in self.class_label:
            self.listWidget.addItem(str(label))
            self.val_class.append(str(label))

    def del_class(self):
        self.listWidget.clear()
        self.val_class = []

    def add_class(self):
        label = self.textEdit.toPlainText()
        if label is not "":
            self.listWidget.addItem(label)
            self.val_class.append(label)
            self.textEdit.clear()

    def splash_func(self):
        if self.pushButton.text() == 'ON':
            self.pushButton.setText('OFF')
            self.splash = 'OFF'
        else:
            self.pushButton.setText('ON')
            self.splash = 'ON'

    def value_change(self, value):
        if value == 'CPU':
            self.val_use = 'CPU'
            self.checkBox_2.setChecked(False)
        elif value == 'GPU':
            self.val_use = 'GPU'
            self.checkBox.setChecked(False)
        elif value == 'resnet50':
            self.val_backbone = 'resnet50'
            self.checkBox_3.setChecked(False)
        elif value == 'resnet101':
            self.val_backbone = 'resnet101'
            self.checkBox_4.setChecked(False)

        if value is 'spinBox':
            self.detection_rate = self.spinBox.value()

    def accepted_button(self):
        self.accept()

    def rejected_button(self):
        self.reject()

    def showModal(self):
        return super().exec_()