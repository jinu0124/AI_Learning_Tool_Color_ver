from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QFileDialog, QApplication, QPushButton, QDialog
from PyQt5.QtCore import QStringListModel, QThread, QRect
from gui.form import *
from gui.dialog import *
import os
import json
from selenium import webdriver
from load.load import train_step
from keras.engine.training import stop
import time
from keras import backend
from gui.validation_inspection import inspection

# Central Main Widget Mask R-CNN GUI

ROOT_DIR = os.path.abspath("")

class CentralWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        print('CentralWidget start')

        self.validation_widget = ValidationWidget(self)
        self.data_widget = DataWidget(self)
        self.show_widget = ShowWidget(self)
        self.class_widget = ClassWidget(self)
        self.train_widget = TrainWidget(self)
        self.json_widget = JsonWidget(self)
        self.config_widget = ConfigWidget(self)
        self.control_widget = ConfigControlWidget(self)


        superbox = QHBoxLayout()
        mainbox = QHBoxLayout()
        formbox = QVBoxLayout()
        upmask = QHBoxLayout()
        midmask = QHBoxLayout()
        rightmask = QVBoxLayout()
        mostrightmask = QVBoxLayout()
        undermask = QHBoxLayout()

        pb1 = QPushButton()
        pb1.setMaximumWidth(2)
        pb1.setMinimumHeight(240)
        pb1.setFlat(True)
        undermask.addWidget(self.show_widget)
        undermask.addWidget(pb1)
        formbox.addLayout(undermask)

        pb2 = QPushButton()
        pb2.setMaximumWidth(2)
        pb2.setMinimumHeight(180)
        pb2.setFlat(True)  # setFlat(True) : 테두리 없애기 + 해당 button Push 무효화
        upmask.addWidget(self.train_widget)
        upmask.addWidget(pb2)
        formbox.addLayout(upmask)

        pb3 = QPushButton()
        pb3.setMaximumWidth(2)
        pb3.setMinimumHeight(150)
        pb3.setFlat(True)  # setFlat(True) : 테두리 없애기 + 해당 button Push 무효화
        midmask.addWidget(self.validation_widget)
        upmask.addWidget(pb3)
        formbox.addLayout(midmask)

        pb4 = QPushButton()
        pb4.setMinimumWidth(280)
        pb4.setMaximumHeight(2)
        pb4.setFlat(True)  # setFlat(True) : 테두리 없애기 + 해당 button Push 무효화
        rightmask.addWidget(self.class_widget)
        rightmask.addWidget(self.json_widget)
        rightmask.addWidget(self.data_widget)
        rightmask.addWidget(pb4)
        mainbox.addLayout(rightmask)

        pb5 = QPushButton()
        pb5.setMinimumWidth(260)
        pb5.setMaximumHeight(2)
        pb5.setFlat(True)  #
        mostrightmask.addWidget(self.config_widget)
        mostrightmask.addWidget(self.control_widget)
        mostrightmask.addWidget(pb5)
        mainbox.addLayout(mostrightmask)

        superbox.addLayout(formbox)
        superbox.addLayout(mainbox)

        self.setLayout(superbox)

# *************** *************** **************** ****************

class ValidationWidget(QWidget, Ui_Validation):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setupUi(self)

        self.init_()

        self.pushButton_3.setEnabled(False)
        self.pushButton_3.clicked.connect(self.validation_func)
        self.pushButton_2.clicked.connect(self.val_config)
        self.pushButton_1.clicked.connect(self.load_weight)
        self.pushButton_0.clicked.connect(self.load_image)

    def init_(self):
        self.flag = [0 for i in range(3)]
        self.val_dict = dict()
        self.h5 = 180000000
        self.class_label = []

        self.listWidget_8.addItem('{0:<20s} {1:>80s}'.format('Image File Ready ', 'NO'))
        self.listWidget_8.addItem('{0:<20s} {1:>80s}'.format('Weight File Ready ', 'NO'))
        self.listWidget_8.addItem('{0:<20s} {1:>80s}'.format('Configuration Ready ', 'NO'))
        self.listWidget_8.addItem('{0:<20s} {1:>84s}'.format('Splash Mode ', 'ON'))
        self.listWidget_8.addItem(' ')
        self.listWidget_8.addItem(' ')

    def val_config(self): # validation을 하기 위한 config
        val_win = ValDialog(self.parent, self.class_label, self.h5)
        a = val_win.showModal()

        self._update(a, val_win)

    def _update(self, a, val_win):
        if a:
            self.val_dict['val_use'] = val_win.val_use
            self.val_dict['val_backbone'] = val_win.val_backbone
            self.val_dict['val_class'] = val_win.val_class
            self.val_dict['detection_rate'] = float(val_win.detection_rate / 100)
            self.val_dict['splash'] = val_win.splash

            self.listWidget_8.takeItem(2)
            self.listWidget_8.insertItem(2, '{0:<20s} {1:>77s}'.format('Configuration Ready ', 'OK'))
            self.listWidget_8.takeItem(4)
            self.listWidget_8.insertItem(4, '{0:<20s} {1:>81} %'.format('Detection Rate ', val_win.detection_rate))
            self.listWidget_8.takeItem(3)
            if self.val_dict['splash'] is 'ON':
                self.listWidget_8.insertItem(3, '{0:<20s} {1:>83s}'.format('Splash Mode ', 'ON'))
            else:
                self.listWidget_8.insertItem(3, '{0:<20s} {1:>83s}'.format('Splash Mode ', 'OFF'))
            self.flag[2] = 1
            self.val_ready()
        else:
            return super()

    def validation_func(self):
        self.listWidget_8.takeItem(5)
        self.listWidget_8.insertItem(5, 'Please Wait... Object Detect Compute a Moment..')
        self.listWidget_8.setCurrentRow(self.listWidget_8.count() - 1)

        self.val_dict['image'] = self.image_dir
        self.val_dict['weight'] = self.weights_dir

        # self.run_val()
        self.thread5 = Worker(self.parent, 5)
        self.thread5.start()  # run() run_val()

    def run_val(self):
        inspection(self, self.val_dict)
        self.listWidget_8.addItem('Finish Prediction')
        self.listWidget_8.addItem('Saved at : ' + str(ROOT_DIR) + '/Prediction')
        self.listWidget_8.setCurrentRow(self.listWidget_8.count() - 1)
        return

    def load_weight(self):
        try:
            options = QFileDialog.Options()  # json file finder 열기
            options |= QFileDialog.ShowDirsOnly
            weights_dir = QFileDialog.getOpenFileName(self,
                                                          "Open Weights File", filter="h5(*.h5)")
        except:
            self.flag[1] = 0
            return

        if len(weights_dir[0]) < 1:
            return super()

        self.weights_dir = weights_dir[0]

        self.listWidget_8.takeItem(1)
        self.listWidget_8.insertItem(1, '{0:<20s} {1:>50s}'.format('Weight File Ready ', str(self.weights_dir)))
        self.flag[1] = 1
        self.val_ready()
        self.hdf5_weight()

    def hdf5_weight(self):
        # h5py.File(self.weights_dir, 'r')  <- .h5파일 읽는 법 / h5.keys()
        self.h5 = os.path.getsize(self.weights_dir)

    def load_image(self):
        file_name = []

        try:
            options = QFileDialog.Options()  # json file finder 열기
            options |= QFileDialog.ShowDirsOnly
            image_dir = QFileDialog.getOpenFileNames(self,
                                                       "Open Image File", filter="(*.jpg; *.bmp; *.png)")
        except:
            self.flag[0] = 0
            return

        if len(image_dir[0]) < 1:
            # 취소 시
            return super()

        self.image_dir = image_dir[0]

        self.listWidget_8.takeItem(0)
        for i in self.image_dir:
            file_name.append(os.path.basename(i))
        self.listWidget_8.insertItem(0, '{0:<20s} {1:>50s}'.format('Image File Ready ', str(file_name)))
        self.flag[0] = 1
        self.val_ready()

    def val_ready(self):
        a = 0
        for i in self.flag:
            a += i
        if a >= 3:
            self.pushButton_3.setEnabled(True)


# *************** *************** **************** ****************

# show_All의 창 구현부
class ShowDialog(QDialog, Ui_Show_All):
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.parent = parent

        self.init_()

    def init_(self):
        textline = []

        line = self.parent.show_widget.listWidget.count()
        for i in range(line):
            textline.append(self.parent.show_widget.listWidget.item(i).text()) # listwidget을 1줄씩 받아서
            self.textBrowser.append(textline[i]) # textBrowser로 한줄씩 append 자동개행됨

    def showModal(self):
        super().exec_()
    
# **************** *************** **************** ***************

class ShowWidget(QWidget, Ui_Show):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setupUi(self)

        self.flag = False
        self.verbose = 1
        self.pushButton.setEnabled(False)
        self.pushButton_3.setEnabled(False) # stop
        self.pushButton_5.setEnabled(False) # cancel

        self.pushButton.clicked.connect(self.view_loss_thread)
        self.pushButton_2.clicked.connect(self.verbose_ctrl)
        self.pushButton_3.clicked.connect(self.stop_signal)
        self.pushButton_4.clicked.connect(self.clear_sheet)
        self.pushButton_5.clicked.connect(self.cancel_stop_action)
        self.pushButton_6.clicked.connect(self.pause_func)

    def pause_func(self):
        self.flag = ~(self.flag)
        try:
            if self.flag:
                self.pushButton_6.setStyleSheet('color:Green')
                self.label_4.setText('Pause')
            else:
                self.pushButton_6.setStyleSheet('color:Black')
                self.label_4.clear()
            if self.pushButton_3.text() == 'Stop at the end of Epoch':
                stop(True, self, pause=1)
            else:
                stop(False, self, pause=1)
        except:
            return

    def view_loss_thread(self):
        self.thread4 = Worker(self.parent, 4)
        self.thread4.start()  # run()

    def view_loss(self):
        show_win = ShowDialog(self.parent)
        show_win.showModal()

    def cancel_stop_action(self):
        cancel_signal = False
        stop(cancel_signal, self)
        self.pushButton_3.setGeometry(QRect(255, 20, 91, 23))
        self.pushButton_3.setText('Stop Training')
        self.pushButton_3.setEnabled(True)
        self.pushButton_5.setEnabled(False)

    def clear_sheet(self):
        self.listWidget.clear()

    def verbose_ctrl(self): # verbose 컨트롤 사전(훈련 전)에 정해서 시작
        if self.verbose is 0:
            self.pushButton_2.setText('Batch Mode')
            self.verbose += 1
        elif self.verbose is 1:
            self.pushButton_2.setText('Epoch Mode')
            self.verbose += 1
        else:
            self.pushButton_2.setText('Silent')
            self.verbose = 0

    def stop_signal(self):
        stop_signal = True
        self.pushButton_3.setGeometry(QRect(155, 20, 190, 23))
        self.pushButton_5.setEnabled(True)  # cancel
        stop(stop_signal, self)
        self.parent.train_widget.pushButton_3.setEnabled(True)


# *************** *************** **************** ****************

class TrainWidget(QWidget, Ui_Train):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setupUi(self)
        self.ready_set = [0 for i in range(3)]
        self.json_data = {}
        self.show_widget = self.parent.show_widget # show widget을 Keras의 generic_utils에서 출력되는 loss값을 프린트 해주기
        # 위해서 training.py까지 불러놓은 loss 값을 show widget에서 출력해주기 위해 self.show_widget을 만들어 traning에 같이 인자로 보냄

        self.listWidget.addItem('{0:<20s} {1:>85s}'.format('Configuration Ready ', 'NO'))
        self.listWidget.addItem('{0:<20s} {1:>83s}'.format('Image Dataset Ready', 'NO'))
        self.listWidget.addItem('{0:<20s} {1:>87s}'.format('JSON File Ready', 'NO'))
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.pushButton_3.clicked.connect(lambda x: [self.train_start('start')])
        self.pushButton_4.clicked.connect(self.show_tensor)
        self.pushButton_5.clicked.connect(self.close_tensor)

    def close_tensor(self):
        if self.driver is not None:
            self.driver.quit()
            self.pushButton_5.setEnabled(False)
            os.system('Exit')

    def show_tensor(self):
        self.thread2 = Worker(self.parent, 2)
        self.thread2.start()  # run()

    def show_tensorboard(self):
        path = ROOT_DIR
        path = os.path.join(path, 'logs')
        get_last = os.listdir(path)[-1]
        self.thread3 = Worker(self.parent, 3)
        self.thread3.start()
        os.system("tensorboard --logdir=./" + str(path) + "/" + str(get_last))

    def start_chrome(self):
        path = ROOT_DIR
        self.driver = webdriver.Chrome(path+'/chrome_driver/chromedriver.exe')
        self.driver.implicitly_wait(2)
        self.pushButton_5.setEnabled(True)
        self.driver.get('localhost:6006')

    def json_exist(self): # data_widget에서 dataset을 load 시에만 호출(+Json 파일이 올라와있지 않을 때)하여 Check
        image_dir = self.parent.data_widget.image_dir
        file_list = os.listdir(image_dir)
        for i in file_list:
            ext = os.path.splitext(i)
            if '.json' in ext[1]:
                self.train_ready(i)
                self.json_dir = os.path.join(image_dir, i)
                with open(self.json_dir, 'r') as LD_json:
                    json_data = json.load(LD_json)
                self.json_data = json_data
                class_list = self.parent.json_widget.extract_class() # class widget과 같이 Update
                for name in class_list:
                    self.parent.class_widget.add(name)
                break
            else:
                self.train_ready('no_json')

    def train_ready(self, ready):
        if ready is 'config':
            self.ready_set[0] = 1
            self.listWidget.takeItem(0)
            self.listWidget.insertItem(0, '{0:<20s} {1:>85s}'.format('Configuration Ready', 'OK'))
        elif ready is 'dataset':
            self.ready_set[1] = 1
            self.listWidget.takeItem(1)
            self.listWidget.insertItem(1, '{0:<20s} {1:>83s}'.format('Image Dataset Ready', 'OK'))
        elif ready is 'no_dataset':
            self.ready_set[1] = 0
            self.listWidget.takeItem(1)
            self.listWidget.insertItem(1, '{0:<20s} {1:>83s}'.format('Image Dataset Ready', 'NO'))
        elif ready is 'json':
            self.ready_set[2] = 1
            self.listWidget.takeItem(2)
            self.listWidget.insertItem(2, '{0:<20s} {1:>87s}'.format('JSON File Ready', 'OK'))
        elif ready is 'no_json':
            self.ready_set[2] = 0
            self.listWidget.takeItem(2)
            self.listWidget.insertItem(2, '{0:<20s} {1:>87s}'.format('JSON File Ready', 'NO JSON Detected'))
        else:
            self.ready_set[2] = 1
            self.listWidget.takeItem(2)
            self.listWidget.insertItem(2, '{0:<20s} {1:>87s}'.format('JSON File Ready', ready + ' Detected'))

        flag = 0
        for i in range(3):
            if self.ready_set[i] is 1:
                flag += 1

        self.pushButton_3.setEnabled(False)
        if flag is 3:
            self.pushButton_3.setEnabled(True)

    def train_start(self, signal):
        print("Your computer's Ideal Thread Number :", QThread.idealThreadCount())
        if signal is 'start':
            self.thread = Worker(self.parent, 1)
            self.thread.start() # run()
        else:
            QThread.terminate()
            self.thread.terminate()
            self.thread.quit()

    def train_stop(self, parent):
        self.parent = parent
        self.train_start('stop')

    def train_command(self):
        backend.clear_session() # keras의 backend session 초기화 -> Created Model 초기화 / 초기화 안하면 session이 남아있어서 에러남
        flag = 0
        self.config = self.parent.config_widget.config_dict
        self.train_py = os.path.join(os.getcwd(), 'load')
        self.dataset_dir = self.parent.data_widget.image_dir
        self.extracted_class = self.parent.json_widget.class_extracted
        self.class_label = ""
        for i in self.extracted_class:
            self.class_label += ","+i

        if len(self.parent.json_widget.json_file_dirs) > 0: # json 파일을 로드했을 때
            self.json_dir = self.parent.json_widget.json_file_dirs[0]
            flag = 1
        else:
            file_list = os.listdir(self.dataset_dir)
            for i in file_list:
                if ".json" in i:
                    self.json_dir = i
                    flag = 1
                    break
            self.json_dir = os.path.join(self.dataset_dir, self.json_dir)
        print(self.dataset_dir, self.json_dir)
        if flag is 1:
            self.show_widget.listWidget.clear() # 재시작 시 화면 clear 용도
            self.show_widget.listWidget.addItem('Training start soon. Please Wait')
            self.show_widget.listWidget.addItem('If you want to show TensorBoard Please Click in a Moment Later(1~2Min)')
            self.pushButton_3.setEnabled(False)
            self.pushButton_4.setEnabled(True)
            self.parent.show_widget.pushButton_2.setEnabled(False)
            self.parent.show_widget.pushButton.setEnabled(True)
            self.load = train_step(self, self.show_widget, self.parent.class_widget.listWidget.count())
            # Train 종료/완료
            self.pushButton_3.setEnabled(True)
            self.pushButton_4.setEnabled(True)

            return

        # 기존 load.py에 명령어를 보내서 argparse를 통해 수행하던 방식에서 load.py의 train_step(Main)클래스를 통해 학습이 진행되도록 수정

        # self.class_label = ""
        # for i in self.extracted_class:
        #     self.class_label += "," + i
        # if flag is 1:
        #     self.pushButton_4.setEnabled(True)
        #     os.system("python " + self.train_py + "/load.py train --dataset=" + dataset_dir + " --weights=coco" +
        #               " -c=" + self.class_label +
        #               " -j=" + str(self.json_dir) +
        #               " -s=" + str(config['step']) +
        #               " -g=" + str(config['gpu']) +
        #               " -i=" + str(config['images']) +
        #               " -b=" + str(config['backbone']) +
        #               " -r=" + str(config['learning_rate']) +
        #               " -l=" + str(config['layers']) +
        #               " -e=" + str(config['epoch']) +
        #               " -o=" + str(config['ratio']) +
        #               " -x=0")

class Worker(QThread):
    def __init__(self, parent, status):
        QThread.__init__(self)
        self.parent = parent
        self._status = status

    def run(self):
        if self._status == 1:
            self.train_thread = QThread.currentThreadId()
            self.parent.train_widget.train_command()
        elif self._status == 2:
            self.parent.train_widget.show_tensorboard()
        elif self._status == 3:
            self.parent.train_widget.start_chrome()
        elif self._status == 4:
            self.parent.show_widget.view_loss()
        elif self._status == 5:
            self.parent.validation_widget.run_val()
        else:
            print('terminate')
            QThread.terminate()

# *************** *************** **************** ****************

class JsonWidget(QWidget, Ui_Json):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setupUi(self)

        self.json_data = {}
        self.json_file_dirs = []
        self.class_extracted = []

        # connect
        self.pushButton_5.clicked.connect(self.load_json)

    def load_json(self):
        # file 불러오기
        options = QFileDialog.Options()  # json file finder 열기
        options |= QFileDialog.ShowDirsOnly

        json_file_dirs = QFileDialog.getOpenFileName(self,
                                                      "Open Json File", filter="json(*.json)")
        json_file_dir = json_file_dirs[0]

        if not os.path.exists(json_file_dir):
            return super() # 존재하지 않는 Path

        # 취소
        if len(json_file_dir) < 1:
            return super()

        loaded_json_file = dict()
        try:
            with open(json_file_dir, 'r') as LD_json:
                loaded_json_file = json.load(LD_json)
        except:
            return CentralWidget(self.parent)

            # # json 파일 병합.
            # json_data.update(loaded_json_file)

        self.json_data = loaded_json_file
        self.json_file_dirs.append(json_file_dir)
        print(self.json_file_dirs)
        self.update()

    def extract_class(self):
        if self.json_data == {} and self.parent.train_widget.json_data == {}:
            return []
        if self.json_data != {}:
            list_data = [self.json_data[k] for k in self.json_data]
            self.parent.train_widget.train_ready('json')
        else:
            list_data = [self.parent.train_widget.json_data[k] for k in self.parent.train_widget.json_data]
        class_list = []
        for data in list_data:
            if 'regions' not in data:
                continue

            regions = data['regions']
            for idx in range(len(regions)):
                region = regions[str(idx)]['region_attributes']
                name = region['name'] if 'name' in region else None
                if name is not None and name not in class_list:
                    class_list.append(name)
        self.class_extracted = class_list
        return class_list

    def update(self):
        model = QStringListModel(self.json_file_dirs)
        self.listView_3.setModel(model)
        self.update_class()

    def update_class(self):
        """ json data 에 포함된 class 를 자동으로 update

        """
        # 기존 class widget 초기화
        self.parent.class_widget.clear()

        # json data 로 부터 class 이름들을 추출
        class_list = self.extract_class()
        self.parent.validation_widget.class_label = class_list

        # class widget 에 추가
        print(class_list)
        if len(class_list) > 0:
            for name in class_list:
                self.parent.class_widget.add(name)


# *************** *************** **************** ****************

class DataWidget(QWidget, Ui_Data):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setupUi(self)

        self.image_dir = None

        # connect
        self.pushButton_4.clicked.connect(self.load_data)

    def load_data(self):
        format = {'.jpg', '.png', '.bmp'}
        # Dir Dialog 로 데이터 불러오기
        options = QFileDialog.Options()  # directory finder 열기
        options |= QFileDialog.ShowDirsOnly
        self.image_dir = (QFileDialog.getExistingDirectory(self))

        # image_dirs = QFileDialog.getOpenFileNames(self, "Open Json File",
        #                                           filter="Images (*.jpg *.png *.bmp);;jpg(*.jpg);;png(*.png);;bmp(*.bmp)")
        # image_dirs = image_dirs[0]
        #
        # for dir in image_dirs:
        #     # load 실패
        #     if not os.path.exists(dir):
        #         return super()
        if self.image_dir == "":
            # 취소 시
            return super()

        self.images = []
        self.image_list = []
        self.image_list.append(os.listdir(self.image_dir))
        for i in self.image_list[0]:
            for j in format:
                if j in i:
                    self.images.append(i)
        # load image
        # self.image_dirs = image_dirs
        self.update(len(self.images))

    def update(self, images):
        model = QStringListModel(self.images)
        self.listView_2.setModel(model)
        if images > 0:
            self.parent.train_widget.train_ready('dataset')
        else:
            self.parent.train_widget.train_ready('no_dataset')
        if self.parent.train_widget.ready_set[2] is not 1: # 이미 Json 파일이 올라와있는지 확인 후 없으면 Auto Detect
            self.parent.train_widget.json_exist()


# *************** *************** **************** ****************

class ConfigWidget(QWidget, Ui_ConfigList):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setupUi(self)
        self.config_dict = dict()
        self.pushButton.clicked.connect(lambda x: [self.configuration(self.parent)])

    def configuration(self, parent):
        print('config')
        self.parent = parent
        config_win = ConfigDialog(self.parent)
        # accept되면 1, reject되면 0
        r = config_win.showModal()

        self.config_dict['step'] = config_win.step
        self.config_dict['use'] = config_win.use
        self.config_dict['batch_size'] = config_win.batch_size
        self.config_dict['gpu'] = config_win.gpu
        self.config_dict['images'] = config_win.images
        self.config_dict['learning_rate'] = config_win.learning_rate
        self.config_dict['backbone'] = config_win.resnet
        self.config_dict['mask'] = config_win.mask
        self.config_dict['layers'] = config_win.layers
        self.config_dict['ratio'] = float(config_win.ratio)/100
        self.config_dict['epoch'] = config_win.epoch

        self._update(r, config_win, self.parent)

    def _update(self, r, config_win, parent):
        if r:
            self.listWidget_5.clear()
            self.listWidget_5.addItem('{0:<20s} {1:>17d}'.format('Epoch :', config_win.epoch))
            self.listWidget_5.addItem('{0:<20s} {1:>16s}'.format('Use :', str(config_win.use)))
            self.listWidget_5.addItem('{0:<20s} {1:>14d}'.format('Batch Size :', config_win.batch_size))
            self.listWidget_5.addItem('{0:<20s} {1:>12d}'.format('GPU count:', config_win.gpu))
            self.listWidget_5.addItem('{0:<20s} {1:>6d}'.format('Images per GPU :', config_win.images))
            self.listWidget_5.addItem('{0:<20s} {1:>10.5f}'.format('Learning Rate :', float(config_win.learning_rate)))
            self.listWidget_5.addItem('{0:<20s} {1:>12s}'.format('Backbone :', str(config_win.resnet)))
            self.listWidget_5.addItem('{0:<20s} {1:>13s}'.format('Fast Mask :', str(config_win.mask)))
            self.listWidget_5.addItem('{0:<20s} {1:>14s}'.format('Layer Opt :', str(config_win.layers)))
            self.listWidget_5.addItem('{0:<20s} {1:>10s}'.format('Steps per Ep :', str(config_win.step)))
            self.listWidget_5.addItem('{0:<20s} {1:>11s}%'.format('Train/Val Ratio :', str(config_win.ratio)))

            parent.control_widget.update_fast(parent, config_win)
            parent.train_widget.train_ready('config')
        else:
            # Cancel
            return super()

# *************** *************** **************** ****************

class ConfigControlWidget(QWidget, Ui_Config):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setupUi(self)

        self.spinBox.setMinimum(1)
        self.spinBox_2.setMinimum(1)
        self.spinBox_3.setMinimum(1)

        self.spinBox.setEnabled(False)
        self.spinBox_2.setEnabled(False)
        self.spinBox_3.setEnabled(False)
        self.lineEdit.setEnabled(False)

        self.spinBox.valueChanged.connect(lambda x: [self._update_list(self.parent)])
        self.spinBox_2.valueChanged.connect(lambda x: [self._update_list(self.parent)])
        self.spinBox_3.valueChanged.connect(lambda x: [self._update_list(self.parent)])
        self.lineEdit.textChanged.connect(lambda x: [self._update_list(self.parent)])

    def update_fast(self, parent, config_win):
        if config_win.use == 'GPU':
            self.spinBox_2.setEnabled(True)
            self.spinBox_3.setEnabled(True)
        else:
            self.spinBox_2.setEnabled(False)
            self.spinBox_3.setEnabled(False)
        self.spinBox.setEnabled(True)
        self.lineEdit.setEnabled(True)

        parent.control_widget.spinBox.setValue(config_win.epoch)
        parent.control_widget.spinBox_2.setValue(config_win.gpu)
        parent.control_widget.spinBox_3.setValue(config_win.images)
        parent.control_widget.lineEdit.setText(str(config_win.learning_rate))

    def _update_list(self, parent):
        # dictionary Update
        epoch = parent.config_widget.config_dict['epoch'] = self.spinBox.value()
        gpu = parent.config_widget.config_dict['gpu'] = self.spinBox_2.value()
        images = parent.config_widget.config_dict['images'] = self.spinBox_3.value()
        batch = parent.config_widget.config_dict['batch_size'] = gpu * images
        learning_rate = parent.config_widget.config_dict['learning_rate'] = self.lineEdit.text()

        # List View update
        parent.config_widget.listWidget_5.takeItem(0)
        parent.config_widget.listWidget_5.insertItem(0, '{0:<20s} {1:>10d}'.format('Epoch :', epoch))
        parent.config_widget.listWidget_5.takeItem(3)
        parent.config_widget.listWidget_5.insertItem(3, '{0:<20s} {1:>10d}'.format('GPU count :', gpu))
        parent.config_widget.listWidget_5.takeItem(4)
        parent.config_widget.listWidget_5.insertItem(4, '{0:<20s} {1:>10d}'.format('Images per GPU :', images))
        parent.config_widget.listWidget_5.takeItem(5)
        parent.config_widget.listWidget_5.insertItem(5, '{0:<20s} {1:>10s}'.format('Learning Rate :', learning_rate))
        parent.config_widget.listWidget_5.takeItem(2)
        parent.config_widget.listWidget_5.insertItem(2, '{0:<20s} {1:>10d}'.format('Batch Size :', batch))

# *************** *************** **************** ****************

class ClassWidget(QWidget, Ui_Class):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setupUi(self)

        # value
        self._item_list = []

        # connect
        # self.pushButton_1.clicked.connect(lambda : self.add())
        # self.pushButton_2.clicked.connect(self.delete)

    def add(self, name = None):
        # text = self.lineEdit_2.text() if name is None else name
        if name == "":
            self.clear()
        self._item_list.append(name)
        if len(self._item_list) > 0:
            self.listWidget.addItem(name)

    def delete(self):
        # remove item
        item = self.listWidget.currentItem()
        if item is None:
            return
        else:
            text = self.listWidget.currentItem().text()
            self.listWidget.takeItem(self.listWidget.row(item))

        # remove text
        if text not in self._item_list:
            return
        else:
            self._item_list.remove(text)

    def clear(self):
        self.listWidget.clear()
