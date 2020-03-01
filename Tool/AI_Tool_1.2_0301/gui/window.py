from PyQt5.QtWidgets import QMainWindow, QDesktopWidget
from PyQt5.QtGui import QIcon
from gui.widget import CentralWidget

class ToolWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        print('window')

        # Widget 생성
        self.central_widget = CentralWidget(self)
        self.setCentralWidget(self.central_widget)

        # Action 생성
        self._create_action()

        # Menu 생성
        self._create_menu()

        # Toolbar 생성
        self._create_toolbar()

        # Connect
        self._connect()

        # Status Bar 생성
        self.statusBar().showMessage('MRCNN AI')

        # UI Title 설정
        self.setWindowTitle('Mask R-CNN GUI')

        # Icon 설정
        self.setWindowIcon(QIcon("./icon.png"))

        # UI Size 설정
        self.resize(1120,700)
        self.setMinimumHeight(700)
        self.setMinimumWidth(1120)

        # UI 출력
        self.show()
        self._center()

        #
        self.activateWindow()

    def load_data(self, dataset):
        pass

    def load_json(self):
        pass

    def _create_action(self):
        self.action = {}

    def _create_menu(self):
        # menubar 생성.
        pass

    def _create_toolbar(self):
        # toolbar 생성
        pass

    def _connect(self):
        pass

    def _center(self):
        # 창을 화면 가운데에 위치시킨다.
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
