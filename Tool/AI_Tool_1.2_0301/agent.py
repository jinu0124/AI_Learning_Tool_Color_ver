from PyQt5.QtWidgets import QApplication
from gui.window import ToolWindow
import sys
import os

if __name__ == '__main__':
    # def widget():
    ROOT_DIR = os.path.abspath("")
    MODEL_PATH = os.path.join(ROOT_DIR, "mrcnn")
    sys.path.append(MODEL_PATH)

    app = QApplication(sys.argv)
    ex = ToolWindow()
    sys.exit(app.exec_())

    #pip install https://github.com/pyinstaller/pyinstaller/archive/develop.zip
    #pyinstaller -F -n myname.exe widget.py --noconsole

    # Training 은 Json 파일의 이미지 dataset 정보를 기준으로 image를 받는다.
    # 이때, json에 없는 이미지가 포함되어 있으면 해당 이미지는 무시된다.
    # json에 있는 dataset정보의 이미지가 존재하지 않으면 해당 이미지의 json data는 사용되지 않고 train에 참여하지 않는다.

