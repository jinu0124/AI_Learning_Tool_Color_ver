import os
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.draw
import skimage.io
import sys
import numpy as np
# import cv2
# import numpy as np
# np.set_printoptions(threshold=sys.maxsize) #console창에 배열을 생략없이 모두 출력할때

# Import Mask RCNN
ROOT_DIR = os.path.abspath("")
sys.path.append("../")  # To find local version of the library
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.config import Config
from keras import backend
from PIL import Image

class inspection:
    def __init__(self, parent, dict):
        super().__init__()
        self.parent = parent
        self.dict = dict
        self.start()

    def start(self):
        DEVICE = str(self.dict['val_use'])  # /cpu:0 or /gpu:0

        MODE = "inference"

        MODEL_DIR = str(os.path.join(ROOT_DIR, "logs"))

        if len(self.dict['val_class']) is 0:
            self.dict['val_class'] = ['None']

        # Load validation dataset
        image_dir = list()
        for i in self.dict['image']:
            image_dir.append(i)

        class InferenceConfig(Config):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

            NAME = 'jinwoo'

            # Number of classes (including background)
            NUM_CLASSES = 1 + len(self.dict['val_class'])  # Background + circle,hole, black, rock...

            IMAGE_META_SIZE = 12 + NUM_CLASSES

            BACKBONE = str(self.dict['val_backbone'])

            # Skip detections with < confidence
            DETECTION_MIN_CONFIDENCE = self.dict['detection_rate']

        config = InferenceConfig()
        config.display()

        self.parent.listWidget_8.addItem('Create Model...')
        self.parent.listWidget_8.setCurrentRow(self.parent.listWidget_8.count() - 1)
        backend.clear_session() # 모델 만들기 전 session 초기화
        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode=MODE, model_dir=MODEL_DIR,
                                      config=config)

        weights_path = self.dict['weight']

        # Load weights
        print("Loading weights ", weights_path)
        try:
            with tf.device(DEVICE):
                model.load_weights(weights_path, DEVICE, by_name=True)
        except:
            self.parent.listWidget_8.addItem('Class Label\'s num is not Match Error')
            self.parent.listWidget_8.setCurrentRow(self.parent.listWidget_8.count() - 1)
            return

        self.parent.listWidget_8.addItem('Compute Image File')
        self.parent.listWidget_8.setCurrentRow(self.parent.listWidget_8.count() - 1)

        if not os.path.exists(ROOT_DIR + '/Prediction'):
            os.makedirs(ROOT_DIR + '/Prediction/detection')
            os.makedirs(ROOT_DIR + '/Prediction/splash')
        predict_dir = os.path.join(ROOT_DIR, 'Prediction')

        for i in range(len(image_dir)):
            image = skimage.io.imread(image_dir[i])
            if '.png' in image_dir[i]: # png 형식일 때 RGB가 아닌 RGBA depth가 4여서 compute시 에러 발생 -> .jpg로 변환 후 읽어오기
                im = Image.open(image_dir[i])
                img = im.convert('RGB')
                img.save('png_to_jpg.jpg')
                image = skimage.io.imread('png_to_jpg.jpg')

            # Run object detection
            with tf.device(DEVICE):
                results = model.detect([image], verbose=1) # Object를 Detection 함

            # class_label 순서가 학습때 사용된 순서랑 맞아야함 -> json에서 detect 해서 list에 있는 순서대로 넣어주면 됨
            class_names = ['BG']
            for k in self.dict['val_class']:
                class_names.append(k)

            def get_ax(rows=1, cols=1, size=16):
                _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
                return ax

            image_file = os.path.basename(str(image_dir[i]))

            # Display results
            ax = get_ax(1)
            r = results[0]
            visualize.display_instances(image, image_file, predict_dir, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'], ax=ax,
                                        title="Predictions")

            def color_splash(image, mask):
                gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

                # Copy color pixels from the original color image where mask is set
                if mask.shape[-1] > 0: # splash할 Object가 있으면
                    # We're treating all instances as one, so collapse the mask into one layer
                    mask = (np.sum(mask, -1, keepdims=True) >= 1)
                    splash = np.where(mask, image, gray).astype(np.uint8)
                else: # splash 할 Object가 하나도 없을 때 gray scale 이미지 리턴
                    splash = gray.astype(np.uint8)
                return splash

            if self.dict['splash'] == 'ON':
                splash = color_splash(image, r['masks'])
                skimage.io.imsave(os.path.join(predict_dir, 'splash/' + str(image_file) + '_splash.jpg'), splash)
                # display_images([splash], cols=1) # visualize.py 함수

        if DEVICE.lower() is 'gpu':
            curr_session = tf.get_default_session()
            # close current session
            if curr_session is not None:
                curr_session.close()
            # reset graph
        backend.clear_session()