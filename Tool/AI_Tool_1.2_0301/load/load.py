"""
Mask R-CNN
Train on the toy load dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python load.py train --dataset=C://Users/jinwo/Mask_RCNN-master/Mask_RCNN-master/samples/load/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 load.py train --dataset=/path/to/load/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 load.py train --dataset=/path/to/load/dataset --weights=imagenet

    # Apply color splash to an image
    python3 load.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 load.py splash --weights=last --video=<URL or path to file>
"""
import subprocess
import os
import sys
import json
import numpy as np
import skimage.draw
import skimage.io
# import argparse

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from PyQt5.QtCore import QRect
import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# session = tf.Session(config=config)

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

############################################################
#  Configurations
############################################################

class loadConfig(Config):
    def __init__(self, config_dict, NUM_CLASSES):
        super().__init__()
        self.num_class = NUM_CLASSES
        self.update(config_dict)

    def update(self, config_dict): # 오버라이딩
        loadConfig.STEPS_PER_EPOCH = int(config_dict['step'])
        loadConfig.GPU_COUNT = int(config_dict['gpu'])
        loadConfig.IMAGES_PER_GPU = int(config_dict['images'])
        loadConfig.BATCH_SIZE = loadConfig.GPU_COUNT * loadConfig.IMAGES_PER_GPU
        loadConfig.BACKBONE = str(config_dict['backbone'])
        loadConfig.LEARNING_RATE = float(config_dict['learning_rate'])
        loadConfig.LAYER = str(config_dict['layers'])
        loadConfig.NUM_CLASSES = int(self.num_class) + 1
        loadConfig.IMAGE_META_SIZE = 12 + loadConfig.NUM_CLASSES
        print('update', loadConfig.IMAGES_PER_GPU, loadConfig.STEPS_PER_EPOCH, loadConfig.LEARNING_RATE, loadConfig.BACKBONE, loadConfig.LAYER)

    # Give the configuration a recognizable name
    NAME = "jinwoo"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    BATCH_SIZE = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + car, Long lane, Short lane

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 25

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.85

    BACKBONE = "resnet50"

    LEARNING_RATE = 0.002

    LAYER = 'heads' # receiveConfig.LAYER

    IMAGE_SHAPE = [1024, 1024, 3]

    IMAGE_META_SIZE = 12 + NUM_CLASSES

############################################################
#  Dataset
############################################################

class loadDataset(utils.Dataset):
    def __init__(self, train_):
        super().__init__()
        self.train_ = train_
        self.image_mask = dict()

    def no_json_file(self):
        print("Program Err")
        os.system("Exit")
        sys.exit()

    def train_val(self, dataset):
        dictionary = {}

        with open('random_choice_json.json', 'w', encoding='utf-8') as outfile:
            json.dump(dictionary, outfile, indent='\t')
        new_annotation = json.load(open('random_choice_json.json'))  # json 파일 생성
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.

        json_directory = self.train_.json_dir
        if os.path.exists(json_directory):
            annotations = json.load(open(os.path.join(json_directory)))  # json 파일 불러오기
        else:
            self.no_json_file()

        annotations = list(annotations.values())  # don't need the dict keys
        for i in range(len(annotations)):
            if annotations[i]['filename'] in dataset:  # dataset(이미지제목 & random으로 뽑은 90%의 dataset) == json 파일의 annotation 이미지제목
                new_annotation[i] = annotations[i]
        new_annotation = list(new_annotation.values()) # random을 선택된 이미지에 대한 새로운 new_annotation json 형태의 list 생성
        new_annotation = [a for a in new_annotation if a['regions']] # JSON 파일에 저장된 데이터 중 Annotation이 1개도 없는 이미지는 제외시킨다.
        return new_annotation

    def load_load(self, dataset_dir, dataset, subset):
        """Load a subset of the load dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        class_label = self.train_.class_label
        # class_list = class_label[0].split(",") # str list 가져와서 , 기준으로 split

        for i in range(len(class_label)):
            self.add_class("jinwoo", i+1, class_label[i])

        # Train or validation dataset?
        assert subset in ["train", "val"]
        new_annotation = self.train_val(dataset)# Train & Val 에 따라서 new_annotation


        # Class Idx 분류를 위하여 dictionary 생성
        class_dict = dict()
        for info in self.class_info:
            class_dict[info["name"]] = info["id"]
        print("이미지 수", len(new_annotation), "class", class_dict)
        # Add images
        for a in new_annotation:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            image_path = os.path.join(dataset_dir, a['filename'])

            if not os.path.exists(image_path): # Json Data에 있는 이미지 파일이 실제 존재하지 않으면 Pass
                continue

            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                class_names = [s['region_attributes'] for s in a['regions'].values()]
                # print(len(class_names))
                num_ids = [class_dict[n['name']] for n in class_names]
                # class_idxs 추가
                #class_names = [r['region_attributes'] for r in a['regions'].values()]
                #class_idxs = [class_dict[r['name']] for r in class_names]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.

            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "jinwoo",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids = num_ids)

    # def load_mask(self, image_id):
    #         image_info = self.image_info[image_id]
    #
    #         if image_info["source"] != "bridge":
    #             return super(self.__class__, self).load_mask(image_id)
    #         info = self.image_info[image_id]
    #
    #         # mask load cost 와 memory cost 중 선택 할 것.
    #         if image_id in self.image_mask:
    #             mask = self.image_mask[image_id]
    #
    #         else:
    #             mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
    #                             dtype=np.uint8)
    #         # skimage lib을 이용한 polygons to a bitmap mask 변환
    #         for i, p in enumerate(info["polygons"]):
    #             # Get indexes of pixels inside the polygon and set them to 1
    #             rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
    #             mask[rr, cc, i] = 1
    #
    #         self.image_mask[image_id] = mask
    #
    #         # class_idxs 추가
    #         class_idxs = np.array(info["class_idxs"])
    #
    #         # Return mask, and array of class IDs of each instance. Since we have
    #         # one class ID only, we return an array of 1s
    #         return mask.astype(np.bool), class_idxs

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a load dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "jinwoo":  # NAME
            return super(self.__class__, self).load_mask(image_id)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if image_id in self.image_mask and self.train_.mask == 'On':
            # class 수가 1개일때 (class + background)
            mask = self.image_mask[image_id]
        else:
            mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                                dtype=np.uint8)
            for i, p in enumerate(info["polygons"]):
                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1

            self.image_mask[image_id] = mask

        num_ids = info['num_ids']
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask.astype(np.bool), num_ids

        # class_idxs = np.array(info["class_idxs"])
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "jinwoo":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model, epoch, parent, show_widget, DEVICE):
    """Train the model."""
    # Training dataset.
    train_ = parent
    format = {'.jpg', '.png', '.bmp'}
    images = []
    image_list = []
    image_list.append(os.listdir(train_.dataset)) # Image의 dir 정보를 받음
    for i in image_list[0]:
        for j in format:
            if j in i:
                images.append(i) # format을 분별하여서 이미지만 골라낸다.
    count = round(len(images)*train_.ratio) # random으로 90% 이미지를 뽑는다.
    train_dataset = np.random.choice(images, count, replace=False)
    dataset_train = loadDataset(train_)
    dataset_train.load_load(train_.dataset, train_dataset, "train")
    dataset_train.prepare()
    print('Train Dataset Loaded')

    if len(images) > count: # validation 이미지가 있을때만 수행 (val비율이 있을때)
        # Validation dataset
        val_perform = True
        val_dataset = set(images) - set(train_dataset) # 교집합이 배제된 val 이미지 set를 사용
        val_dataset = list(val_dataset)

        dataset_val = loadDataset(train_)
        dataset_val.load_load(train_.dataset, val_dataset, "val")
        dataset_val.prepare()
        print('Validation Dataset Loaded')
        show_widget.label_3.setText('Validation Mode On')

        show_widget.listWidget.addItem("Classify Random Validation Dataset to " + ROOT_DIR)
        val_file_classify(val_dataset, train_.dataset)  # Validation file classify
    else:
        val_perform = False # training.py에서 validation 수행 x 넘어감

        dataset_val = dataset_train
        show_widget.listWidget.addItem('No Validation Mode')
        show_widget.label_3.setText('Validation Mode Off')

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network assign")
    show_widget.listWidget.addItem('Now Training Start!')

    with tf.device(str(DEVICE)):
        model.train(dataset_train, dataset_val, show_widget, val_perform,
                    learning_rate=loadConfig.LEARNING_RATE,
                    epochs=epoch,
                    layers=loadConfig.LAYER
                    )# 3+는 Trained model에서 layer 3+층이후부터 Update
    show_widget.pushButton_3.setGeometry(QRect(255, 20, 91, 23))
    show_widget.pushButton_3.setEnabled(False)
    show_widget.pushButton_5.setEnabled(False)
    show_widget.pushButton_2.setEnabled(True)
    show_widget.listWidget.addItem('Model Train is All Done. Initial Set Epoch : ' + str(epoch))
    return

############################################################
#  Validation File Classify
############################################################
def val_file_classify(validation_list, dataset_dir):
    title_ = ""

    for title in validation_list: # validation_list
        title_ += title + "\n"

    # 읽어온 파일 디렉토리 새로 생성하여 저장하기
    try:
        os.makedirs(ROOT_DIR + '/Validation/' + os.path.basename(dataset_dir))
        dir = os.path.join(ROOT_DIR, 'Validation/' + os.path.basename(dataset_dir))
        for i in validation_list:
            image = skimage.io.imread(os.path.join(dataset_dir, i))
            skimage.io.imsave((dir + '/' + i), image)
        with open(dir + '/Validation File.txt', 'w', encoding='utf-8') as outfile:  # val file title .txt로 저장하기
            outfile.write(title_)
    except:
        print("Err about Validation Directory")
        return
    return

############################################################
#  Training
############################################################

class train_step:
    def __init__(self, parent, show_widget, CLASS_COUNT):
        super().__init__()
        # 학습 후 재시작 시 기존에 학습으로 부터 return 해서 메인으로 돌아는 왔지만
        # keras의 backend의 Model session이 여전히 남아있어서 재시작 할때 모델을 새로 생성하는데 이때 오류가 발생함
        # 따라서 백엔드의 session clear함수를 수행하고 train을 위한 model을 create

        self.parent = parent
        self.image_mask = None
        self.show_widget = show_widget

        self.command = "train"
        self.dataset = self.parent.dataset_dir
        self.weights = "coco"
        self.logs = DEFAULT_LOGS_DIR
        # self.class_label = self.parent.class_label
        self.class_label = self.parent.extracted_class
        self.json_dir = self.parent.json_dir
        self.config_dict = self.parent.config

        self.step = self.parent.config['step']
        self.gpu = self.parent.config['gpu']
        self.images = self.parent.config['images']
        self.backbone = self.parent.config['backbone']
        self.learning_rate = self.parent.config['learning_rate']
        self.layers = self.parent.config['layers']
        self.epoch = self.parent.config['epoch']
        self.ratio = self.parent.config['ratio']
        self.mask = self.parent.config['mask']
        DEVICE = self.parent.config['use']
        NUM_CLASSES = CLASS_COUNT
        self.exit = 0

        # Validate arguments
        if self.command == "train":
            assert self.dataset, "Argument --dataset is required for training"

        print("Weights: ", self.weights)
        print("Dataset: ", self.dataset)
        print("Logs: ", self.logs)

        # Configurations
        config = self.configurate(NUM_CLASSES)

        self.show_widget.listWidget.addItem("Weights : " + str(self.weights))
        self.show_widget.listWidget.addItem("Dataset : " + str(self.dataset))
        self.show_widget.listWidget.addItem("Logs : " + str(self.logs))

        # Create model
        # 사용할 device 설정 CPU 모드/ GPU 가속 모드
        with tf.device(str(DEVICE)): # inspection_train_model 소스 참고
            if self.command == "train":
                self.model = modellib.MaskRCNN(mode="training", config=config,
                                          model_dir=self.logs)
            else:
                self.model = modellib.MaskRCNN(mode="inference", config=config,
                                          model_dir=self.logs)

        self.show_widget.listWidget.takeItem(0)
        self.show_widget.listWidget.takeItem(0)
        self.show_widget.listWidget.addItem('Model Create')

        # # Select weights file to load
        weights_path = self.weights_load()

        # # Load weights
        print("Loading weights ", weights_path)
        self.show_widget.listWidget.addItem('Weights Load')

        if self.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            with tf.device(str(DEVICE)):
                self.model.load_weights(weights_path, DEVICE, by_name=True, exclude=[
                    "mrcnn_class_logits", "mrcnn_bbox_fc",
                    "mrcnn_bbox", "mrcnn_mask"])
        else:
            with tf.device(str(DEVICE)):
                self.model.load_weights(weights_path, by_name=True)

        assert int(self.exit) is 0, "Exit Training"  # If assert condition is True, Pass & Continue

        self.show_widget.listWidget.addItem('Loading Dataset(Train/Validation) from JSON info..')
        # # Train or evaluate
        if self.command == "train":
            train(self.model, self.epoch, self, self.show_widget, DEVICE)
        else:
            print("'{}' is not recognized. "
                  "Use 'train' or 'splash'".format(self.command))

        return

    def weights_load(self):
        if self.weights.lower() == "coco":
            weights_path = COCO_WEIGHTS_PATH
            # Download weights file
            if not os.path.exists(weights_path):
                self.show_widget.listWidget.addItem('Downloading CoCo Weight about 250MB')
                utils.download_trained_weights(weights_path)
                self.show_widget.listWidget.addItem('Download Complete')
        elif self.weights.lower() == "last":
            # Find last trained weights
            weights_path = self.model.find_last()
        elif self.weights.lower() == "imagenet":
            # Start from ImageNet trained weights
            weights_path = self.model.get_imagenet_weights()
        else:
            weights_path = self.weights

        return weights_path

    def configurate(self, NUM_CLASSES):
        if self.command == "train":
            config = loadConfig(self.config_dict, NUM_CLASSES)
        else:
            class InferenceConfig(loadConfig): # 오버라이딩
                # Set batch size to 1 since we'll be running inference on
                # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1
            config = InferenceConfig()
        config.display()

        return config






# if __name__ == '__main__':
    # image_mask = None
    # parser = argparse.ArgumentParser(description="Rescale Images")
    #
    # # Parse command line arguments
    # parser = argparse.ArgumentParser(
    #     description='Train Mask R-CNN to detect loads.')
    # parser.add_argument("command",
    #                     metavar="<command>",
    #                     help="'train' or 'splash'")
    # parser.add_argument('--dataset', required=False,
    #                     metavar="/path/to/load/dataset/",
    #                     help='Directory of the load dataset')
    # parser.add_argument('--weights', required=True,
    #                     metavar="/path/to/weights.h5",
    #                     help="Path to weights .h5 file or 'coco'")
    # parser.add_argument('--logs', required=False,
    #                     default=DEFAULT_LOGS_DIR,
    #                     metavar="/path/to/logs/",
    #                     help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--image', required=False,
    #                     metavar="path or URL to image",
    #                     help='Image to apply the color splash effect on')
    # parser.add_argument('--video', required=False,
    #                     metavar="path or URL to video",
    #                     help='Video to apply the color splash effect on')
    #
    # parser.add_argument('-s', '--step', type=int, required=False, nargs=1)
    # parser.add_argument('-g', '--gpu', type=int, required=False, nargs=1)
    # parser.add_argument('-i', '--images', type=int, required=False, nargs=1)
    # parser.add_argument('-b', '--backbone', type=str, required=False, nargs=1)
    # parser.add_argument('-r', '--learningrate', type=float, required=False, nargs=1)
    # parser.add_argument('-l', '--layer', type=str, required=False, nargs=1)
    # parser.add_argument('-o', '--ratio', type=float, required=False, nargs=1)
    #
    # parser.add_argument('-e', '--epoch', type=int, required=False, nargs=1)
    # parser.add_argument('-j', '--json', type=str, required=False, nargs=1)
    # parser.add_argument('-c', '--classes', type=str, required=False, nargs=1)
    #
    # parser.add_argument('-x', '--exit', type=int, required=True, nargs=1)
    #
    # args = parser.parse_args()
    #
    # config_dict = dict()
    # config_dict['step'] = args.step[0]
    # config_dict['gpu'] = args.gpu[0]
    # config_dict['images'] = args.images[0]
    # config_dict['backbone'] = args.backbone[0]
    # config_dict['learning_rate'] = args.learningrate[0]
    # config_dict['layer'] = args.layer[0]
    #
    # # Validate arguments
    # if args.command == "train":
    #     assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "splash":
    #     assert args.image or args.video,\
    #            "Provide --image or --video to apply color splash"
    #
    # print("Weights: ", args.weights)
    # print("Dataset: ", args.dataset)
    # print("Logs: ", args.logs)
    #
    # # Configurations
    # if args.command == "train":
    #     config = loadConfig(config_dict)
    # else:
    #     class InferenceConfig(loadConfig):
    #         # Set batch size to 1 since we'll be running inference on
    #         # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    #         GPU_COUNT = 1
    #         IMAGES_PER_GPU = 1
    #     config = InferenceConfig()
    # config.display()
    #
    # # Create model
    # if args.command == "train":
    #     model = modellib.MaskRCNN(mode="training", config=config,
    #                               model_dir=args.logs)
    # else:
    #     model = modellib.MaskRCNN(mode="inference", config=config,
    #                               model_dir=args.logs)
    #
    # # Select weights file to load
    # if args.weights.lower() == "coco":
    #     weights_path = COCO_WEIGHTS_PATH
    #     # Download weights file
    #     if not os.path.exists(weights_path):
    #         utils.download_trained_weights(weights_path)
    # elif args.weights.lower() == "last":
    #     # Find last trained weights
    #     weights_path = model.find_last()
    # elif args.weights.lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     weights_path = model.get_imagenet_weights()
    # else:
    #     weights_path = args.weights
    #
    # # Load weights
    # print("Loading weights ", weights_path)
    # if args.weights.lower() == "coco":
    #     # Exclude the last layers because they require a matching
    #     # number of classes
    #     model.load_weights(weights_path, by_name=True, exclude=[
    #         "mrcnn_class_logits", "mrcnn_bbox_fc",
    #         "mrcnn_bbox", "mrcnn_mask"])
    # else:
    #     model.load_weights(weights_path, by_name=True)
    #
    # assert int(args.exit[0]) is 0, "Exit Training"  # If assert condition is True, Pass & Continue
    #
    # # Train or evaluate
    # if args.command == "train":
    #     train(model, args.epoch[0])
    # else:
    #     print("'{}' is not recognized. "
    #           "Use 'train' or 'splash'".format(args.command))