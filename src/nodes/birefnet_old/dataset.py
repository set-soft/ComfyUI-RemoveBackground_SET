import os
# import cv2
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import transforms

from .preproc import preproc
from .config import Config
from .utils import path_to_image
from .labels import class_labels_TR_sorted

Image.MAX_IMAGE_PIXELS = None       # remove DecompressionBombWarning
config = Config()


class MyData(data.Dataset):
    def __init__(self, datasets, image_size, is_train=True):
        self.size_train = image_size
        self.size_test = image_size
        self.keep_size = not config.size
        self.data_size = (config.size, config.size)
        self.is_train = is_train
        self.load_all = config.load_all
        self.device = config.device
        if self.is_train and config.auxiliary_classification:
            self.cls_name2id = {_name: _id for _id, _name in enumerate(class_labels_TR_sorted)}
        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ][self.load_all or self.keep_size:])
        self.transform_label = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
        ][self.load_all or self.keep_size:])
        dataset_root = os.path.join(config.data_root_dir, config.task)
        # datasets can be a list of different datasets for training on combined sets.
        self.image_paths = []
        for dataset in datasets.split('+'):
            image_root = os.path.join(dataset_root, dataset, 'im')
            self.image_paths += [os.path.join(image_root, p) for p in os.listdir(image_root)]
        self.label_paths = []
        for p in self.image_paths:
            for ext in ['.png', '.jpg', '.PNG', '.JPG', '.JPEG']:
                ## 'im' and 'gt' may need modifying
                p_gt = p.replace('/im/', '/gt/').replace('.'+p.split('.')[-1], ext)
                if os.path.exists(p_gt):
                    self.label_paths.append(p_gt)
                    break
        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            self.class_labels_loaded = []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                _image = path_to_image(image_path, size=(config.size, config.size), color_type='rgb')
                _label = path_to_image(label_path, size=(config.size, config.size), color_type='gray')
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)
                self.class_labels_loaded.append(
                    self.cls_name2id[label_path.split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1
                )


    def __getitem__(self, index):

        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
            class_label = self.class_labels_loaded[index] if self.is_train and config.auxiliary_classification else -1
        else:
            image = path_to_image(self.image_paths[index], size=(config.size, config.size), color_type='rgb')
            label = path_to_image(self.label_paths[index], size=(config.size, config.size), color_type='gray')
            class_label = self.cls_name2id[self.label_paths[index].split('/')[-1].split('#')[3]] if self.is_train and config.auxiliary_classification else -1

        # loading image and label
        if self.is_train:
            image, label = preproc(image, label, preproc_methods=config.preproc_methods)
        # else:
        #     if _label.shape[0] > 2048 or _label.shape[1] > 2048:
        #         _image = cv2.resize(_image, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        #         _label = cv2.resize(_label, (2048, 2048), interpolation=cv2.INTER_LINEAR)

        image, label = self.transform_image(image), self.transform_label(label)

        if self.is_train:
            return image, label, class_label
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)
