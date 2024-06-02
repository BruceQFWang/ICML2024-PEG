
import os
import torch
import numpy as np
import torch.distributed as dist
from timm.data import Mixup

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler

import os.path
import sys
import json
import pandas as pd
import cv2

from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.datasets.utils import download_file_from_google_drive

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from pathlib import Path
import PIL.Image
from typing import Any, Callable, Optional, Tuple, TypeVar, Iterable

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


'''
class Food101(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Food101, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split 

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.list = []

        # tuples (image_path, label) 

        path_json = os.path.join(root, 'split.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        if split == 'train':
            for elem in data['train']:
                self.list.append((elem[0], elem[1]))
        else:
            for elem in data['test']:
                self.list.append((elem[0], elem[1]))

    def __getitem__(self, index):
        # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        dir = self.root + "/images/" + self.list[index][0]

        image = pil_loader(dir)
        label = self.list[index][1]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        # Provide a way to get the length (number of elements) of the dataset
        length = len(self.list)
        return length'''


T = TypeVar("T", str, bytes)
def verify_str_arg(
    value: T,
    arg: Optional[str] = None,
    valid_values: Optional[Iterable[T]] = None,
    custom_msg: Optional[str] = None,
) -> T:
    if not isinstance(value, str):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = "Unknown value '{value}' for argument {arg}. Valid values are {{{valid_values}}}."
            msg = msg.format(value=value, arg=arg, valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)

    return value

class Food101(VisionDataset):
    """`The Food-101 Data Set <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.
    The Food-101 is a challenging data set of 101 food categories with 101,000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images.
    On purpose, the training images were not cleaned, and thus still contain some amount of noise.
    This comes mostly in the form of intense colors and sometimes wrong labels. All images were
    rescaled to have a maximum side length of 512 pixels.
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json") as f:
            metadata = json.loads(f.read())

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)

class miniImagenet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.test_dir = os.path.join(self.root_dir, "test")

        if (self.Train):
            self.train_transform = transforms.Compose([
                transforms.CenterCrop(224)
            ])
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_test()
        
        
        self._make_dataset(Train = self.Train)
        

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".jpg"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    
    def _create_class_idx_dict_test(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.test_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))]
        
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.test_dir):
            for f in files:
                if f.endswith(".jpg"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

        
    def _make_dataset(self, Train=True):

        self.data = []
        self.targets = []
        
            
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.test_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        
        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            
            if not os.path.isdir(dirs):
                continue
            
            for root, _, files in sorted(os.walk(dirs)):
                
                for fname in sorted(files):
                    if (fname.endswith(".jpg")):
                        path = os.path.join(root, fname)
                        
                        if Train:
                            class_index = self.class_to_tgt_idx[tgt]
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            class_index = self.class_to_tgt_idx[tgt]
                            item = (path, self.class_to_tgt_idx[tgt])
                        # add
                        #sample = Image.open(path)
                        #sample = sample.convert('RGB')
                        sample = PIL.Image.open(path).convert("RGB")
                        
                        if Train:
                            sample = self.train_transform(sample)

                        self.data.append(sample)
                        self.targets.append(class_index)
                        
                
    def __len__(self):
        return self.len_dataset

    
    def __getitem__(self, idx):
        sample, tgt = self.data[idx], self.targets[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt
    

# TinyImagenet
class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(val_image_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        
        self.data = []
        self.targets = []
        
            
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            class_index = self.class_to_tgt_idx[tgt]
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            class_index = self.class_to_tgt_idx[self.val_img_to_class[fname]]
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                       
                        # add
                        sample = Image.open(path)
                        sample = sample.convert('RGB')
                        self.data.append(sample)
                        self.targets.append(class_index)
                        

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        sample, tgt = self.data[idx], self.targets[idx]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt
    
'''
class Cub2011(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cub2011, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
'''


class CubDataset(Dataset):
    def __init__(self,
                 root='CUB_200_2011/',
                 test=False,
                 transform=None
                 ):
        self.root = root
        self.transform = transform
        self.x = []
        self.y = []
        self.load_data_list(test)

    def load_data_list(self, test):
        images = open(os.path.join(self.root, 'images.txt')).readlines()
        labels = open(os.path.join(self.root, 'image_class_labels.txt')).readlines()
        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            for line in f.readlines():
                lines = line.strip().split(' ')
                if test and lines[1] == '0':
                    self.x.append(os.path.join(self.root, 'images', images[int(lines[0])-1].strip().split(' ')[1]))
                    self.y.append(int(labels[int(lines[0])-1].strip().split(' ')[1])-1)
                if not test and lines[1] == '1':
                    self.x.append(os.path.join(self.root, 'images', images[int(lines[0])-1].strip().split(' ')[1]))
                    self.y.append(int(labels[int(lines[0])-1].strip().split(' ')[1])-1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        filepath = self.x[item]
        label = self.y[item]

        image = PIL.Image.open(filepath).convert("RGB")

        #image = cv2.imread(filepath)
        #print(image.size)
        #exit()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)

        return image, label

import pathlib
from typing import Any, Callable, Optional, Tuple
from PIL import Image


class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

    def download(self) -> None:
        if self._check_exists():
            return

    def _check_exists(self) -> bool:

        if not (self._base_folder / "devkit").is_dir():
            return False
        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()

def build_loader(config, args):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config, args=args)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config, args=args)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )

    # print(config.DATA.BATCH_SIZE)
    # exit()
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn



def build_dataset(is_train, config, args):
    transform = build_transform(is_train, args)

    if config.DATA.DATASET == 'CIFAR10':
        dataset = datasets.CIFAR10(config.DATA.DATA_PATH, train=is_train, transform=transform, download=True)
        nb_classes = 10
    
    elif config.DATA.DATASET == 'flowers102':
        dataset = datasets.Flowers102(config.DATA.DATA_PATH, split='train' if is_train else 'test', transform=transform, download=True)
        nb_classes = 102

    elif config.DATA.DATASET == 'cars196':
        dataset = StanfordCars(config.DATA.DATA_PATH, split='train' if is_train else 'test', transform=transform)
        nb_classes = 196
    
    elif config.DATA.DATASET == 'MiniIMNET':
        dataset = miniImagenet(root=config.DATA.DATA_PATH, train=is_train, transform=transform)
        nb_classes = 100
    
    elif config.DATA.DATASET == 'TinyIMNET':
        dataset = TinyImageNet(root=config.DATA.DATA_PATH, train=is_train, transform=transform)
        nb_classes = 200
        
    elif config.DATA.DATASET == 'CIFAR100':
        dataset = datasets.CIFAR100(config.DATA.DATA_PATH, train=is_train, transform=transform, download=True)
        nb_classes = 100
    
    elif config.DATA.DATASET == 'IMNET':
        root = os.path.join(config.DATA.DATA_PATH, 'train' if is_train else 'val')
        # print(root)
        # exit()
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    elif config.DATA.DATASET == 'INAT19':
        dataset = INatDataset(config.DATA.DATA_PATH, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    elif config.DATA.DATASET == 'food101':
        dataset = Food101(root=config.DATA.DATA_PATH, split='train' if is_train else 'test', transform=transform)
        nb_classes = 101

    elif config.DATA.DATASET == 'cub200':
        dataset = CubDataset(config.DATA.DATA_PATH, False if is_train else True, transform=transform)   #CubDataset(args.data_path, False if is_train else True)  #Cub2011(args.data_path, train=is_train)
        nb_classes = 200

    return dataset, nb_classes


def build_transform(is_train, args):
    # print(args)
    # exit()
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

# def build_dataset(is_train, config):
#     transform = build_transform(is_train, config)
#     if config.DATA.DATASET == 'imagenet':
#         prefix = 'train' if is_train else 'val'
#         if config.DATA.ZIP_MODE:
#             ann_file = prefix + "_map.txt"
#             prefix = prefix + ".zip@/"
#             dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
#                                         cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
#         else:
#             root = os.path.join(config.DATA.DATA_PATH, prefix)
#             dataset = datasets.ImageFolder(root, transform=transform)
#         nb_classes = 1000
#     elif config.DATA.DATASET == 'imagenet22K':
#         raise NotImplementedError("Imagenet-22K will come soon.")
#     else:
#         raise NotImplementedError("We only support ImageNet Now.")

#     return dataset, nb_classes


# def build_transform(is_train, config):
#     resize_im = config.DATA.IMG_SIZE > 32
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=config.DATA.IMG_SIZE,
#             is_training=True,
#             color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
#             auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
#             re_prob=config.AUG.REPROB,
#             re_mode=config.AUG.REMODE,
#             re_count=config.AUG.RECOUNT,
#             interpolation=config.DATA.INTERPOLATION,
#         )
#         if not resize_im:
#             # replace RandomResizedCropAndInterpolation with
#             # RandomCrop
#             transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
#         return transform

#     t = []
#     if resize_im:
#         if config.TEST.CROP:
#             size = int((256 / 224) * config.DATA.IMG_SIZE)
#             t.append(
#                 transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
#                 # to maintain same ratio w.r.t. 224 images
#             )
#             t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
#         else:
#             t.append(
#                 transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
#                                   interpolation=_pil_interp(config.DATA.INTERPOLATION))
#             )

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
#     return transforms.Compose(t)