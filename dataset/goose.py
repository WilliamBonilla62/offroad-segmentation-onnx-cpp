import torch
import os
import glob
import csv
import tqdm
import numpy as np
from typing import Dict, List, Iterable
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def __check_labels(img_path: str, lbl_path: str) -> bool:
    '''
    Check if pair of labels and images exist. Filter non-existing pairs.
    '''
    name = os.path.basename(img_path)
    name, ext = name.split('.')
    name = name.split('_')[:-2]
    name = '_'.join(name)

    names = []
    for l in ['color', 'instanceids', 'labelids']:
        # Check if label exists
        lbl_name = name + '_' + l + '.' + ext
        if not os.path.exists(os.path.join(lbl_path, lbl_name)):
            return False, None
        names.append(lbl_name)

    return True, names

def __goose_datadict_folder(img_path: str, lbl_path: str):
    '''
    Create a data Dictionary with image paths
    '''
    subfolders = glob.glob(os.path.join(img_path, '*/'), recursive = False)
    subfolders = [f.split('/')[-2] for f in subfolders]

    valid_imgs = []
    valid_lbls = []
    valid_insta= []
    valid_color= []

    datadict = []

    for s in tqdm.tqdm(subfolders):
        imgs_p = os.path.join(img_path, s)
        lbls_p = os.path.join(lbl_path, s)
        imgs = glob.glob(os.path.join(imgs_p, '*.png'))
        for i in imgs:
            valid, lbl_names = __check_labels(i, lbls_p)
            if not valid:
                continue

            valid_imgs.append(i)
            valid_color.append(os.path.join(lbls_p, lbl_names[0]))
            valid_insta.append(os.path.join(lbls_p, lbl_names[1]))
            valid_lbls.append(os.path.join(lbls_p,  lbl_names[2]))

    for i,m,p,c in zip(valid_imgs, valid_lbls, valid_insta, valid_color):
        datadict.append({
                'img_path': i,
                'semantic_path': m,
                'instance_path':p,
                'color_path': c,
            })   

    return datadict

def goose_create_dataDict(src_path: str, mapping_csv_name: str = 'goose_label_mapping.csv') -> Dict:
    '''
    Parameters:

        src_path            :   path to dataset

    Returns:

        datadict_train      : dict with the dataset train images information

        datadict_val        : dict with the dataset validation images information

        datadict_test       : dict with the dataset test images information
    '''
    if mapping_csv_name is not None:
        mapping_path = os.path.join(src_path, mapping_csv_name)
        mapping = []
        with open(mapping_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                mapping.append(r)
    else:
        mapping = None

    img_path = os.path.join(src_path, 'images')
    lbl_path = os.path.join(src_path, 'labels')

    datadicts = []
    for c in ['test', 'train', 'val']:
        print("### " + c.capitalize() + " Data ###")
        datadicts.append(
            __goose_datadict_folder(
                os.path.join(img_path, c),
                os.path.join(lbl_path, c)
                )
            )

    test,train,val = datadicts

    return test,train,val, mapping

class GOOSE_SemanticDataset(Dataset):
    """
    Example Pytorch Dataset Module for semantic tasks with GOOSE.
    """

    def __init__(self, dataset_dict: List[Dict], crop: bool = True, resize_size: Iterable[int] = None):
        '''
        Parameters:
            dataset_dict  [Iter]    : List of  Dicts with the images information generated by *goose_create_dataDict*

            crop          [Bool]    : Whether to make a square crop of the images or not

            resize_size   [Iter]    : List with the target resize size of the images (After the crop if crop == True)
        '''
        self.dataset_dict   = dataset_dict
        self.transforms     = transforms.Compose([
            transforms.ToTensor(),
            ])
        self.resize_size    = resize_size
        self.crop           = crop

    def preprocess(self, image):
        if image is None:
            return None

        if self.crop:
            # Square-Crop in the center
            s = min([image.width , image.height])
            image = transforms.CenterCrop((s,s)).forward(image)

        if self.resize_size is not None:
            # Resize to given size
            image = image.resize(self.resize_size, resample=Image.NEAREST)

        return image


    def __getitem__(self, i):
        '''
        Parameter:
            i   [int]                   : Index of the image to get

        Returns:
            image_tensor [torch.Tensor] : 3 x H x W Tensor

            label_tensor [torch.Tensor] : H x W Tensor as semantic map
        '''
        image = Image.open(self.dataset_dict[i]['img_path']).convert('RGB')
        label = Image.open(self.dataset_dict[i]['semantic_path']).convert('L')

        image = self.preprocess(image)
        label = self.preprocess(label)

        image_tensor = self.transforms(image)
        label_tensor = torch.from_numpy(np.array(label)).long()

        return image_tensor, label_tensor

    def __len__(self):
        return len(self.dataset_dict)
if __name__ == "__main__":
    """
    Example main function to show how to use GOOSE_SemanticDataset and goose_create_dataDict.
    """

    # Step 1: Define the dataset path
    dataset_root = "dataset/goose-dataset/"  # adjust if needed

    # Step 2: Create the dictionaries
    test_dict, train_dict, val_dict, mapping = goose_create_dataDict(dataset_root)

    print(f"✅ Loaded dataset splits:")
    print(f"  - Train samples: {len(train_dict)}")
    print(f"  - Val samples: {len(val_dict)}")
    print(f"  - Test samples: {len(test_dict)}")
    print()

    # Step 3: Create a dataset instance
    resize_size = (768, 768)
    train_dataset = GOOSE_SemanticDataset(train_dict, crop=False, resize_size=resize_size)

    # Step 4: Load a sample
    print("🔍 Loading a sample from the dataset...")
    image_tensor, label_tensor = train_dataset[0]

    print(f"Image shape: {image_tensor.shape} (C, H, W)")
    print(f"Label shape: {label_tensor.shape} (H, W)")
    print(f"Unique label classes: {torch.unique(label_tensor)}")
    print("✅ Dataset working properly.")
