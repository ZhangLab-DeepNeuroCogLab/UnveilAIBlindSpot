import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import Main.set_logging as set_logging
logger = set_logging.get_logger(__name__)

class CIFAR100C(Dataset):
    def __init__(self, root, transform=None, corruption='gaussian_noise', level=1):
        super(CIFAR100C, self).__init__()
        self.root = root
        self.transform = transform
        self.corruption = corruption
        self.level = level
        
        self.data = np.load(os.path.join(root, f'{corruption}.npy'))
        self.labels = np.load(os.path.join(root, 'labels.npy'))
        
        self.data = self.data[(level-1)*10000:level*10000]
        self.labels = self.labels[(level-1)*10000:level*10000]
        
        logger.info(f"CIFAR100-C file path: {os.path.join(root, f'{corruption}.npy')} of level {level}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = Image.fromarray(img.astype(np.uint8))
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    
class CIFAR100C_wo_level(Dataset):
    def __init__(self, root, transform=None, corruption='gaussian_noise'):
        super(CIFAR100C_wo_level, self).__init__()
        self.root = root
        self.transform = transform
        self.corruption = corruption
       
        self.data = np.load(os.path.join(root, f'{corruption}.npy'))
        self.labels = np.load(os.path.join(root, 'labels.npy'))
        
        logger.info(f"CIFAR100-C file path: {os.path.join(root, f'{corruption}.npy')} of all levels")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = Image.fromarray(img.astype(np.uint8))
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
def load_cifar100_c(pathroot,corruption_type,corruption_level):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    if corruption_level == 10:
        cifar100c_dataset = CIFAR100C_wo_level(root=pathroot, transform=transform, corruption=corruption_type)
    else:
        cifar100c_dataset = CIFAR100C(root=pathroot, transform=transform, corruption=corruption_type, level=corruption_level)
    
    return cifar100c_dataset



class CIFAR100C_SUPP(Dataset):
    def __init__(self, root, transform=None, corruption='gaussian_noise', std=1):
        super(CIFAR100C_SUPP, self).__init__()
        self.root = root
        self.transform = transform
        self.corruption = corruption
        self.std = std
        
        stds = {0.01:'std_0d01', 0.06:'std_0d06', 0.15:'std_0d15', 0.6:'std_0d6'} 
        
        self.data = np.load(os.path.join(root, f'{corruption}_{stds[std]}.npy'))
        self.labels = np.load("path to the cifar100-c label")
        
        logger.info(f"CIFAR100-C file path: {os.path.join(root, f'{corruption}_{stds[std]}.npy')}.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = Image.fromarray(img.astype(np.uint8))
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    
def load_cifar100_c_supp(pathroot,corruption_type,std):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
   
    cifar100c_dataset = CIFAR100C_SUPP(root=pathroot, transform=transform, corruption=corruption_type, std=std)
    
    return cifar100c_dataset
