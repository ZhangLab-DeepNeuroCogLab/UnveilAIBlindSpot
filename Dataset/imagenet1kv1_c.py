import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset
import os
import Main.set_logging as set_logging
logger = set_logging.get_logger(__name__)

class IMAGENET1KV1C(Dataset):
    def __init__(self, root, transform=None, corruption='gaussian_noise', level=1):
        super(IMAGENET1KV1C, self).__init__()
        self.root = root
        self.transform = transform
        self.corruption = corruption
        self.level = level
        
        self.data = torchvision.datasets.ImageFolder(
            root=os.path.join(self.root, self.corruption, str(self.level)),
            transform=self.transform
        )
        
        logger.info(f"IMAGENET1KV1-C file path: {os.path.join(self.root, self.corruption, str(self.level))} of level {level}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
            
        return img, label
    
    
class IMAGENET1KV1C_wo_level(Dataset):
    def __init__(self, root, transform=None, corruption='gaussian_noise'):
        super(IMAGENET1KV1C_wo_level, self).__init__()
        self.root = root
        self.transform = transform
        self.corruption = corruption
        self.datasets = []
        
        for level in range(1, 6):  
            level_path = os.path.join(root, corruption, str(level))
            if os.path.exists(level_path):
                level_dataset = torchvision.datasets.ImageFolder(level_path, transform=transform)
                self.datasets.append(level_dataset)
                logger.info(f"IMAGENET1KV1-C file path: {level_path}")

        self.data = ConcatDataset(self.datasets)
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
            
        return img, label
    
def load_imagenet1kv1_c(pathroot,corruption_type,corruption_level):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    if corruption_level == 10:
        imagenet1kv1c_dataset = IMAGENET1KV1C_wo_level(root=pathroot, transform=transform, corruption=corruption_type)
    else:
        imagenet1kv1c_dataset = IMAGENET1KV1C(root=pathroot, transform=transform, corruption=corruption_type, level=corruption_level)
    return imagenet1kv1c_dataset



class IMAGENET1KV1C_SUPP(Dataset):
    def __init__(self, root, transform=None, corruption='gaussian_noise', std=1):
        super(IMAGENET1KV1C_SUPP, self).__init__()
        self.root = root
        self.transform = transform
        self.corruption = corruption
        self.std = std
        
        stds = {0.01:'std_0d01', 0.06:'std_0d06', 0.15:'std_0d15', 0.6:'std_0d6'} 
        
        self.data = torchvision.datasets.ImageFolder(
            root=os.path.join(root,corruption,stds[std]),
            transform=self.transform
        )
        
        logger.info(f"IMAGENET1KV1-C file path: {os.path.join(root, corruption,stds[std])}.")
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
            
        return img, label
    
    
def load_imagenet1kv1_c_supp(pathroot,corruption_type,std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
   
    imagenet1kv1c_dataset = IMAGENET1KV1C_SUPP(root=pathroot, transform=transform, corruption=corruption_type, std=std)
    
    return imagenet1kv1c_dataset