import torch
from Dataset.cifar10 import *
from Dataset.cifar10_c import *
from Dataset.cifar100 import *
from Dataset.cifar100_c import *
from Dataset.imagenet1kv1 import *
from Dataset.imagenet1kv1_c import *
from torch.utils.data import ConcatDataset,Dataset,random_split,Subset,DataLoader,TensorDataset
import random
import h5py

from utils import progress_bar
import Main.set_logging as set_logging
logger = set_logging.get_logger(__name__)

cifar10_C_datasets_list = {
                 "cifar10-Normal": None, 
                 "cifar10-C-GaB": "gaussian_blur", 
                 "cifar10-C-Sat": "saturate",
                 "cifar10-C-Spat": "spatter",
                 "cifar10-C-SpN": "speckle_noise",
                 }

cifar10_Adv_datasets_list = {"cifar10-A_PGD": "adversarial_PGD",
                            "cifar10-A_CW": "adversarial_CW",
                            "cifar10-A_Jitter": "adversarial_Jitter",
                            "cifar10-A_PIFGSM": "adversarial_PIFGSM",                 
}

cifar100_C_datasets_list = {
                "cifar100-Normal": None, 
                "cifar100-C-Spat": "spatter",
                "cifar100-C-GaB": "gaussian_blur", 
                "cifar100-C-SpN": "speckle_noise", 
                "cifar100-C-Sat": "saturate", 
}

cifar100_Adv_datasets_list = { 
                            "cifar100-A_PGD": "adversarial_PGD",
                            "cifar100-A_CW": "adversarial_CW",
                            "cifar100-A_Jitter": "adversarial_Jitter",
                            "cifar100-A_PIFGSM": "adversarial_PIFGSM",  
}

imagenet1kv1_C_datasets_list = {
                "imagenet1kv1-Normal": None, 
                "imagenet1kv1-C-Spat": "spatter",
                "imagenet1kv1-C-GaB": "gaussian_blur", 
                "imagenet1kv1-C-SpN": "speckle_noise", 
                "imagenet1kv1-C-Sat": "saturate", 
}

imagenet1kv1_Adv_datasets_list = { 
                            "imagenet1kv1-A_PGD": "adversarial_PGD",
                            "imagenet1kv1-A_CW": "adversarial_CW",
                            "imagenet1kv1-A_Jitter": "adversarial_Jitter",
                            "imagenet1kv1-A_PIFGSM": "adversarial_PIFGSM",                   
}

def check_dataset_in_list(datasets_list, target_dataset):
    for dataset in datasets_list:
        for s in dataset.keys():
            if s in target_dataset:
                if s + "-" in target_dataset:
                    return s, target_dataset.replace(s + "-", "")
                else:
                    return s, False
    return False, False

def split_testing_subsets(dataset, train_flag, random_flag):
    if random_flag:
        all_indices = list(range(len(dataset)))
        
        test_size = int(len(dataset) * 0.3)  # 30% for testing
        local_rand = random.Random(999) 
        testing_set_indices = local_rand.sample(all_indices, test_size)
        
        testing_set = Subset(dataset, testing_set_indices)
        
        if not train_flag:
            return testing_set, None, testing_set_indices
            
        all_indices_set = set(all_indices)
        remaining_indices = list(all_indices_set - set(testing_set_indices))
            
        remaining_data = SubsetWithPseudoLabels(dataset, remaining_indices)
        return testing_set, remaining_data, testing_set_indices
    

def training_dataset_index_split(dataset):
    idx_label_0 = [i for i, label in enumerate(dataset.binary_pseudo_labels) if label == 0]
    idx_label_1 = [i for i, label in enumerate(dataset.binary_pseudo_labels) if label == 1]

    return idx_label_0,idx_label_1
    
def split_training_dataset(dataset,idx_label_0,idx_label_1):  
    if len(idx_label_0) == 0 or len(idx_label_1) == 0:
        return dataset
    else:
        min_count = min(len(idx_label_0), len(idx_label_1))
        
        balanced_idx_label_0 = random.sample(idx_label_0, min_count)
        balanced_idx_label_1 = random.sample(idx_label_1, min_count)
        
        balanced_indices = balanced_idx_label_0 + balanced_idx_label_1
        random.shuffle(balanced_indices)

        training_dataset = Subset(dataset, balanced_indices)

        return training_dataset


def build_dataset(cfg, dataset=None, model=None):
    checked_dataset_list = [cifar10_C_datasets_list,
                            cifar10_Adv_datasets_list,
                            cifar100_C_datasets_list,
                            cifar100_Adv_datasets_list,
                            imagenet1kv1_C_datasets_list,
                            imagenet1kv1_Adv_datasets_list
    ]
    selected_dataset, aux = check_dataset_in_list(checked_dataset_list, dataset)
    print(f"Selected dataset: {selected_dataset}, Aux: {aux}")
    assert selected_dataset

    if "Normal" in selected_dataset:
        return load_normal_dataset(selected_dataset)
    elif any(x in selected_dataset for x in ["cifar10-C", "cifar100-C", "imagenet1kv1-C", "NCTCRCHE100K-C","cifar2-C"]):
        return load_corrupted_dataset(selected_dataset, aux)
    elif any(x in selected_dataset for x in ["cifar10-A", "cifar100-A", "imagenet1kv1-A", "NCTCRCHE100K-A","cifar2-A","EPattack"]):
        return load_adversarial_dataset(selected_dataset, aux, cfg, model)

def load_normal_dataset(dataset):
    if "cifar10-" in dataset:
        return load_cifar10_test("path to the cifar10 dataset")
    elif "cifar100-" in dataset:
        return load_cifar100_test("path to the cifar100 dataset")
    elif "imagenet1kv1-" in dataset:
        return load_imagenet1kv1_test("path to the imagenet1k dataset")

    
def load_corrupted_dataset(dataset, aux):
    stds = {'std_0d01': 0.01, 'std_0d06': 0.06, 'std_0d15': 0.15, 'std_0d6': 0.6} 
    if "cifar10-" in dataset:
        if aux in stds:
            return load_cifar10_c_supp("path to the supplementary cifar10-c dataset",str(cifar10_C_datasets_list[dataset]),std=stds[aux])
        else:
            return load_cifar10_c("path to the cifar10-c dataset", cifar10_C_datasets_list[dataset], int(aux))
    elif "cifar100-" in dataset:
        if aux in stds:
            return load_cifar100_c_supp("path to the supplementary cifar100-c dataset",str(cifar100_C_datasets_list[dataset]),std=stds[aux])
        else:
            return load_cifar100_c("path to the cifar100-c dataset", cifar100_C_datasets_list[dataset], int(aux))
    elif "imagenet1kv1-" in dataset:
        if aux in stds:
            return load_imagenet1kv1_c_supp("path to the supplementary imagenet1k-c dataset",str(imagenet1kv1_C_datasets_list[dataset]),std=stds[aux])
        else:
            return load_imagenet1kv1_c("path to the imagenet1k-c dataset", imagenet1kv1_C_datasets_list[dataset], int(aux))

def load_adversarial_dataset(dataset, aux, cfg, model):
    attack_type = dataset.split('-')[-1]
    base_file_path = "path to the adversarial attack dataset"

    logger.info(f"Base attack file path: {base_file_path}")
    
    if os.path.exists(base_file_path):
        logger.info(f"Loading existing adversarial dataset from {base_file_path}")
    
        return HDF5AdversarialDataset(base_file_path,normalize_fn=normalize_images)


def normalize_images(images):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (images.cpu() - mean) / std
    
    
class DatasetWithPseudoLabels(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.dataset_name = None
        self.binary_pseudo_labels = None
        self.mc_pseudo_labels = None
        self.mc_pseudo_logits = None

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name
        
    def set_binary_pseudo_labels(self, binary_pseudo_labels):
        self.binary_pseudo_labels = binary_pseudo_labels

    def set_mc_pseudo_labels(self, mc_pseudo_labels):
        self.mc_pseudo_labels = mc_pseudo_labels

    def set_mc_pseudo_logits(self, mc_pseudo_logits):
        self.mc_pseudo_logits = mc_pseudo_logits

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, gt_label = self.dataset[idx]
        
        gt_label = torch.tensor(gt_label) if not torch.is_tensor(gt_label) else gt_label

        binary_pseudo_label = self.binary_pseudo_labels[idx] if self.binary_pseudo_labels is not None else None
        mc_pseudo_label = self.mc_pseudo_labels[idx] if self.mc_pseudo_labels is not None else None
        mc_pseudo_logits = self.mc_pseudo_logits[idx] if self.mc_pseudo_logits is not None else None

        return data, gt_label, mc_pseudo_label, binary_pseudo_label, mc_pseudo_logits

class SubsetWithPseudoLabels(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.mc_pseudo_labels = [dataset.mc_pseudo_labels[i] for i in indices]
        self.mc_pseudo_logits = [dataset.mc_pseudo_logits[i] for i in indices]
        self.binary_pseudo_labels = [dataset.binary_pseudo_labels[i] for i in indices]
                
class HDF5AdversarialDataset(Dataset):
    def __init__(self, h5_file, normalize_fn=None):
        self.h5_path = h5_file  
        self.h5_file = None     
        self.normalize_fn = normalize_fn 
        
        try:
            self.h5_file = h5py.File(h5_file, "r")
            self.samples = self.h5_file["samples"] 
            self.labels = self.h5_file["labels"]
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not open HDF5 file at {h5_file}: {e}")
        except KeyError as e:
            self.h5_file.close()  
            raise KeyError(f"Missing required dataset in {h5_file}: {e}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.samples[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.normalize_fn is not None:
            sample = self.normalize_fn(sample)
        
        return sample, label
    
    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
            except (AttributeError, TypeError, ValueError):
                pass 
    

