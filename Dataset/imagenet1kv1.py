import torchvision
import torchvision.transforms as transforms
import numpy as np
import Main.set_logging as set_logging
logger = set_logging.get_logger(__name__)

def load_imagenet1kv1_test(pathroot):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    testset = torchvision.datasets.ImageFolder(
        root=pathroot,transform=transform_test)
    
    logger.info(f"IMAGENET1KV1 file path: {pathroot}")
    
    return testset
