import torchvision
import torchvision.transforms as transforms
import Main.set_logging as set_logging
logger = set_logging.get_logger(__name__)

def load_cifar100_train(pathroot):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(
        root=pathroot, train=True, download=True, transform=transform_train)
    
    logger.info(f"CIFAR100 file path: {pathroot}")
    
    return trainset

def load_cifar100_test(pathroot):
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    testset = torchvision.datasets.CIFAR100(
        root=pathroot, train=False, download=True, transform=transform_test)
    
    logger.info(f"CIFAR100 file path: {pathroot}")
    
    return testset
