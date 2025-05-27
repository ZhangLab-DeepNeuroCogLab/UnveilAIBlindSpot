import torchvision
from torch import nn
from models import *
import Main.set_logging as set_logging
logger = set_logging.get_logger(__name__)


def build_base_model(cfg): 
    if cfg.BASE_MODEL.MODEL_NAME == "resnet50":   
        if cfg.BASE_MODEL.PRETRAIN:
            net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            logger.info("Model {} is trained with pretrained weight.".format(cfg.BASE_MODEL.MODEL_NAME))
            if cfg.BASE_MODEL.CHECKPOINT_FILE_PATH == "default":
                return net
        else:
            net = torchvision.models.resnet50(weights=None)
            logger.info("Model {} is trained from stratch.".format(cfg.BASE_MODEL.MODEL_NAME))
        net.fc = nn.Linear(net.fc.in_features, cfg.BASE_MODEL.NUM_CLASSES, bias=True)
    elif cfg.BASE_MODEL.MODEL_NAME == "vit": 
        if cfg.BASE_MODEL.PRETRAIN:
            net = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
            logger.info("Model {} is trained with pretrained weight.".format(cfg.BASE_MODEL.MODEL_NAME))
            if cfg.BASE_MODEL.CHECKPOINT_FILE_PATH == "default":
                return net
        else:
            net = torchvision.models.vit_b_16(weights=None)
            logger.info("Model {} is trained from stratch.".format(cfg.BASE_MODEL.MODEL_NAME))
        net.heads.head = nn.Linear(net.heads.head.in_features, cfg.BASE_MODEL.NUM_CLASSES, bias=True)
    return net