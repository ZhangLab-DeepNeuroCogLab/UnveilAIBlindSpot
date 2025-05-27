import torchvision
import torch
from torch import nn
import torchvision.models as models
import Main.set_logging as set_logging
import torch.nn.functional as F

logger = set_logging.get_logger(__name__)


class ResNet50_Teacher(nn.Module):
    def __init__(self,cfg):
        super(ResNet50_Teacher, self).__init__()
        self.net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Identity()
                
        self.mc_classifier = nn.Sequential(
            nn.Linear(2048, 256, bias=True),  
            nn.ReLU(),                   
            nn.Linear(256, 256, bias=True), 
            nn.ReLU(),  
            nn.Dropout(p=0.2),
            nn.Linear(256, cfg.TEACHER.NUM_CLASSES, bias=True)                 
        )
        self.binary_classifier = nn.Sequential(
            nn.Linear(2048, 256, bias=True),  
            nn.ReLU(),                   
            nn.Linear(256, 256, bias=True), 
            nn.ReLU(),  
            nn.Dropout(p=0.2),
            nn.Linear(256, 1, bias=True)                 
        )
        
    def forward(self, x):
        feature = self.net(x)
        mc_logit = self.mc_classifier(feature)
        binary_logit = self.binary_classifier(feature)
        return mc_logit, binary_logit


class ViT_Teacher(nn.Module):
    def __init__(self,cfg):
        super(ViT_Teacher, self).__init__()
        self.net = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.net.heads.head = nn.Identity()
        
        self.inter_dim = 256
        self.mc_classifier = nn.Sequential(
            nn.Linear(768, self.inter_dim, bias=True),  
            nn.ReLU(),                 
            nn.Linear(self.inter_dim, self.inter_dim, bias=True), 
            nn.ReLU(),  
            nn.Dropout(p=0.2),
            nn.Linear(self.inter_dim, cfg.TEACHER.NUM_CLASSES, bias=True)                 
        )
        self.binary_classifier = nn.Sequential(
            nn.Linear(768, self.inter_dim, bias=True),  
            nn.ReLU(),                 
            nn.Linear(self.inter_dim, self.inter_dim, bias=True), 
            nn.ReLU(),  
            nn.Dropout(p=0.2),
            nn.Linear(self.inter_dim, 1, bias=True)                 
        )
                    
    def forward(self, x):      
        feature = self.net(x)
        mc_logit = self.mc_classifier(feature)
        binary_logit = self.binary_classifier(feature)
        return mc_logit, binary_logit


def build_teacher(cfg):
    if cfg.TEACHER.MODEL_NAME == "resnet50": 
        net = ResNet50_Teacher(cfg)
    elif cfg.TEACHER.MODEL_NAME == "vit":   
        net = ViT_Teacher(cfg)
    
    logger.info("Model {} is used as the teacher.".format(cfg.TEACHER.MODEL_NAME))
    return net
    