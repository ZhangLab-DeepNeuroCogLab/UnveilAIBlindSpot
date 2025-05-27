import sys
sys.path.append('replace to your own path to this directory.')

import os
import argparse
from Config.defaults import assert_and_infer_cfg
from Utils.parser import load_config, parse_args

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import pprint
import shutil

from torch.utils.data import DataLoader, ConcatDataset
from Main.data_split_on_mentee import *
from Dataset.build_dataset import *
from models.build_model import *
from models.build_teacher import *
from Utils.loss import *
from utils import progress_bar
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR

import Main.set_logging as set_logging
logger = set_logging.get_logger(__name__)

def get_optimizer(model,cfg):
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if any(nd in name for nd in ['bias', 'norm', 'pos_embed', 'cls_token']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    return optim.AdamW(param_groups, lr=cfg.SOLVER.BASE_LR)
  
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = parse_args() 
for path_to_config in args.cfg_files: 
    cfg = load_config(args, path_to_config) 
    cfg = assert_and_infer_cfg(cfg) 
    shutil.copy(path_to_config, cfg.OUTPUT_DIR)
    set_logging.setup_logging(cfg.OUTPUT_DIR,overwrite=True)  
    logger.info("config files: {}".format(args.cfg_files))
    set_seed(cfg.RNG_SEED)

logger.info("Train with config:")
logger.info(pprint.pformat(cfg))
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion_BCE = nn.BCEWithLogitsLoss()
criterion_Dist = DistillationLoss(temperature=1.0)

print('==> Building student net..')
student_net = build_base_model(cfg)
student_net = student_net.to(device)

if device == 'cuda':
    student_net = torch.nn.DataParallel(student_net)
    logger.info("Numer of GPUs: {}".format(torch.cuda.device_count()))

print("cfg.BASE_MODEL.CHECKPOINT_FILE_PATH:",cfg.BASE_MODEL.CHECKPOINT_FILE_PATH)  
if cfg.BASE_MODEL.CHECKPOINT_FILE_PATH and cfg.BASE_MODEL.CHECKPOINT_FILE_PATH != "default":
    trained_student_net = torch.load(cfg.BASE_MODEL.CHECKPOINT_FILE_PATH,map_location="cpu")
    student_net.load_state_dict(trained_student_net['state_dict']) 
    logger.info("Student {} is started from the checkpoint {}.".format(cfg.BASE_MODEL.MODEL_NAME,str(cfg.BASE_MODEL.CHECKPOINT_FILE_PATH)))
    
if cfg.TRAIN.ENABLE:
    print('==> Obtain the performance of student net..')
    teacher_testset_list, teacher_trainset_alldata_list,_ = data_split_on_mentee(cfg,student_net,cfg.TRAIN.DATASET,train_flag = True, device = device)

print('==> Building the teacher..')
teacher = build_teacher(cfg)
teacher = teacher.to(device)

if device == 'cuda':
    teacher = torch.nn.DataParallel(teacher)
    
if cfg.TRAIN.ENABLE and cfg.TRAIN.CHECKPOINT_FILE_PATH:
    trained_model = torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH,map_location="cpu")
    teacher.load_state_dict(trained_model['state_dict'])  
    logger.info("TRAIN CHECKPOINT FILE PATH is {}".format(cfg.TRAIN.CHECKPOINT_FILE_PATH))

optimizer = get_optimizer(teacher,cfg)
scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    
def train(epoch,train_indices):
    teacher_trainset_list = []
    for dataset, (idx_0, idx_1) in zip(teacher_trainset_alldata_list,train_indices):
        train_subset = split_training_dataset(dataset,idx_0, idx_1)
        teacher_trainset_list.append(train_subset)
    teacher_trainset = ConcatDataset(teacher_trainset_list)
    teacher_trainloader = torch.utils.data.DataLoader(
    teacher_trainset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.DATA_LOADER.NUM_WORKERS,pin_memory=False)
        
    print('\nEpoch: %d/%d Student: %s Teacher: %s' % (epoch+1,cfg.SOLVER.MAX_EPOCH,cfg.BASE_MODEL.MODEL_NAME,cfg.TEACHER.MODEL_NAME))
    teacher.train()
    train_loss = 0
    binary_correct = 0
    mc_correct = 0
    total = 0
    total_0 = 0
    total_1 = 0
    binary_correct_0 = 0
    binary_correct_1 = 0
    
    for batch_idx, (inputs, gt_labels,mc_pseudo_label, binary_pseudo_labels,mc_pseudo_logits) in enumerate(teacher_trainloader):
        inputs, gt_labels, mc_pseudo_label, binary_pseudo_labels,mc_pseudo_logits= inputs.to(device), gt_labels.to(device), mc_pseudo_label.to(device), binary_pseudo_labels.to(device),mc_pseudo_logits.to(device)
        optimizer.zero_grad()
                        
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            mc_logit, binary_logit = teacher(inputs)
            binary_predicted = (torch.sigmoid(binary_logit) >= 0.50).int().squeeze() 
            mc_predicted = mc_logit.max(1)[1]

            progress = (epoch + 1) / cfg.SOLVER.MAX_EPOCH
            alpha_weight = progress ** cfg.ALPHA 
            alpha_weight = min(alpha_weight, 1.0)
            
            loss = alpha_weight*criterion_BCE(binary_logit, binary_pseudo_labels.view(-1,1).float())
            loss += (1-alpha_weight)*criterion_Dist(mc_logit, mc_pseudo_logits)     
                
        if math.isnan(loss):
            raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    
        if cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                teacher.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        parameters = [p for p in teacher.parameters() if p.grad is not None]
        grad_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach(), 2.0).to(device)
                    for p in parameters
                ]
            ),
            2.0,
        )
        
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        total += binary_pseudo_labels.size(0)
        total_0 += (binary_pseudo_labels == 0).sum().item()
        total_1 += (binary_pseudo_labels == 1).sum().item()
                
        mc_correct += mc_predicted.eq(mc_pseudo_label).sum().item()
        binary_correct += binary_predicted.eq(binary_pseudo_labels).sum().item()
        binary_correct_0 += (binary_predicted.eq(binary_pseudo_labels) & (binary_pseudo_labels == 0)).sum().item()
        binary_correct_1 += (binary_predicted.eq(binary_pseudo_labels) & (binary_pseudo_labels == 1)).sum().item()
        acc_binary_0 = 100.*binary_correct_0/total_0 if total_0 != 0 else 0
        acc_binary_1 = 100.*binary_correct_1/total_1 if total_1 != 0 else 0
        
        progress_bar(batch_idx, len(teacher_trainloader), 
        'lr:%.6f |Loss: %.3f |GradNorm: %.3f |M Pse. Acc: %.3f%% (%d/%d) |Bin Pse. Acc: %.3f%% (%d/%d) 0 - %.3f%% (%d/%d), 1 - %.3f%% (%d/%d)'
        % (optimizer.param_groups[0]['lr'], train_loss/(batch_idx+1), grad_norm,
        100.*mc_correct/total, mc_correct, total,
        100.*binary_correct/total, binary_correct, total,
        acc_binary_0, binary_correct_0, total_0, 
        acc_binary_1, binary_correct_1, total_1
        ))   
        
    logger.info("|Training epoch: {}/{} |lr: {:.8f} |Loss: {:.3f} |ALPHA: {:.8f} |MCAcc: {:.3f} ({}/{}) |BinAcc: {:.3f} ({}/{})"
                .format(epoch+1,cfg.SOLVER.MAX_EPOCH,optimizer.param_groups[0]['lr'],train_loss/(batch_idx+1), alpha_weight,
                        100.*mc_correct/total, mc_correct, total,
                        100.*binary_correct/total, binary_correct, total))  

def test(epoch,teacher,test_dataset,test_dataset_name):
    teacher_indiv_testloader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=False) 
    teacher.eval()
    student_net.eval() 
    
    with torch.no_grad():
        print("Dataset:%s Student: %s Teacher: %s"% (test_dataset_name,cfg.BASE_MODEL.MODEL_NAME,cfg.TEACHER.MODEL_NAME))
        total = 0
        total_0 = 0
        total_1 = 0
        binary_pseudo_correct_0 = 0
        binary_pseudo_correct_1 = 0 
        student_net_gt_correct = 0
        mc_pseudo_correct = 0
        mc_gt_correct = 0
        average_val_loss = 0
        
        student_binary_pseudo_labels = []
        student_mc_pseudo_labels = []
        teacher_binary_pseudo_labels = []
        teacher_mc_pseudo_labels = []
    
            
        for batch_idx, data in enumerate(teacher_indiv_testloader):
            if len(data) == 5:
                inputs, gt_label,_,_,_ = data
                inputs, gt_label = inputs.to(device), gt_label.to(device)
            else:
                inputs, gt_label = data
                inputs, gt_label = inputs.to(device), gt_label.to(device)

            student_net_outputs= student_net(inputs)
            mc_student_net_prediction = student_net_outputs.max(1)[1]          
               
            binary_student_net_prediction = (mc_student_net_prediction == gt_label).squeeze()
            student_mc_pseudo_labels.extend(mc_student_net_prediction.cpu().numpy().astype(int))
            student_binary_pseudo_labels.extend(binary_student_net_prediction.cpu().numpy().astype(int))
            
            mc_output,binary_output = teacher(inputs)
            binary_teacher_prediction = (torch.sigmoid(binary_output) >= 0.50).int().squeeze() 
            mc_teacher_prediction = mc_output.max(1)[1] 
            
            average_val_loss += criterion_BCE(binary_output, binary_student_net_prediction.view(-1,1).float()).item()
   
            teacher_binary_pseudo_labels.extend(binary_teacher_prediction.cpu().numpy().astype(int))
            teacher_mc_pseudo_labels.extend(mc_teacher_prediction.cpu().numpy().astype(int))

            total += gt_label.size(0)
            total_0 += (binary_student_net_prediction == 0).sum().item()
            total_1 += (binary_student_net_prediction == 1).sum().item()
            
            binary_pseudo_correct_0 += (binary_teacher_prediction.eq(binary_student_net_prediction) & (binary_student_net_prediction == 0)).sum().item()
            binary_pseudo_correct_1 += (binary_teacher_prediction.eq(binary_student_net_prediction) & (binary_student_net_prediction == 1)).sum().item()
            acc_binary_pseudo_correct_0 = 100.*binary_pseudo_correct_0/total_0 if total_0 != 0 else 0
            acc_binary_pseudo_correct_1 = 100.*binary_pseudo_correct_1/total_1 if total_1 != 0 else 0
            
            mc_gt_correct += mc_teacher_prediction.eq(gt_label).sum().item()
            acc_mc_gt = 100.*mc_gt_correct/total
                
            mc_pseudo_correct += mc_teacher_prediction.eq(mc_student_net_prediction).sum().item()
            acc_mc_pseudo = 100.*mc_pseudo_correct/total
            
            student_net_gt_correct += mc_student_net_prediction.eq(gt_label).sum().item()
            acc_student_net_gt = 100.*student_net_gt_correct/total
                    
            progress_bar(batch_idx, len(teacher_indiv_testloader), 'St. GT Acc: %.3f%% (%d/%d) | Te. GT Acc: %.3f%% (%d/%d) |Te. MC Pse Acc: %.3f%% (%d/%d) | Te. Bin. Pse Acc: %.3f%% with 0 - %.3f%% (%d/%d), 1 - %.3f%% (%d/%d)' 
                            % (acc_student_net_gt, student_net_gt_correct, total, 
                            acc_mc_gt, mc_gt_correct, total,
                            acc_mc_pseudo, mc_pseudo_correct, total,
                            (acc_binary_pseudo_correct_0+acc_binary_pseudo_correct_1)/2.0, 
                            acc_binary_pseudo_correct_0, binary_pseudo_correct_0, total_0, 
                            acc_binary_pseudo_correct_1, binary_pseudo_correct_1, total_1))
            
            
            dataset_binary_pseudo_acc = (acc_binary_pseudo_correct_0+acc_binary_pseudo_correct_1)/2.0 
            average_val_loss = average_val_loss / len(teacher_indiv_testloader)
            
            
        logger.info("|Testing epoch {} on {}: | Avg. Val Loss: {:.3f} |St. GT Acc: {:.3f} ({}/{})  |Te. GT Acc: {:.3f} ({}/{}) |Te. MC Pse. Acc:{:.3f} ({}/{}) |Te. Bin. Pse. Acc:{:.3f} |0 - {:.3f} ({}/{}) |1 - {:.3f} ({}/{}))"
                .format(epoch+1, test_dataset_name, average_val_loss,
                        acc_student_net_gt, student_net_gt_correct, total, 
                        acc_mc_gt, mc_gt_correct, total,
                        acc_mc_pseudo, mc_pseudo_correct, total,
                        (acc_binary_pseudo_correct_0+acc_binary_pseudo_correct_1)/2.0, 
                        acc_binary_pseudo_correct_0, binary_pseudo_correct_0, total_0, 
                        acc_binary_pseudo_correct_1, binary_pseudo_correct_1, total_1))
    
        
    return dataset_binary_pseudo_acc, average_val_loss
    

if cfg.TRAIN.ENABLE:
    print('==> Training the teacher..')
    
    start_epoch = 0 
    best_avg_binary_pseudo_acc = 0.0
    worst_avg_val_loss = 1e10
    best_epoch = 0
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.SOLVER.MAX_EPOCH, last_epoch=-1)
    
    train_idxes = []
    for dataset in teacher_trainset_alldata_list:
        train_idx = training_dataset_index_split(dataset)
        train_idxes.append(train_idx)
       
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH): 
        train(epoch,train_idxes)
        scheduler.step()
         
        if (epoch+1)%cfg.TEST.TEST_PERIOD == 0:
            print('==> Testing the teacher..')
            
            epoch_acc = 0.0
            for teacher_testset_idx,teacher_testset in enumerate(teacher_testset_list):
                one_epoch_acc, one_epoch_val_loss = test(epoch, teacher,teacher_testset, cfg.TRAIN.DATASET[teacher_testset_idx])
                epoch_acc += one_epoch_acc

            epoch_acc = epoch_acc/(len(teacher_testset_list)) 
            epoch_val_loss = one_epoch_val_loss/(len(teacher_testset_list))
            
            if epoch_acc >= best_avg_binary_pseudo_acc:
                print('Saving Checkpoint!!! Avg. Binary Pseudo Acc: %.3f. Avg. Val loss: %.3f.' % (epoch_acc,epoch_val_loss) )
                states = {
                    'epoch': epoch+1,
                    'acc': epoch_acc,
                    'teacher': cfg.TEACHER.MODEL_NAME,
                    'state_dict': teacher.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                ckpt_folder_path = os.path.join(cfg.OUTPUT_DIR, "best_checkpoint")
                if os.path.exists(ckpt_folder_path):
                    shutil.rmtree(ckpt_folder_path)
                os.makedirs(ckpt_folder_path)
                name = "best_checkpoint_epoch_{:05d}.pyth".format(epoch + 1)
                save_ckpt_path = os.path.join(ckpt_folder_path, name)
                torch.save(states, save_ckpt_path)
                best_avg_binary_pseudo_acc = epoch_acc
                best_epoch = epoch+1
                           
    logger.info("Best Val acc {}, Val loss {}, on epoch {}".format(best_avg_binary_pseudo_acc, worst_avg_val_loss, best_epoch))
    

if cfg.TEST.ENABLE:
    if cfg.TEST.CHECKPOINT_FILE_PATH:
        trained_model = torch.load(cfg.TEST.CHECKPOINT_FILE_PATH,map_location="cpu")
        logger.info("TEST CHECKPOINT FILE PATH is {}".format(cfg.TEST.CHECKPOINT_FILE_PATH))
        teacher.load_state_dict(trained_model['state_dict']) 
    else:
        trained_model = torch.load(save_ckpt_path,map_location="cpu")
        logger.info("From the best CHECKPOINT!")
        teacher.load_state_dict(trained_model['state_dict'])   
    
    specific_binary_pseudo_acc ={}
    print('==> Final Test the teacher..\n')
    
    for idx, dataset in enumerate(cfg.TEST.DATASET):
        cur_dataset_name = cfg.TEST.DATASET[idx]          
        teacher_extra_testset, _, common_test_id_list = data_split_on_mentee(cfg,student_net,[dataset],train_flag = False, device = device)
        dataset_acc,_ = test(999, teacher, teacher_extra_testset[0], cur_dataset_name)
        specific_binary_pseudo_acc[cur_dataset_name] = dataset_acc
    
    avg_binary_pseudo_acc = torch.mean(torch.tensor(list(specific_binary_pseudo_acc.values())))
    

logger.info("Student:{}, Teacher:{}, Avg acc: {}".format(cfg.BASE_MODEL.MODEL_NAME,cfg.TEACHER.MODEL_NAME,avg_binary_pseudo_acc))
logger.info("Specific Binary Pseudo Acc: {}".format(specific_binary_pseudo_acc))
