from Dataset.build_dataset import *
from torch.utils.data import DataLoader
import types

def set_dataset_name(self, name):
    self.dataset_name = name
    
    
def data_split_on_mentee(cfg, student_net, split_dataset, train_flag, device):
    student_net.eval()
    teacher_set_list = []
    for dataset_idx, dataset in enumerate(split_dataset):
        test_batch_size = cfg.TEST.BATCH_SIZE

        student_testset = build_dataset(cfg,dataset,student_net)
        if not train_flag:
            student_testset.set_dataset_name = types.MethodType(set_dataset_name, student_testset)
            student_testset.set_dataset_name(dataset)
            teacher_set_list.append(student_testset)
        else:    
            student_indiv_testloader = DataLoader(student_testset, batch_size=test_batch_size, shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS,pin_memory=False)
            print("Test student %s on the Dataset %s"%(cfg.BASE_MODEL.MODEL_NAME,split_dataset[dataset_idx]))
            
            test_loss = 0
            correct = 0
            total = 0
            mc_pseudo_labels = []
            mc_pseudo_logits=[]
            binary_pseudo_labels = []
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(student_indiv_testloader):       
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = student_net(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    binary_pseudo_labels.extend((predicted == targets).cpu().numpy().astype(int))
                    mc_pseudo_labels.extend(predicted.cpu().numpy().astype(int))
                    mc_pseudo_logits.extend(outputs.cpu().numpy())
                    
                    progress_bar(batch_idx, len(student_indiv_testloader), ' Acc: %.3f%% (%d/%d)'
                                    % (100.*correct/total, correct, total))
                
                logger.info("Student Net Performence on {}: | Acc: {:.3f} ({}/{})".format(split_dataset[dataset_idx], 100.*correct/total, correct, total))
                binary_pseudo_labels = torch.from_numpy(np.array(binary_pseudo_labels))
                mc_pseudo_labels = torch.from_numpy(np.array(mc_pseudo_labels))
                mc_pseudo_logits = torch.from_numpy(np.array(mc_pseudo_logits))
                
                dataset_with_pseudo_labels = DatasetWithPseudoLabels(student_testset)
                dataset_with_pseudo_labels.set_dataset_name(dataset)
                dataset_with_pseudo_labels.set_binary_pseudo_labels(binary_pseudo_labels)
                dataset_with_pseudo_labels.set_mc_pseudo_labels(mc_pseudo_labels)
                dataset_with_pseudo_labels.set_mc_pseudo_logits(mc_pseudo_logits)
                teacher_set_list.append(dataset_with_pseudo_labels)
            
    print('==> Split the training and testing set ...')
    teacher_testset_list = []
    teacher_trainset_alldata_list = []
    teacher_testing_set_indice_list = []
    
    for dataset in teacher_set_list:
        dataset_name  = dataset.dataset_name
        if cfg.DATA_SPLIT_MODE == "random":
            print("Splitting the dataset randomly (fixed seed)")
            test_subset,remaining_data,testing_subset_indices = split_testing_subsets(dataset, train_flag = train_flag, random_flag = True)
        else:
            raise ValueError("Unknown data split mode")
            
        teacher_testset_list.append(test_subset)
        teacher_trainset_alldata_list.append(remaining_data)
        teacher_testing_set_indice_list.append(testing_subset_indices)
        
    
    return teacher_testset_list, teacher_trainset_alldata_list, teacher_testing_set_indice_list