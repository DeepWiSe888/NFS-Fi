"""
This is a simple demo for NSF-Fi dataset
Train a neural network for human activities recognition
PT: pre-training
FT: fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from GRU_model import GRUClassifier
from tool_repo import (SequenceDataset, split_dataset, load_data_selected,
                       circular_list, setup_logger_file, select_domain, split_data_for_all,
                       split_dataset_each, test_model)
from domain_adaption_tools import ForeverDataIterator


DEVICE_ID = 1
SEED = 10
torch.cuda.set_device(DEVICE_ID)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda')
# #####################################################################################################################
FILE_PATH = '/Data/lixin/P2C_project/NFS_Fi'         # Data path (folder path): all the data
# #####################################################################################################################
# #####################################################################################################################
# region Running parameters
max_epoch = 50               # Training epoch
FT_epoch = 200               # FT epoch
DATA_VEC_LEN = 370           # Length of the data vector
NUM_CLASSES = 10             # Number of classes
BATCH_SIZE_train = 64        # batch size
Early_stopping = 90          # early stopping threshold
FLAG_PreData = True          # Default: True. Prepare data or not.
FLAG_CrossDomain_FT = True   # add PT data in FT stage，default True。
FLAG_last_log = True         # log information
FLAG_Retrain = True          # Train a new model or load. True for the first time  True/False
FLAG_FT = True               # FT or not
FLAG_missAct = False         # absent activities or no

# Custom Dataset
Domain = [i for i in range(16)]
tr_r = [0.9]*10
ft_r = [0.0]*10
FT_ratio = 0.2
PT_FT_ratio = 2
PT_number = 30
per_class_FT_size = [PT_number] * NUM_CLASSES
# Set the activities without FT samples
if FLAG_missAct:
    per_class_FT_size[5] = 0
    per_class_FT_size[8] = 0


total_domain = 5     # total used domain: target domain + source domain
# for sel_domain in range(56):   # use this to compute all the domain
special_list = [1]   # the source domain index
for sel_domain in special_list:  # 56
    Selcted_domain = circular_list(55,total_domain,sel_domain)
    # Creat save path
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    Save_root_dir = parent_directory + '/Result_demo/demo/'   # Result and model path (folder path)
    SAVE_PATH = Save_root_dir + "Result_PTFT" + "_totalD" + str(total_domain) + "_D" + str(sel_domain)
    Load_PATH = SAVE_PATH
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

# endregion
# #####################################################################################################################
# #####################################################################################################################
# region Load data
    all_sequences, all_labels, all_domains = load_data_selected(FILE_PATH,Selcted_domain)   # load all data!
    (target_x, target_y, target_d, source_x,
     source_y, source_d) = select_domain(all_sequences, all_labels, all_domains, sel_domain)   # Label is one-shot [0,0,...,1]
    use_domain = None  # list [1,2,3] or None ! train domain
    (train_sequences, train_labels, train_domains,
     test_sequences_source, test_labels_source, test_domains_source) = split_data_for_all(source_x, source_y, source_d, tr_r, use_domain)

    if train_sequences.shape[0] % BATCH_SIZE_train != 1:
        BATCH_SIZE = BATCH_SIZE_train
    else:
        BATCH_SIZE = BATCH_SIZE_train - 2

    train_dataset = SequenceDataset(train_sequences, train_labels)
    test_dataset = SequenceDataset(target_x, target_y)
    test_dataset_source = SequenceDataset(test_sequences_source, test_labels_source)
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    test_size_source = len(test_dataset_source)
# endregion
# #####################################################################################################################
# #####################################################################################################################
# region predate
    if FLAG_PreData:  # Prepare dataset
        FT_size = sum(per_class_FT_size)
        pure_test_size = test_size - FT_size
        print(f'FT_size: {FT_size}, per_class_FT_size: {per_class_FT_size}, pure_test_size: {pure_test_size}')
        FT_dataset, pure_test_dataset = split_dataset_each(test_dataset, NUM_CLASSES, per_class_FT_size, hold_class = [])

        # AP_PT_dataset: Extract part of pretrain dataset for the FT.
        per_class_PT_size = int(PT_number * PT_FT_ratio)
        PT_dataset, _ = split_dataset(train_dataset, NUM_CLASSES, per_class_PT_size)

        FT_loader = DataLoader(FT_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(pure_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        PT_loader = DataLoader(PT_dataset, batch_size=BATCH_SIZE*PT_FT_ratio, shuffle=True)
        PT_loader_iter = ForeverDataIterator(PT_loader, device=DEVICE)

        test_loader_source = DataLoader(test_dataset_source, batch_size=BATCH_SIZE, shuffle=False)
    else:
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader_source = DataLoader(test_dataset_source, batch_size=BATCH_SIZE, shuffle=False)

    if FLAG_Retrain:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# endregion
# #####################################################################################################################
# #####################################################################################################################
# region write output into txt file
    logger_Train = setup_logger_file(SAVE_PATH, 'GRU_Output_Train.txt', 'Train', 'w')
    logger_FT = setup_logger_file(SAVE_PATH, 'GRU_Output_FT.txt', 'FT', 'w')
    # final summary
    if FLAG_last_log:
        logger_summary = setup_logger_file(Save_root_dir, 'GRU_Output_FT_summary.txt', 'summary', 'a')
# endregion
# #####################################################################################################################
# #####################################################################################################################
# region Model, Loss, Optimizer
    model = GRUClassifier(DATA_VEC_LEN, num_classes=NUM_CLASSES, hidden_size = 64, num_layers = 1).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
# endregion
# #####################################################################################################################
# #####################################################################################################################
# region pre train
    if FLAG_Retrain:   # train a new model or load trained model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Training Loop
        for epoch in range(max_epoch):  # number of epochs
            model.train()
            total = 0
            correct = 0
            for sequences, labels, indeces in train_loader:
                sequences, labels = sequences.to(DEVICE).type(torch.float32), labels.to(DEVICE).type(torch.float32)
                indeces = indeces.cpu().numpy().astype(np.int64).squeeze()   # this is the real length of data, length before padding -1
                # Forward pass
                outputs,_ = model(sequences, indeces)
                loss = criterion(outputs, labels)
                 # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

            train_info = f'Epoch [{epoch + 1}/{max_epoch}], Loss: {loss.item():.4f}, Training accuracy: {100 * correct / total:.4f}%'
            print(train_info)
            logger_Train.info(train_info)

            # test target domain
            test_model(model, test_loader, DEVICE, logger_Train, SAVE_PATH, 'target', epoch, max_epoch)
            # test source domain
            test_model(model, test_loader_source, DEVICE, logger_Train, SAVE_PATH, 'source', epoch, max_epoch)

        # save model
        torch.save(model, os.path.join(SAVE_PATH, 'raw_gru.pth'))
    else:
        del model
        model = torch.load(os.path.join(Load_PATH, 'raw_gru.pth'))
# endregion
# #####################################################################################################################
# #####################################################################################################################
# region FT continue Train
    if FLAG_FT:
        # Differentiated learning rate for the GRU
        for param in model.fc.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = False

        FT_optimizer = torch.optim.AdamW([
            {'params': model.mlp.parameters(), 'lr': 3e-3},
            {'params': model.gru.parameters(), 'lr': 9e-4}],
        weight_decay=1e-3)

        CrossEntropyLoss = torch.nn.CrossEntropyLoss(label_smoothing=0.15)

        for epoch in range(FT_epoch):  # number of epochs
            model.train()

            for sequences, labels, indeces in FT_loader:
                sequences, labels = sequences.to(DEVICE).type(torch.float32), labels.to(DEVICE).type(torch.float32)
                indeces = indeces.cpu().numpy().astype(np.int64).squeeze()

                # Forward pass
                outputs,_ = model(sequences, indeces)
                loss = criterion(outputs, labels)

                # add PT data to the FT
                if FLAG_CrossDomain_FT:
                    batch_src_seq, batch_src_label,batch_src_indeces = next(PT_loader_iter)
                    batch_src_seq = torch.tensor(batch_src_seq,dtype=torch.float32).to(DEVICE)
                    batch_src_label = torch.tensor(batch_src_label,dtype=torch.float32).to(DEVICE)
                    batch_src_indeces = batch_src_indeces.cpu().numpy().astype(np.int64).squeeze()
                    outputs_src,feat_src = model(batch_src_seq,batch_src_indeces)

                    loss_src = criterion(outputs_src, batch_src_label)
                    loss = loss + loss_src

                # Backward and optimize
                FT_optimizer.zero_grad()
                loss.backward()
                FT_optimizer.step()

            FT_loss_info = f'FT Epoch [{epoch + 1}/{FT_epoch}], Loss: {loss.item():.4f}'
            print(FT_loss_info)
            logger_FT.info(FT_loss_info)

            # test target domain
            rec_rate = test_model(model, test_loader, DEVICE, logger_FT, SAVE_PATH, 'FT_target', epoch, FT_epoch)
            # test source domain
            test_model(model, test_loader_source, DEVICE, logger_FT, SAVE_PATH, 'FT_source', epoch, FT_epoch)

            if rec_rate > Early_stopping:
                print(f'Early Stopping at Epoch: {epoch}')
                epoch = FT_epoch - 1
                # test target domain
                test_model(model, test_loader, DEVICE, logger_FT, SAVE_PATH, 'FT_target', epoch, FT_epoch)
                # test source domain
                test_model(model, test_loader_source, DEVICE, logger_FT, SAVE_PATH, 'FT_source', epoch, FT_epoch)
                break

        torch.save(model, os.path.join(SAVE_PATH, 'raw_gru_FT_0.pth'))
# endregion
# #####################################################################################################################
# #####################################################################################################################
# region del
    del model, target_x, target_y, target_d, source_x, source_y,\
        source_d, train_sequences, train_labels, train_domains,\
        test_sequences_source, test_labels_source, test_domains_source
    torch.cuda.empty_cache()