import os
import logging
# import random
import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import copy
import h5py

# np.random.seed(100)


def split_dataset(dataset, num_classes, x_per_class, hold_class=[]):
    class_indices = {i: [] for i in range(num_classes)}

    # Iterate over the dataset to distribute indices into class buckets
    for idx, (_, label, _) in enumerate(dataset):
        label_idx = np.argmax(label)  # Assuming one-hot encoded labels
        class_indices[label_idx].append(idx)

    # Introduce randomness: Shuffle indices within each class
    FLAG_shuffle = True

    if FLAG_shuffle:
        for indices in class_indices.values():
            np.random.shuffle(indices)

    # Select X samples per class and separate the rest
    selected_indices = []
    remaining_indices = []
    for label_idx, indices in class_indices.items():
        if len(hold_class) > 0:
            if label_idx in hold_class:
                this_x_per_class = 0
            else:
                this_x_per_class = x_per_class
        else:
            this_x_per_class = x_per_class
        selected_indices.extend(indices[:this_x_per_class])
        remaining_indices.extend(indices[this_x_per_class:])

    # Create two datasets
    selected_dataset = Subset_SequenceDataset(dataset, selected_indices)
    remaining_dataset = Subset_SequenceDataset(dataset, remaining_indices)

    return selected_dataset, remaining_dataset   


def split_dataset_each(dataset, num_classes, x_per_class, hold_class=[]):
    class_indices = {i: [] for i in range(num_classes)}
    # Iterate over the dataset to distribute indices into class buckets
    for idx, (data, label, _) in enumerate(dataset):
        label_idx = np.argmax(label)  # Assuming one-hot encoded labels
        class_indices[label_idx].append(idx)

    # Introduce randomness: Shuffle indices within each class
    FLAG_shuffle = True

    if FLAG_shuffle:
        for indices in class_indices.values():
            np.random.shuffle(indices)

    # Select X samples per class and separate the rest
    selected_indices = []
    remaining_indices = []
    for label_idx, indices in class_indices.items():
        if len(hold_class) > 0:
            if label_idx in hold_class:
                this_x_per_class = 0
            else:
                this_x_per_class = x_per_class[label_idx]
        else:
            this_x_per_class = x_per_class[label_idx]
        selected_indices.extend(indices[:this_x_per_class])
        remaining_indices.extend(indices[this_x_per_class:])

    # Create two datasets
    selected_dataset = Subset_SequenceDataset(dataset, selected_indices)
    remaining_dataset = Subset_SequenceDataset(dataset, remaining_indices)

    return selected_dataset, remaining_dataset


# Load data from .mat file
def load_data(file_dir, data_name):
    file_list = os.listdir(file_dir)
    file_list = [file for file in file_list if file.startswith(data_name+'_for_py_dataset_p_')]
    file_list.sort()
    print(file_list)

    all_sequences = []
    all_labels = []
    for file in file_list:
        file_path = os.path.join(file_dir, file)
        print(file_path)
        mat = scipy.io.loadmat(file_path)
        sequences = mat['data']
        labels = mat['label']
        # print(sequences.shape)
        # print(labels.shape)
        all_sequences.append(sequences[0])
        all_labels.append(labels[0])

    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_sequences.shape)
    print(all_labels.shape)
    return all_sequences, all_labels


def load_data_all(file_dir):
    file_list = os.listdir(file_dir)
    file_list.sort()
    print(file_list)

    all_sequences = []
    all_labels = []
    all_domains = []
    for file in file_list:
        file_path = os.path.join(file_dir, file)
        print(file_path)
        mat = scipy.io.loadmat(file_path)   # load data
        sequences = mat['data']
        labels = mat['label']
        domains = mat['domain']

        all_sequences.append(sequences[0])
        all_labels.append(labels[0])
        all_domains.append(domains[0])

    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_domains = np.concatenate(all_domains, axis=0)

    print(all_sequences.shape)
    print(all_labels.shape)
    print(all_domains.shape)
    return all_sequences, all_labels, all_domains

def load_data_selected(file_dir, Selcted):

    file_list = os.listdir(file_dir)
    if not Selcted:
        selected_file = file_list.sort()
    else:
        selected_file = []
        for i in Selcted:
            target_filename = f"Domain{i}_for_py_dataset.mat"
            if target_filename in file_list:
                selected_file.append(target_filename)
            else:
                raise ValueError('No specified files in the folder!')

    print(selected_file)

    all_sequences = []
    all_labels = []
    all_domains = []
    for file in selected_file:
        file_path = os.path.join(file_dir, file)
        print(file_path)
        mat = scipy.io.loadmat(file_path)  # load data
        sequences = mat['data']
        labels = mat['label']
        domains = mat['domain']

        all_sequences.append(sequences[0])
        all_labels.append(labels[0])
        all_domains.append(domains[0])

    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_domains = np.concatenate(all_domains, axis=0)

    print(all_sequences.shape)
    print(all_labels.shape)
    print(all_domains.shape)
    return all_sequences, all_labels, all_domains



def select_domain_and_environment(all_sequences, all_labels, all_domains, sel_test_domain, sel_train_envDomains):

    cache_ind_extract = np.where(all_domains == sel_test_domain)[0]
    target_x = all_sequences[cache_ind_extract]
    target_y = all_labels[cache_ind_extract]
    target_d = all_domains[cache_ind_extract]   # just for checking

    source_x = []
    source_y = []
    source_d = []
    for d in sel_train_envDomains:
        cache_ind = np.where(all_domains == d)[0]
        source_x.extend(all_sequences[cache_ind])
        source_y.extend(all_labels[cache_ind])
        source_d.extend(all_domains[cache_ind])   # just for checking
    source_x = np.array(source_x)
    source_y = np.array(source_y)
    source_d = np.array(source_d)
    return target_x, target_y, target_d, source_x, source_y, source_d


def select_domain(all_sequences, all_labels, all_domains, sel_domain):
    cache_ind_extract = np.where(all_domains == sel_domain)[0]
    target_x = all_sequences[cache_ind_extract]
    target_y = all_labels[cache_ind_extract]
    target_d = all_domains[cache_ind_extract]   # just for checking

    cache_ind_rest = np.where(all_domains != sel_domain)[0]
    source_x = all_sequences[cache_ind_rest]
    source_y = all_labels[cache_ind_rest]
    source_d = all_domains[cache_ind_rest]   # just for checking

    return target_x, target_y, target_d, source_x, source_y, source_d


def split_data_for_all(all_sequences, all_labels, all_domains, ratio, use_domain = None):
    # action labels
    action = np.array([i for i in range(len(ratio))])
    if use_domain is None:
        d_in_data = np.unique(all_domains).astype(int)
    else:
        d_in_data = np.unique(use_domain).astype(int)

    train_x = []
    train_y = []
    train_d = []
    test_x = []
    test_y = []
    test_d = []
    for d in d_in_data:  # domain-subject
        cache_ind = np.where(all_domains == d)[0]
        data_cache = all_sequences[cache_ind]
        y_cache = all_labels[cache_ind]
        y_cache_num = [np.argmax(x[0]) for x in y_cache]
        print(d)
        for i in range(len(action)):  # action
            ac = action[i]
            cache_ac_ind = np.where(y_cache_num == ac)[0]
            len_extract = int(len(cache_ac_ind) * ratio[i])
            FLAG_shuffle = True
            if FLAG_shuffle:
                np.random.shuffle(cache_ac_ind)
            cache_ac_ind_extract = np.copy(cache_ac_ind[0:len_extract])
            cache_ac_ind_rest = np.copy(cache_ac_ind[len_extract:len(cache_ac_ind)])

            train_x.append(data_cache[cache_ac_ind_extract])
            # train_x = np.concatenate(data_cache[cache_ac_ind_extract], axis=0)
            train_y = np.concatenate((train_y, np.copy(y_cache[cache_ac_ind_extract])), axis=0)
            train_d = np.concatenate((train_d, np.full((len(cache_ac_ind_extract)), d)), axis=0)
            test_x.append(data_cache[cache_ac_ind_rest])
            # test_x = np.concatenate(data_cache[cache_ac_ind_rest], axis=0)
            test_y = np.concatenate((test_y, np.copy(y_cache[cache_ac_ind_rest])), axis=0)
            test_d = np.concatenate((test_d, np.full((len(cache_ac_ind_rest)), d)), axis=0)
    train_x = np.concatenate(train_x, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    return train_x, train_y, train_d, test_x, test_y, test_d



# Load data from .mat file with h5py, used for the matlab data v7.3
def load_data_h5py(file_dir, data_name):
    file_list = os.listdir(file_dir)
    file_list = [file for file in file_list if file.startswith(data_name+'_for_py_dataset_p_')]
    file_list.sort()
    print(file_list)

    all_sequences = []
    all_labels = []
    for file in file_list:
        file_path = os.path.join(file_dir, file)
        print(file_path)
        mat = h5py.File(file_path, 'r')
        labels = np.array([i for i in range(len(mat['label']))], dtype=object).reshape(1,-1)
        sequences = np.array([i for i in range(len(mat['label']))], dtype=object).reshape(1,-1)
        sequences_cache = [mat[element[0]][:].transpose(1,0).astype(np.float64) for element in mat['data']]
        labels_cache = [mat[element[0]][:].transpose(1,0).astype(np.uint8) for element in mat['label']]

        for i in range(len(mat['label'])):
            labels[0,i] = labels_cache[i]
            sequences[0,i] = sequences_cache[i]
        # print(sequences.shape)
        # print(labels.shape)
        all_sequences.append(sequences[0])
        all_labels.append(labels[0])

    all_sequences = np.concatenate(all_sequences, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_sequences.shape)
    print(all_labels.shape)
    return all_sequences, all_labels


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.intend_labels = None
        self.seq_len_list = None
        self.determine_seq_len()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # print(self.sequences.shape)
        sequence = self.sequences[idx]
        label = self.labels[idx][0]
        if self.intend_labels is not None:
            intend_label = self.intend_labels[idx][0]
            return sequence, label, intend_label
        if self.seq_len_list is not None:
            return sequence, label, self.seq_len_list[idx]
        else:
            return sequence, label
        
    def determine_seq_len(self):
        self.seq_len_list = []
        for i in range(self.sequences.shape[0]):
            cur_seq = self.sequences[i]
            # cur_seq is a 2D array, with shape (seq_len, feature_dim)
            # the len of cur_seq is the index of the first all -1 feature
            # determine the len of the cur_seq based on above rule
            seq_len = cur_seq.shape[0]
            for j in range(cur_seq.shape[0]):
                if cur_seq[j][0] == -1 and cur_seq[j][1] == -1:
                    seq_len = j
                    break
            self.seq_len_list.append(seq_len-1)
        self.seq_len_list = np.array(self.seq_len_list).reshape(-1,1)

class Subset_SequenceDataset(SequenceDataset):
    def __init__(self, dataset, indices):
        super(Subset_SequenceDataset, self).__init__(dataset.sequences[indices], dataset.labels[indices])
        if dataset.intend_labels is not None:
            self.intend_labels = dataset.intend_labels[indices]
        else:
            self.intend_labels = None
        if dataset.seq_len_list is not None:
            self.seq_len_list = dataset.seq_len_list[indices]
        else:
            self.seq_len_list = None


def save_to_file(dataset:SequenceDataset, file_dir:str, file_name:str):
    file_path = os.path.join(file_dir, file_name + '_dataset.npz')
    if dataset.intend_labels is None:
        np.savez(file_path, data=dataset.sequences, label=dataset.labels)
    else:
        np.savez(file_path, data=dataset.sequences, label=dataset.labels, intend_label=dataset.intend_labels)


def load_data_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    target_x = data['target_x']
    target_y = data['target_y']
    target_d = data['target_d']
    train_sequences = data['train_sequences']
    train_labels = data['train_labels']
    train_domains = data['train_domains']
    test_sequences_source = data['test_sequences_source']
    test_labels_source = data['test_labels_source']
    test_domains_source = data['test_domains_source']
    return (target_x, target_y, target_d, train_sequences, train_labels,
            train_domains, test_sequences_source, test_labels_source,test_domains_source)

def circular_list(total_d, num_d, test_d):
    """
    total_d: total number of domains is total+1 , 0:total
    num_d: total number in this training, including testing and training domains
    test_d: the domain for testing
    """
    result = []
    for i in range(num_d):
        index = (test_d + i) % (total_d + 1)
        result.append(index)
    return result

def setup_logger_file(SAVE_PATH, filename, logger_name,mode):
    """
    Set up the logger for logging summary information.

    Args:
    SAVE_PATH (str): The directory path where the summary log file will be saved.
    filename (str): The name of the summary log file (default is 'GRU_Output_FT_summary.txt').

    Returns:
    logger_summary (logging.Logger): The logger for summary information.
    """
    logging_txt = os.path.join(SAVE_PATH, filename)
    logger_con = logging.getLogger(logger_name)
    logger_con.setLevel(logging.INFO)

    for handler in logger_con.handlers[:]:
        handler.close()
        logger_con.removeHandler(handler)

    # Create file handler with append mode
    file_handler = logging.FileHandler(logging_txt, mode=mode)
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    # Add the file handler to the logger
    logger_con.addHandler(file_handler)

    return logger_con

def test_model(model, data_loader, DEVICE, logger_f, SAVE_PATH, save_name, epoch, FT_epoch):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        pre_result = []
        real_result = []
        for sequences, labels, indeces in data_loader:
            sequences, labels = sequences.to(DEVICE).type(torch.float32), labels.to(DEVICE).type(torch.float32)
            indeces = indeces.cpu().numpy().astype(np.int64).squeeze()
            outputs, feat = model(sequences, indeces)
            _, predicted = torch.max(outputs.data, 1)
            _, labels_real = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
            if epoch == FT_epoch - 1:
                pre_result.extend(predicted.data.cpu().numpy())
                real_result.extend(labels_real.data.cpu().numpy())
        if epoch == FT_epoch - 1:
            scipy.io.savemat(SAVE_PATH + '/' + save_name + '_pre_result.mat', {'pre_result': pre_result})
            scipy.io.savemat(SAVE_PATH + '/' + save_name + '_real_result.mat', {'real_result': real_result})
        FT_target_info = f'FT Epoch [{epoch + 1}/{FT_epoch}] Test accuracy of {save_name} domain: {100 * correct / total}%'
        # FT_target_info = f'FT Epoch [{epoch + 1}/{FT_epoch}] Test accuracy of target domain: {100 * correct / total}%'
        print(FT_target_info)
        logger_f.info(FT_target_info)
    return 100 * correct / total

