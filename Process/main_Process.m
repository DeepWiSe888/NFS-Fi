%% Process raw data for the neural network model
clc
clear all;
fclose all;
close all;
%% set path
path_root = '\\10.96.187.229\p2p\NFS-Fi\Raw_Data';
save_root = '\\10.96.187.229\p2p\NFS-Fi\Proc_Data';
if ~exist(save_root, 'dir')
    mkdir(save_root);
    fprintf('Build the path: %s\n', save_root);
end
%% proc parameters
ProPara = para_ProPara();
para_set.ProPara = ProPara; 
para_set.max_seq_length = 240; 
para_set.num_action = 10;
%% proc
for d = 1:56
    fprintf('Processing Domain %d \n', d);
    ind = d-1;
    data_name = ['Domain' num2str(ind)];
    laod_path = [path_root '\Data_' data_name '.mat'];
    load(laod_path)
    data_set = cell(1,size(Label,2));
    for i = 1:size(Label,2)
        data_set{1,i}.act_data = Data{1,i};
        data_set{1,i}.act_label = Label(1,i);
        data_set{1,i}.act_domain = Domain(1,i);
        data_set{1,i}.act_scen = Scena(1,i);
    end  % i = 1:size(Label,2)
    [proc_data,rssi_norm_coeff, csi_abs_norm_coeff, max_seq_length] =...
                func_proc_user_dataset(data_set);
    
    para_set.rssi_norm_coeff = rssi_norm_coeff;
    para_set.csi_abs_norm_coeff = csi_abs_norm_coeff;
    para_set.save_dir = save_root; 

    func_form_train_test_dataset(proc_data,para_set,data_name);

    clearvars Data Label Domain Scena data_set...
        proc_data rssi_norm_coeff csi_abs_norm_coeff max_seq_length
end % i = 1:56
fprintf('Complete! \n')
