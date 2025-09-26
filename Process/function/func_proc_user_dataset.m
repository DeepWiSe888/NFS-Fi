function [proc_data,rssi_norm_coeff, csi_abs_norm_coeff, max_seq_length] = func_proc_user_dataset(data_set)
% data_name: 'train' or 'test'
% data_set = test_data_set; 
    rssi_norm_coeff = [];
    csi_abs_norm_coeff = [];
    max_seq_length = 0;

    proc_data = cell(0);
    id_proc = 1;
    num_data = numel(data_set);
    for id_data = 1:num_data
        % extract certain action samples from the dataset
        raw_data_item = data_set{id_data}.act_data;
        if raw_data_item.flag_cond_satisfied == 0
            continue;
        else
            [result, rssi_norm_coeff, csi_abs_norm_coeff, max_seq_length] = func_form_data_label_pair(raw_data_item, rssi_norm_coeff, csi_abs_norm_coeff, max_seq_length);
            proc_data{id_proc}.data = result;
            proc_data{id_proc}.label = data_set{id_data}.act_label;
            proc_data{id_proc}.domain = data_set{id_data}.act_domain;
            proc_data{id_proc}.scen = data_set{id_data}.act_scen;
            id_proc = id_proc + 1;
        end
    end
end