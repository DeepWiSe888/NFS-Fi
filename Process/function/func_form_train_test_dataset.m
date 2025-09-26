function func_form_train_test_dataset(proc_data,para_set,data_name)
     
    collect_data_mat = {};
    collect_label_mat = {};
    seq_len_list = [];
    collect_id = 1;

    % unpack para_set
    rssi_norm_coeff = para_set.rssi_norm_coeff;
    csi_abs_norm_coeff = para_set.csi_abs_norm_coeff;
    max_seq_length = para_set.max_seq_length;
    save_dir = para_set.save_dir;
    num_action = para_set.num_action;
    ProPara = para_set.ProPara;

    num_data = numel(proc_data);

    for id_data = 1:num_data 
        
        id_action = proc_data{id_data}.label;
        action_label_vec = zeros(1, num_action);
        action_label_vec(id_action+1) = 1;    

        temp_proc_data_item = proc_data{id_data}.data;
        % (x - min)/(max-min)
        % size: time * 3 
        rssi_norm = (temp_proc_data_item.rssi - rssi_norm_coeff(:,1)') ./ (rssi_norm_coeff(:,2)' - rssi_norm_coeff(:,1)');
        % size: time * 117 
        csi_abs_ratio_norm = (temp_proc_data_item.csi_abs_ratio - csi_abs_norm_coeff(:, 1)') ./ (csi_abs_norm_coeff(:, 2)' - csi_abs_norm_coeff(:, 1)');
        % time * 16 
        time_cache = [0;diff(temp_proc_data_item.time)]*10; 
        time_embed = func_time_embedding(time_cache, ProPara.action_duration, ProPara.time_embed_dim);
        % size: time * [16 + 3 + 117 + 117 + 117] = 370
        data_mat = [time_embed, rssi_norm, csi_abs_ratio_norm, temp_proc_data_item.csi_angle_cos_norm, temp_proc_data_item.csi_angle_sin_norm];
        seq_len_list(collect_id) = size(data_mat, 1);
        padding_len = max_seq_length - size(data_mat, 1);
        if padding_len < 0
            % select max_seq_length items from data_mat in an ordered way
            sel_index = round(linspace(1, size(data_mat, 1), max_seq_length));
            % ensure unique
            sel_index = unique(sel_index);
            data_mat = data_mat(sel_index, :);
            padding_len = max_seq_length - size(data_mat, 1);
            % disp(id_action)
        end
        % padding -1
        collect_data_mat{collect_id} = [data_mat; -1 * ones(padding_len, size(data_mat, 2))];
        collect_label_mat{collect_id} = action_label_vec;
        collect_domain_mat{collect_id} = proc_data{id_data}.domain;
        collect_scen_mat{collect_id} = proc_data{id_data}.scen;
        collect_id = collect_id + 1;
    end

    fprintf('==> %s ==> Dimension of feature vector: %d \n', data_name, size(collect_data_mat{1}, 2));

    figure('Visible','off');
    histogram(seq_len_list);
    title(['Histogram of sequence length ', data_name]);
    xlabel('Sequence length');
    ylabel('Number of sequences');
    saveas(gcf, [save_dir, '/histogram_of_sequence_length',data_name,'.png']);
    saveas(gcf, [save_dir, '/histogram_of_sequence_length',data_name,'.fig']);


    num_elements = numel(collect_data_mat);
    elements_per_file = 3000;  
    num_files = ceil(num_elements / elements_per_file);
    % Loop through each segment and save
    for i = 1:num_files
        % Calculate the range of elements for this file
        start_idx = (i - 1) * elements_per_file + 1;
        end_idx = min(i * elements_per_file, num_elements);

        % Extract the segment
        data = collect_data_mat(start_idx:end_idx);
        label = collect_label_mat(start_idx:end_idx);
        domain = collect_domain_mat(start_idx:end_idx);
        scen = collect_scen_mat(start_idx:end_idx);
        % dateInfo = collect_dateInfo_mat(start_idx:end_idx);

        % Construct the filename
        filename = sprintf('%s/%s_for_py_dataset.mat', save_dir,data_name);

        for n = 1:length(data)
            hasNaN = any(isnan(data{1,n}), 'all');
            if hasNaN
                NaN_cache = isnan(data{1,n});
                [rowIndices, colIndices] = find(NaN_cache);
                for m = 1:length(rowIndices)
                    data{1,n}(rowIndices,colIndices) = -1;
                end
            end
        end

        % Save this segment
        save(filename, 'data', 'label', 'domain', 'scen', 'max_seq_length', '-v7'); % '-v6'/-v7/-v7.3
    end
fprintf('============ %s dataset generation has finished ============ \n', data_name);
end

