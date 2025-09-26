function [result,rssi_norm_coeff,csi_abs_norm_coeff, max_seq_length]  = func_form_data_label_pair(raw_data_item, rssi_norm_coeff, csi_abs_norm_coeff, max_seq_length)
%{
    temp_act.csi (sample_num, 2*subcarrier_num) 2 stands for Rx1, Rx2;
    temp_act.rssi 
    temp_act.time 

    temp_act.num_point
    temp_act.max_interval
    temp_act.mean_interval
    temp_act.spanning
    
    temp_act.flag_cond_satisfied
%}
    if raw_data_item.flag_cond_satisfied == 0
        result = struct();
        return;
    end

    subcarrier_num = size(raw_data_item.csi, 2) / 2;
    temp = raw_data_item.csi(:, 1:subcarrier_num) .* conj(raw_data_item.csi(:, subcarrier_num+1:end)); 
    result.csi_abs_ratio = abs(temp);
    is_nan_indicator = isnan(temp);
    temp(is_nan_indicator) = 0;
    result.csi_angle_cos_norm = cos(angle(temp))/2+0.5;
    result.csi_angle_sin_norm = sin(angle(temp))/2+0.5;
    result.rssi = raw_data_item.rssi;
    result.time = raw_data_item.time;

    temp = [min(result.rssi,[],1)', max(result.rssi,[],1)']; 
    if isempty(rssi_norm_coeff)
        rssi_norm_coeff = temp;
    else
        rssi_norm_coeff = [min(rssi_norm_coeff(:,1), temp(:,1)), max(rssi_norm_coeff(:,2), temp(:,2))];
    end

    temp = [min(result.csi_abs_ratio,[],1)', max(result.csi_abs_ratio,[],1)']; 
    if isempty(csi_abs_norm_coeff)
        csi_abs_norm_coeff = temp;
    else
        csi_abs_norm_coeff = [min(csi_abs_norm_coeff(:,1), temp(:,1)), max(csi_abs_norm_coeff(:,2), temp(:,2))];
    end
    
    if max_seq_length < size(result.time, 1)
        max_seq_length = size(result.time, 1);
    end
end