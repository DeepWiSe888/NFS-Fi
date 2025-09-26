function result = func_time_embedding(time_vec, max_time_scale, time_embed_len)
%{
time_vec = temp_proc_data_item.time;       % timestamp
max_time_scale = ProPara.action_duration;  % 2s
time_embed_len = ProPara.time_embed_dim;   % 16
%}
    result = zeros(length(time_vec), time_embed_len);
    for k = 1:time_embed_len/2
        scale = max_time_scale ^ (2.0 * k / time_embed_len);
        result(:, 2*k-1) = sin(time_vec / scale);
        result(:, 2*k) = cos(time_vec / scale);
    end
end