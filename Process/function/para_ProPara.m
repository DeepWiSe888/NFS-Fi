function ProPara = para_ProPara()
    ProPara = struct();
    ProPara.check_dua = 262.0;            % Error if sampling length < this value
    ProPara.sampling_time = 270;          % Predefined sampling time 
    ProPara.freeze_time = 9;              % Pre/post freeze duration
    ProPara.slice_time = 3;               % Duration per action
    ProPara.sampling_num_check = 10;      % Verify minimum samples per second requirement
    ProPara.sampling_dua_check = 0.50;    % Verify minimum data interval
    ProPara.sampling_last_check = 1.00;   % Verify data duration
    ProPara.flag_exclude_last_sec = 1;    % Final stationary period
    ProPara.action_duration = ProPara.slice_time - ProPara.flag_exclude_last_sec;
    ProPara.time_embed_dim = 16;          % Time embedding dimension
end