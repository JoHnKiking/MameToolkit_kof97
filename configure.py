from attrdict import AttrDict


# v2 base config
v2_base_config = AttrDict({
    "env_str": "V2_env",                                    # train env name
    "env_id": "kof97:kof97-v1",                             # train env id
    "cpu_num": 3,                                           # train env number
    "trail": 12,                                            # train number
    "target_learning_rate": 3.058232389849691e-04,          # train learning rate
    "CH": "ctj",                                            # train character name
    "level": "8",                                           # train level
    "render": True,
})

paths= AttrDict({
    "log_dir": f"ts-log/{v2_base_config.env_str}",                                                  # log dir output path
    "model_input_path": "./opt_2/",                                                                 # model zip input path
    "model_output_path": f"./logs/trail_{v2_base_config.trail}/",                                   # model zip output path
    "win_rate_data": "./win_rate_data/",                                                            # win rate output path
})  

names= AttrDict({                     
    "tb_log_name": f"PPO_V2_t3_transfer_CH_{v2_base_config.CH}",                                        # log name
    "model_input_name": f"trail_{v2_base_config.CH}_{v2_base_config.level}_last_model.zip",             # model zip input name
    "model_output_name": f"Kof97_PPO_V2_t3_{v2_base_config.trail}_transfer_CH_{v2_base_config.CH}",     # model zip output name
})

freq= AttrDict({
    "model_zip_freq": 10000,       # model zip output freq
    "win_rate_freq": 10000,        # win rate output freq
    "total_freq": 3000000,      # total frequency
})  

# default model params
params= {
    'batch_size': 2048,
    'learning_rate': 1.058232389849691e-04,
}

# linux fatal config
linux_config = AttrDict({
    "v2": AttrDict({
        "env_str": v2_base_config.env_str,
        "env_id": v2_base_config.env_id,
        "cpu_num": v2_base_config.cpu_num,
        "trail": v2_base_config.trail,
        "target_learning_rate": v2_base_config.target_learning_rate,
        "CH": v2_base_config.CH,
        "level": v2_base_config.level,
        "render": v2_base_config.render,
        "params": params,
        "paths": paths,
        "names": names,
        "freq": freq,
    }),
    "LSTM": AttrDict({

    })
})



win32_config = AttrDict({
        # 
        "env_id":"kof97:kof97-v1",

        # model params:
        "params": params  

})

import sys

def get_config():
    if sys.platform == 'win32':
        return win32_config
    elif sys.platform == 'linux':
        return linux_config
    else:
        return default_config

