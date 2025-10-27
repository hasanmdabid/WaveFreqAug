dataset_configs = {
    "ETTh1": {
        "data_path": "/home/abid/a_c_p/dataset/ETTh1.csv",
        "data_name": "ETTh1",
        "seq_len": 336,
        "pred_lens": [96, 192, 336, 720],
        "enc_in": 7,
        "batch_size": 32,
        "aug_types": ["Wave-Freq"],
        "aug_params": {
            96: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            192: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            336: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            720: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
        },
    },
    "ETTh2": {
        "data_path": "/home/abid/a_c_p/dataset/ETTh2.csv",
        "data_name": "ETTh2",
        "seq_len": 336,
        "pred_lens": [96, 192, 336, 720],
        "enc_in": 7,
        "batch_size": 32,
        "aug_types": ["Wave-Freq"],
        "aug_params": {
            96: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            192: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            336: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            720: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
        },
    },    
    "ILI": {
        "data_path": "/home/abid/a_c_p/dataset/national_illness.csv",
        "data_name": "ILI",
        "seq_len": 36,
        "enc_in": 7,
        "batch_size": 32,
        "pred_lens": [24, 36, 48, 60],
        "aug_types": ["Wave-Freq"],
        "aug_params": {
            24: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            36: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            48: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            60: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
        },
    },
    "weather": {
        "data_path": "/home/abid/a_c_p/dataset/weather.csv",
        "data_name": "weather",
        "seq_len": 336,
        "pred_lens": [96, 192, 336, 720],
        "enc_in": 21,
        "batch_size": 32,
        "aug_types": ["Wave-Freq"],
        "aug_params": {
            96: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            192: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            336: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            720: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
        },
    },
    # Add other datasets as needed
}
