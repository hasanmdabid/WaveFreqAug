import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataset_configs = {
    "ETTh1": {
        "data_path": "/home/abidhasan/Document/Project/WaveFreqAug_Forecasting/dataset/ETTh1.csv",
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
        "data_path": "/home/abidhasan/Document/Project/WaveFreqAug_Forecasting/dataset/ETTh2.csv",
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
        "data_path": "/home/abidhasan/Document/Project/WaveFreqAug_Forecasting/dataset/national_illness.csv",
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
        "data_path": "/home/abidhasan/Document/Project/WaveFreqAug_Forecasting/dataset/weather.csv",
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
    # New dataset — aug_params below are unused by main/main.py (its grid/
    # random search builds hyperparameter combos from the MASK_RATES/LEVELS/
    # WAVELETS/LAMBDS/WINDOWS constants directly, not from this dict) and are
    # only kept here for schema consistency with the other datasets.
    "exchange_rate": {
        "data_path": "/home/abidhasan/Document/Project/WaveFreqAug_Forecasting/dataset/exchange_rate.csv",
        "data_name": "exchange_rate",
        "seq_len": 336,
        "pred_lens": [96, 192, 336, 720],
        "enc_in": 8,
        "batch_size": 32,
        "aug_types": ["Wave-Freq"],
        "aug_params": {
            pred_len: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            }
            for pred_len in [96, 192, 336, 720]
        },
    },
}
