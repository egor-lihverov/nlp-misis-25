from itertools import product

# Основные конфигурации
configs = [
    # Базовый конфиг для SimpleNet
    {
        "experiment_name": "SimpleNetV1_32_epochs30",
        "model_class": "SimpleNet",
        "model_params": {"input_size": 11, "hidden_size": 32},
        "n_epochs": 30,
        "lr": 0.01,
        "batch_size": 128,
        "train_path": "/kaggle/input/playground-series-s4e10/train.csv",
        "test_path": "/kaggle/input/playground-series-s4e10/test.csv",
    },
    # Конфиг для SimpleNetV2
    {
        "experiment_name": "SimpleNetV2_128_epochs15",
        "model_class": "SimpleNetV2",
        "model_params": {"input_size": 11, "hidden_size": 128},
        "n_epochs": 15,
        "lr": 0.01,
        "batch_size": 128,
    },
    
    # Конфиг для SimpleNetV3
    {
        "experiment_name": "SimpleNetV3_128_epochs15",
        "model_class": "SimpleNetV3",
        "model_params": {"input_size": 11, "hidden_size": 128},
        "n_epochs": 15,
        "lr": 0.01,
        "batch_size": 128,
    },
]

# Конфиги для перебора dropout (p) в SimpleNetV4
p_range = [0.01, 0.1, 0.2, 0.5, 0.9]
for p in p_range:
    configs.append(
        {
            "experiment_name": f"SimpleNetV4_p{p}_epochs15",
            "model_class": "SimpleNetV4",
            "model_params": {"input_size": 11, "hidden_size": 128, "p": p},
            "n_epochs": 15,
            "lr": 0.01,
            "batch_size": 128,
        }
    )

# Конфиги для перебора lr и weight_decay
weight_decay_range = [0.1, 0.01, 0.001]
lr_range = [0.01, 0.05, 0.1]
for wd, lr in product(weight_decay_range, lr_range):
    configs.append(
        {
            "experiment_name": f"SimpleNetV4_wd{wd}_lr{lr}_epochs15",
            "model_class": "SimpleNetV4",
            "model_params": {"input_size": 11, "hidden_size": 128, "p": 0.01},
            "n_epochs": 15,
            "lr": lr,
            "batch_size": 128,
            "weight_decay": wd,
        }
    )
