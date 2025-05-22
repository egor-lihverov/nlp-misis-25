import torch
from ..DZ1.data import DataManager
from ..DZ1.models import SimpleNetV1, SimpleNetV2, SimpleNetV3, SimpleNetV4
from ..DZ1.trainer import Trainer
from ..DZ1.config import configs

def get_model_class(config):
    model_name = config['model_class']
    model_map = {
        'SimpleNet': SimpleNetV1,
    }
    return model_map[model_name]

def run_experiment(config):
    dm = DataManager(config)
    train_loader, val_loader = dm.get_data_loaders(config['batch_size'])
    
    model_class = get_model_class(config)
    model = model_class(**config['model_params'])
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    trainer.train()

if __name__ == '__main__':
    for config in configs:
        print(f"Running experiment: {config['experiment_name']}")
        config["train_path"] = "/home/egorl/ML_misis/24MISISAI/mine/NLP/DZ1/train.csv"
        config["test_path"] = "/home/egorl/ML_misis/24MISISAI/mine/NLP/DZ1/test.csv"
        run_experiment(config)