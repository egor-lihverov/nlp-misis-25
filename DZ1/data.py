import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class LoanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataManager:
    def __init__(self, config):
        self.config = config
        self.cat_mapping = {
            "map_cb_person_default_on_file": {
                "N": 0,
                "Y": 1,
            },
            "map_loan_grade": {
                "A": 7,
                "B": 6,
                "C": 5,
                "D": 4,
                "E": 3,
                "F": 2,
                "G": 1,
            },
            "map_person_home_ownership": {
                "RENT": 2,
                "OWN": 4,
                "MORTGAGE": 3,
                "OTHER": 1,
            },
            "map_loan_intent": {
                "EDUCATION": 1,
                "MEDICAL": 2,
                "PERSONAL": 3,
                "VENTURE": 4,
                "DEBTCONSOLIDATION": 5,
                "HOMEIMPROVEMENT": 6,
            },
        }
        self.load_data()
        self.preprocess()
        self.split_data()

    def load_data(self):
        self.train_df = pd.read_csv(self.config["train_path"], index_col="id")
        self.test_df = pd.read_csv(self.config["test_path"], index_col="id")

    def preprocess(self):
        cat_features = self.train_df.select_dtypes("object").columns
        for feat in cat_features:
            train_map = self.cat_mapping[f"map_{feat}"]
            test_map = self.cat_mapping[f"map_{feat}"]
            self.train_df[feat] = self.train_df[feat].map(train_map)
            self.test_df[feat] = self.test_df[feat].map(test_map)

        self.X_train = self.train_df.drop("loan_status", axis=1).values
        self.y_train = self.train_df["loan_status"].values

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)

    def split_data(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_scaled,
            self.y_train,
            test_size=self.config.get("test_size", 0.2),
            random_state=42,
        )

    def get_datasets(self):
        return (
            LoanDataset(self.X_train, self.y_train),
            LoanDataset(self.X_val, self.y_val),
        )

    def get_data_loaders(self, batch_size):
        train_dataset, val_dataset = self.get_datasets()
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size),
        )
