import pandas as pd
import os


class DataLoader:
    def __init__(self, base_path=None):
        if base_path is None:
            self.base_path = os.path.dirname(__file__)
        else:
            self.base_path = base_path

    def load(self, filename):
        path = os.path.join(self.base_path, filename)
        return pd.read_csv(path)
