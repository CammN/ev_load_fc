from pathlib import Path

class Paths:
    """ Generates Path objects for key folder directories"""


    def __init__(self, root_path:str=''):
        
        # Root
        if root_path == '':
            self.root_path = Path(__file__).resolve().parents[3]
        else:
            self.root_path = root_path

        # Data paths
        self.raw_data_path = self.root_path / "data" / "01_raw"
        self.interim_data_path = self.root_path / "data" / "02_interim"
        self.processed_data_path = self.root_path / "data" / "03_processed"

        # Model paths
        self.exp_models_path = self.root_path / "models" / "experiments"
        self.prod_models_path = self.root_path / "models" / "production"


    def get_data_paths(self):
        """Returns all data directory paths as Path objects"""
        return self.raw_data_path, self.interim_data_path, self.processed_data_path


    def get_model_paths(self):
        """Returns all model directory paths as Path objects"""
        return self.exp_models_path, self.prod_models_path