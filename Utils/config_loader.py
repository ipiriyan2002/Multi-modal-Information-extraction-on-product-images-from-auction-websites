from Utils.utils import load_config_file
import os, yaml


class ConfigLoader:
    """
    Loads the base configs and the given configs into one dict overwriting base setting from given config file
    """
    def __init__(self, config,base_model="_base_model_config.yaml", base_dataset="_base_dataset_paths.yaml"):
        file_ext = config.split(".")[-1]
        assert file_ext == "yaml", f"Expected yaml file got {file_ext}"
        
        del file_ext
        
        #Loading the config file into a dict
        self.config_dict = load_config_file(config)
        self.base_model_dict = load_config_file(base_model)
        self.base_dataset_dict = load_config_file(base_dataset)
        #Get the final dict
        self.final_dict = self.genFinalDict()
    
    def genFinalDict(self):
        """
        generate the final dict by overwriting base model settings with the given config file
        """
        #Add the base paths
        final_dict = {k:v for k,v in self.base_dataset_dict.items()}
        
        #Add the base settings
        for k,v in self.base_model_dict.items():
            final_dict[k] = v
        
        #Overwrite the base settings
        for k,v in self.config_dict.items():
            final_dict[k] = v
        
        return final_dict
    
    def get(self, key):
        """
        Get the value at key from the final dict
        """
        return self.final_dict[key]
    
    def getDict(self):
        """
        Get the final dict
        """
        return self.final_dict
    
    def setDict(self, dict_):
        self.final_dict = dict_
    
    def setValue(self, key, value):
        
        self.final_dict[key] = value

    def writeFinalConfig(self, save_dir):
        with open(save_dir, 'w') as f:
            _ = yaml.dump(self.final_dict, f)
