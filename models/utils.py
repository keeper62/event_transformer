import yaml
import os
import glob

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def load_all_configs(config_dir="configs"):
    config_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    configs = {}
    for config_file in config_files:
        config_name = os.path.basename(config_file).replace(".yaml", "")
        with open(config_file, 'r') as file:
            configs[config_name] = yaml.safe_load(file)
    return configs

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params