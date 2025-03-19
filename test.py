from models.utils import count_parameters, load_config
from models import LogTokenizer, Transformer
from dataset_class.bgl_dataset import BGLDataset
from torchinfo import summary
import torch

config = load_config("configs/most_intense_config.yaml")['base_config']

tokenizer = LogTokenizer()
    
# Load dataset
dataset = BGLDataset(path=config['dataset']['path'], columns=config['dataset']['columns'], transform=tokenizer.transform(), data_column="Content")
dataset.construct_steps(config['model']['prediction_steps'], config['model']['context_length'])
    
tokenizer.train_template_miner(dataset.data)
config['model']['vocab_size'] = tokenizer.get_vocab_size()
    
# Load Transformer model
model = Transformer(config)

summary(model, (config['training']['batch_size'], config['model']['context_length']), col_width=32, dtypes=['torch.IntTensor'], verbose=2, col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
    row_settings=["var_names"])