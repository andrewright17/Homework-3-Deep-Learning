from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_dataset
from transformers import AutoTokenizer, BertForQuestionAnswering
from accelerate import Accelerator

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accelerator = Accelerator(mixed_precision='no')

model_name = 'bert-base-uncased'

# Load the tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained(model_name)

model = BertForQuestionAnswering.from_pretrained(model_name)

print(accelerator.device)
