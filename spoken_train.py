from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AutoTokenizer, BertForQuestionAnswering, get_scheduler
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
import os
from spokensquad_dataset import SpokenSQuAD

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accelerator = Accelerator(mixed_precision='no')

model_name = 'bert-base-uncased'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = BertForQuestionAnswering.from_pretrained(model_name)

spoken = SpokenSQuAD()

trainloader = DataLoader(
    spoken,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=8
)
optimizer = AdamW(model.parameters(), lr=2e-5)


model, optimizer, trainloader = accelerator.prepare(
    model, optimizer, trainloader
)

num_train_epochs = 3
num_update_steps_per_epoch = len(trainloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# train model
progress_bar = tqdm(range(num_training_steps))

output_dir = 'checkpoints/spoken_squad/' + model_name
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for epoch in range(num_train_epochs):
    model.train()
    for step, batch in enumerate(trainloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
# save the model to checkpoints directory
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
spoken.tokenizer.save_pretrained(output_dir)