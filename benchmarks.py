import torch
from transformers import AutoTokenizer, BertForQuestionAnswering, default_data_collator
from datasets import load_dataset
import evaluate
import collections
import numpy as np
from spokensquad_dataset import SpokenSQuAD
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator

def preprocess_squad_valid_examples(examples):
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = 384
    stride = 128
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k,o in enumerate(offset)
        ]
        inputs["example_id"] = example_ids
        return inputs

def compute_f1(start_logits, end_logits, features, examples):
    n_best = 20
    max_answer_length = 30
    predicted_answers = []
    metric = evaluate.load("squad")
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            start_indexes = np.argsort(start_logit)[-1: -n_best -1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best -1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][0]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text":""})
    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references= theoretical_answers)

def main():
    squad = load_dataset("squad")
    squad_valid = squad["validation"].map(
        preprocess_squad_valid_examples,
        batched = True,
        remove_columns = squad["validation"].column_names,
    )
    squad_model_checkpoint_path = 'checkpoints/squad/'
    spoken_model_checkpoint_path = 'checkpoints/spoken_squad/'

    squad_valid_set = squad_valid.remove_columns(["example_id", "offset_mapping"])
    squad_valid_set.set_format("torch")
    squad_bert = BertForQuestionAnswering.from_pretrained(squad_model_checkpoint_path+'bert-based-uncased')
    spoken_bert = BertForQuestionAnswering.from_pretrained(spoken_model_checkpoint_path+'bert-base-uncased')

    squad_eval_loader = DataLoader(
        squad_valid_set,
        collate_fn=default_data_collator,
        batch_size=8,
    )
    accelerator = Accelerator(mixed_precision='no')
    squad_bert, spoken_bert, squad_eval_loader = accelerator.prepare(squad_bert, spoken_bert, squad_eval_loader)

    squad_bert.eval()
    spoken_bert.eval()

    start_logits = []
    end_logits = []
    accelerator.print("Evaluation of squad_bert:")
    for batch in tqdm(squad_eval_loader):
        with torch.no_grad():
            outputs = squad_bert(**batch)
        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(squad_valid_set)]
    end_logits = end_logits[: len(squad_valid_set)]
    metrics = compute_f1(
        start_logits=start_logits, end_logits=end_logits, features=squad_valid_set, examples=squad["validation"]
    )
    print(metrics)

    start_logits = []
    end_logits = []
    accelerator.print("Evaluation of spoken_bert:")
    for batch in tqdm(squad_eval_loader):
        with torch.no_grad():
            outputs = spoken_bert(**batch)
        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(squad_valid_set)]
    end_logits = end_logits[: len(squad_valid_set)]
    metrics = compute_f1(
        start_logits=start_logits, end_logits=end_logits, features=squad_valid_set, examples=squad["validation"]
    )
    print(metrics)