import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import numpy as np

class SpokenSQuAD(Dataset):
    def __init__(self, train = True, unprocessed = False, max_length = 384, stride = 128, model_checkpoint ='bert-base-uncased'):
        super(SpokenSQuAD, self).__init__()
        self.train = train
        self.unprocessed = unprocessed
        if self.train:
            self.data_path = 'spoken_train-v1.1.json'
        else:
            self.data_path = 'spoken_test-v1.1.json'
        self.max_length = max_length
        self.stride = stride
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        contexts, questions, answers = self.read_data()
        self.examples = {'context': contexts, 'question': questions, 'answer': answers}
        self.encodings = self.preprocess_examples()
        print(len(self.encodings['input_ids']))

        def __getitem__(self, idx):
            if self.unprocessed == False:
                return {key: val[idx] for  key,val in self.encodings.items()}
            else:
                return {key: val[idx] for  key,val in self.examples.items()}
        
        def __len__(self):
            return len(self.encodings['input_ids'])

        def read_data(self):
            with open(self.data_path) as f:
                data = json.load(f)['data']
                contexts = []
                questions = []
                answers = []
                ids = []

                for title in data:
                    for paragraph in title['parageraphs']:
                        context = paragraph['context']
                        for qas in paragraph['qas']:
                            question = qas['question']
                            answer_text = qas['answers'][0]['text']
                            answer_start = qas['answers'][0]['answer_start']
                            id = qas['id']
                            contexts.append(context)
                            questions.append(question)
                            if self.train == True:
                                answers.append({'text': answer_text, 'answer_start': answer_start})
                            else:
                                answer_starts = []
                                texts = []
                                for answer in qas['answers']:
                                    texts.append(answer['text'])
                                    answer_starts.append(answer['answer_start'])
                                answers.append({'text': answer_text, 'answer_start': answer_start})
                            
            return contexts, questions, answers
        
        def preprocess_examples(self):
            questions = [q.strip() for q in self.examples["question"]]
            inputs = self.tokenizer(
                questions,
                self.examples["context"],
                max_length=self.max_length,
                truncation="only_second",
                stride=self.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
            if self.train == True:
                offset_mapping = inputs.pop("offset_mapping")
                sample_map = inputs.pop("overflow_to_sample_mapping")
                answers = self.examples["answer"]
                start_positions = []
                end_positions = []
                
                for i, offset in enumerate(offset_mapping):
                    sample_idx = sample_map[i]
                    answer = answers[sample_idx]
                    start_char = answer["answer_start"]
                    end_char = answer["answer_start"] + len(answer["text"])
                    sequence_ids = inputs.sequence_ids(i)

                    print()
                    # Find the start and end of the context
                    idx = 0
                    while sequence_ids[idx] != 1:
                        idx += 1
                    context_start = idx
                    while sequence_ids[idx] == 1:
                        idx += 1
                    context_end = idx - 1

                    # If the answer is not fully inside the context, label is (0, 0)
                    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                        start_positions.append(0)
                        end_positions.append(0)
                    else:
                        # Otherwise it's the start and end token positions
                        idx = context_start
                        while idx <= context_end and offset[idx][0] <= start_char:
                            idx += 1
                        start_positions.append(idx - 1)

                        idx = context_end
                        while idx >= context_start and offset[idx][1] >= end_char:
                            idx -= 1
                        end_positions.append(idx + 1)
                
                inputs["start_positions"] = start_positions
                inputs["end_positions"] = end_positions
                return inputs 
            
            else:
                sample_map = inputs.pop("overflow_to_sample_mapping")
                example_ids = []

                for i in range(len(inputs["input_ids"])):
                    sample_idx = sample_map[i]
                    example_ids.append(self.examples["id"][sample_idx])

                    sequence_ids = inputs.sequence_ids(i)
                    offset = inputs["offset_mapping"][i]
                    inputs["offset_mapping"][i] = [
                        o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
                    ]

            inputs["example_id"] = example_ids
            return inputs