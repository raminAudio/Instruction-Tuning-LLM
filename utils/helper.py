from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments,Trainer, AutoModelForSequenceClassification
import torch
import pandas as pd
import evaluate
from collections import Counter
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from transformers import pipeline
import numpy as np
from IPython.display import Image, display

dataset_name = 'knkarthick/dialogsum'
dataset = load_dataset(dataset_name)
split = 'test'

class Helper():
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        
    def create_prompt(self, example_indicies, target_index=1):
        prompt = ''

        start_prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary:'

        for i,index in enumerate(example_indicies):
            dialoge = dataset[split][index]['dialogue']
            summary = dataset[split][index]['summary']
            prompt += start_prompt+dialoge+end_prompt+summary

        index = target_index
        dialoge = dataset[split][index]['dialogue']
        ground_truth = dataset[split][index]['summary']
        prompt = start_prompt+dialoge+end_prompt

        return prompt, ground_truth

    def gen_output(self, prompt, original_model, fine_tuned_model):
        inputs  = self.tokenizer(prompt, return_tensors='pt')
        outputO = self.tokenizer.decode(original_model.generate(inputs['input_ids']  )[0],skip_special_tokens=True,do_sample=True, tempreture=0.5)
        outputF = self.tokenizer.decode(fine_tuned_model.generate(input_ids=inputs['input_ids'])[0],skip_special_tokens=True,do_sample=True, tempreture=0.5)

        return ' '.join(outputO.split('Dialogue:')[-1:]), ' '.join(outputF.split('Dialogue:')[-1:])

    def tokenizer_function(self, example):
        start_prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary:'
        prompt = [start_prompt+dialogue+end_prompt for dialogue in example['dialogue']]
        example['input_ids'] = self.tokenizer(prompt, padding='max_length', truncation=True, return_tensors="pt").input_ids
        example['labels'] = self.tokenizer(example['summary'], padding='max_length', truncation=True, return_tensors="pt").input_ids

        return example

    def build_dataset(self, model_name,dataset_name,minLen,maxLen):
        dataset = load_dataset(dataset_name,split='train')
        dataset = dataset.filter(lambda x: len(x['dialogue'])>minLen and len(x['dialogue'])<=maxLen)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize(sample):
            prompt = f"""
            Summarize the following conversation.
            {sample['dialogue']}

            Summary:
            """

            sample['input_ids'] = tokenizer(prompt, return_tensors='pt').input_ids.to("mps")
            sample['query'] = tokenizer.decode(sample['input_ids'][0])
            return sample 

        dataset = dataset.map(tokenize, batched=False)
        dataset.set_format(type='torch')
        dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=False, seed=42)
        return dataset_splits

    def evaluate_toxic(self, model, toxic_eval, tokenizer, dataset, num_samples):
        max_new_tokens = 100
        toxics = []
        input_texts = []
        for i, sample in enumerate(dataset):
            input_text = sample["query"]
            if i > num_samples: 
                break
            input_ids = tokenizer(input_text, return_tensors = 'pt', padding = True).input_ids
            gen_config = GenerationConfig(max_new_tokens = max_new_tokens, 
                                         top_k=0.0,top_p=1.0, do_sample=True)
            response_token_ids = model.generate(input_ids=input_ids , 
                                               generation_config = gen_config)
            generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
            toxic_score = toxic_eval.compute(predictions = [(input_text + " "+ generated_text)])
            toxics.extend(toxic_score['toxicity'])

        mean = np.mean(toxics)
        std = np.std(toxics)

        return mean, std

