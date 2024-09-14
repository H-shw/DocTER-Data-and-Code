import os
import sys
from typing import List
import transformers
import fire
import torch
import random
import json
import math
import re
import pdb
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset

import os
os.environ["WANDB_DISABLED"] = "true"

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    debug: str = "false",
    data_path: str = "",
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 4,
    num_epochs: int = 30,
    learning_rate: float = 3e-5,
    cutoff_len: int = 512,
    val_set_size: int = 100,
    seed: str = "42",
    ratio: str = "1",
    data_size: str = "all",
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = True,
    resume_from_checkpoint: str = "",
):
    seed = int(seed)
    ratio = float(ratio)

    fn = data_path
    if '/' in data_path:
        fn = data_path.split('/')[-1]
    if '.json' in data_path:
        fn = fn.split('.json')[0]
    model_name = base_model.split('pretrained_models/')[-1].replace('/', '')
    output_dir = f"./output/{model_name}-FT-{num_epochs}-{learning_rate}-{seed}-{fn}/"
    print(
        f"Training model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"output_dir: {output_dir}\n"
        f"seed: {seed}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size


    device_map = "auto" 
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        print('ddp!')
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
   
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.unk_token_id  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    
    class DocTERDataset(Dataset):
        def __init__(self, data, dataset_size, reverse=False):       
            assert isinstance(data,list), f"Expected data object of type list got {type(data)}"
            if reverse:
                self.data = data[-dataset_size:]
            else:
                self.data = data[:dataset_size]
            
        
        def __getitem__(self,idx):
            return tokenize(self.data[idx]["src"])
        
        def __len__(self) -> int:
            return len(self.data)
        
    def tokenize(prompt, add_eos_token=True):
        
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=True,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        
        return result

    def load_dataset(data_path, data_size, val_set_size=0, file_type='json', data_field='data'):
        assert file_type == 'json', 'Invalid file type'
        if val_set_size > 0:
            with open(data_path, 'r') as f2:
                data = json.load(f2)["data"]
                # random.shuffle(data)
                if data_size == 'all':
                   data_size =  len(data)
                else:
                    data_size =  int(data_size)
                train_dataset = DocTERDataset(data,data_size)
                val_dataset = DocTERDataset(data,val_set_size,reverse=True)
            return train_dataset,val_dataset
        else:
            with open(data_path) as f:
                data=json.load(f)
                data = data[data_field]
                train_dataset = DocTERDataset(data,len(data))
            return train_dataset

    if val_set_size > 0:
        train_data, val_data = load_dataset(data_path=data_path, val_set_size=val_set_size, data_field="data", data_size = data_size)
    else:
        train_data = load_dataset(data_path=data_path, val_set_size=val_set_size ,data_field="data", data_size=data_size)
        val_data = None


    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer = tokenizer,
        args=transformers.TrainingArguments(
            report_to=None,
            seed = seed,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size = micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            max_grad_norm=5.0,
            bf16=True,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=2,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train()
    model.save_pretrained(output_dir)




if __name__ == "__main__":
    fire.Fire(train)