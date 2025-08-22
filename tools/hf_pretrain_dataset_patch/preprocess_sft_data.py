from  datasets import load_dataset
from transformers import AutoTokenizer

model_path = "/root/work/huangxin/nanda/models/Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset = load_dataset("json",  data_files="/root/work/huangxin/nanda/QAlign-master/gsm8kinstruct_question_translation.json")['train']

print(dataset)
seqlen = 1024

def preprocess(samples):
    inputs = [a + b  for  a, b in zip(samples['instruction'],  samples['input'])]
    outputs = samples['output']
    input_ids = tokenizer(inputs,  add_special_tokens=False)['input_ids']
    output_ids = tokenizer(outputs,  add_special_tokens=False)['input_ids']
    ids, loss_masks =  [],  []
    for a, b in  zip(input_ids,  output_ids):
        sent_ids = a  + b  + [tokenizer.eos_token_id]
        loss_mask = [0] * len(a) + [1] * (len(b) + 1)
        if len(sent_ids) < seqlen:
            sent_ids += [tokenizer.pad_token_id] * (seqlen - len(sent_ids))
            loss_mask += [0] * (seqlen - len(loss_mask))
            assert len(sent_ids) == len(loss_mask)
        elif len(sent_ids) > seqlen:
            print(len(sent_ids))
            sent_ids =  sent_ids[:seqlen]
            loss_mask  = sent_ids[:seqlen]
        ids.append(sent_ids)
        loss_masks.append(loss_mask)
    return {"input_ids": ids, "loss_mask": loss_masks, "labels": ids}


dataset.map(preprocess, batched=True, num_proc=16)

print(dataset)

dataset.save_to_disk("/root/work/huangxin/nanda/data/gsm8btrans")