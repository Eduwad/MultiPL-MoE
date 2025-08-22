from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/root/work/filestorage/xinglehao/moe/Megatron-LM/models/qwen1.5-1.8B")
model = AutoModel.from_pretrained("/root/work/filestorage/xinglehao/moe/Megatron-LM/models/qwen1.5-1.8B")

print(tokenizer.special_tokens_map)
special_tokens_dict = {"additional_special_tokens": ["<fim_predix>", "<fim_middle>", "<fim_suffix>", "<fim_pad>"]}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict,replace_additional_special_tokens = False)
print("We have added", num_added_toks, "tokens")
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer)) # 

model.save_pretrained('/root/work/filestorage/xinglehao/moe/Megatron-LM/models/qwen1.5-1.8B-fim', safe_serialization = False)
tokenizer.save_pretrained('/root/work/filestorage/xinglehao/moe/Megatron-LM/models/qwen1.5-1.8B-fim')
