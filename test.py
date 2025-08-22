# import json
# from transformers import AutoTokenizer

# file_path = '/root/work/filestorage/xinglehao/moe/dataset/code_select/pascal.json'
# # # 读取文件
# texts = []
# with open(file_path, 'r', encoding='utf-8') as f:
#     for l in f:
#         texts.append(json.loads(l))

# tokenizer = AutoTokenizer.from_pretrained("/root/work/filestorage/xinglehao/moe/Megatron-LM/models/qwen1.5-1.8B")

# out = tokenizer(texts[0][0]['content'])["input_ids"]

# print(out[0], out[-1])


import json
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

import sys
sys.setrecursionlimit(10000)  # 设置最大递归深度为10000

# 辅助函数：递归地处理文本分割
def split_text(tokens):
    if len(tokens) > 1024:
        first_part_tokens = tokens[:1024]
        second_part_tokens = tokens[1024:]
        first_part_text = tokenizer.decode(first_part_tokens)
        return [first_part_text] + split_text(second_part_tokens)
    else:
        return [tokenizer.decode(tokens)]

# 处理每个数据条目
def process_item(json_l):
    item = json.loads(json_l)
    content = item['content']
    tokens = tokenizer.encode(content)
    
    # 如果 token 长度超过 1024，分割文本
    if len(tokens) > 1024:
        split_parts = split_text(tokens)
        return int(len(split))
    else:
        return int(len(tokens))

# 处理数据：并行化任务
def process_data_parallel(data, num_processes):
    # 使用多进程池并行处理数据
    with Pool(processes=num_processes) as pool:
        # 使用 map 函数并行处理每个数据条目
        result = list(tqdm(pool.imap(process_item, data), total=len(data), desc="Processing items"))
        
    # 扁平化结果列表
    flat_result = sum(result)
    print(flat_result)
    return flat_result



tokenizer = AutoTokenizer.from_pretrained("/root/work/filestorage/xinglehao/moe/Megatron-LM/models/qwen1.5-1.8B")
# root_folder = "/root/work/filestorage/xinglehao/moe/dataset/code_kuo"
# for subfolder_name in os.listdir(root_folder):
# file_path = os.path.join(root_folder, subfolder_name)
# print(file_path)

texts = []
fim = open("/root/work/filestorage/xinglehao/moe/dataset/rust+go+pa+text+math_1.json", 'r', encoding='utf-8')
num_processes = cpu_count()

processed_data = process_data_parallel(fim, num_processes)

    # output_file_path = os.path.join("/root/work/filestorage/xinglehao/moe/dataset/code_re", subfolder_name)

    # with open(output_file_path, 'w', encoding='utf-8') as f:
    #     json.dump(processed_data, f, ensure_ascii=False)

    # print(f"Processed data has been saved to {output_file_path}")


    #     lines = f.readlines()
    # for line in lines:
    #     text = json.loads(line)
    #     texts.append({"content":text['content']})
    
    # for i, d in enumerate(data):
    #     processed_data = process_data_parallel(d, num_processes)
    #     name = os.path.join(subfolder_name.split(".")[0],'f{i}.json')
    #     output_file_path = os.path.join("/root/work/filestorage/xinglehao/moe/dataset/code_re", name)

    #     with open(output_file_path, 'w', encoding='utf-8') as f:
    #         json.dump(processed_data, f, ensure_ascii=False)

    #     print(f"Processed data has been saved to {output_file_path}")


# import json
# from transformers import AutoTokenizer


# tokenizer = AutoTokenizer.from_pretrained("/root/work/filestorage/xinglehao/moe/Megatron-LM/models/qwen1.5-1.8B")

# # 读取原始 JSON 文件
# input_file_path = "/root/work/filestorage/xinglehao/moe/dataset/code_select/pascal.json"

# # 加载原始数据
# with open(input_file_path, 'r', encoding='utf-8') as f:
#     for l in f:
#         data=json.loads(l)
# print(data[0])
        
# new_data = []

# def split_text(tokens):
#     if len(tokens) > 1024:
#         first_part_tokens = tokens[:1024]
#         second_part_tokens = tokens[1024:]
#         first_part_text = tokenizer.decode(first_part_tokens)
#         return [first_part_text] + split_text(second_part_tokens)
#     else:
#         return [tokenizer.decode(tokens)]

# # 处理数据
# content = data[0]['content']
# tokens = tokenizer.encode(content)
# if len(tokens) > 1024:
#     split_parts = split_text(tokens)
#     for part in split_parts:
#         new_data.append({"content": part})
# else:
#     new_data.append({"content": content})

# print(new_data)
