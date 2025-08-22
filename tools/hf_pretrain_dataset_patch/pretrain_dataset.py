# Copyright (c) 2023 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import io
import copy
import json
import torch
from megatron.training import get_args
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from transformers import AutoTokenizer

class HFPretrainDataset(torch.utils.data.Dataset):
    """A class for processing a LLama text dataset"""

    def __init__(self, path, split='train'):
        self.samples = load_from_disk(path).shuffle(seed=22)

        print('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample)

    def gpt_convert_example_to_feature(self, sample):
        """
        Convert a single sample containing input_id, label and loss_mask into a format suitable for GPT training.
        """
        input_ids, labels, lang_mask = sample['input_ids'], sample['labels'], sample['lang_mask']
        train_sample = {
            'input_ids': np.array(input_ids),
            'labels': np.array(labels),
            'lang_mask': np.array(lang_mask),
        }

        return train_sample