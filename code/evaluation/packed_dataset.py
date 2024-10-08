import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from datasets import Dataset as Dataset_hf

import math
import json

class EvalDataset(Dataset):
    def __init__(self, args, task_name, block_size, stride, tokenizer, cluster, file_num=-1, dtype="auto", vocab_size=None):
        self.args = args
        self.task_name = task_name
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.file_num = file_num
        self.data = None
        self.stride = stride
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self.cluster = cluster

        self.ids = []
        self.token_lens = []
        self.char_number_list = []

        self._prepare()
        self.prev_end_loc = 0
        self.seq_len = len(self.data)
        self.begin_loc = 0


    def _prepare(self):
        self._curr_idx = 0
        self._arr = []
        # self._raw_dataset = load_dataset(
        #     "hkust-nlp/llm-compression",
        #     self.task_name,
        #     split='test[:]' if self.file_num == -1 else f"test[:{self.file_num}]",
        #     cache_dir=self.args.cache_dir,

        # )
        self._raw_dataset = []
        count = 0
        with open(f"/university/Clustering/Pretrain-Data-Selection-Clustering/documents_clean_with_cluster_30M/cluster_{self.cluster}.json", "r") as f:
            for line in f:
                data = json.loads(line)
                if len(data["raw_content"]) > 0:
                    self.ids.append(data["doc_id"])
                    # self.token_lens.append()
                    # count += 1
                    # print(len(data["raw_content"]))
                    self._raw_dataset.append(data)
                    # if count > 20:
                    #     break
        self.raw_dataset =  Dataset_hf.from_list(self._raw_dataset)
        # self.raw_dataset = self._raw_dataset.filter(lambda example: len(example['content']) > 0)
        self.character_num = 0
        for i in range(len(self.raw_dataset)):
            self.character_num += len(self.raw_dataset[i]['raw_content'])
            self.char_number_list.append(len(self.raw_dataset[i]['raw_content']))
        self.data = self.raw_dataset.map(
            lambda example: {"encoding": np.array(self.tokenizer.encode(example['raw_content']), dtype=self._dtype)}, num_proc=8)
        for i in range(len(self.data)):
            self.token_lens.append(len(self.data[i]['encoding']))
        self.data = np.concatenate([a['encoding'] for a in self.data], axis=0)

    def __len__(self):
        return math.floor((len(self.data)-self.block_size)/self.stride+1)

    def __getitem__(self,item):
        end_loc = min(self.begin_loc+self.block_size, self.seq_len)
        trg_len = end_loc - self.prev_end_loc
        input_ids = self.data[self.begin_loc:end_loc]
        attention_mask = np.ones((len(input_ids),), dtype=bool)
        attention_mask[:-trg_len] = False
        self.prev_end_loc = end_loc
        self.begin_loc = self.begin_loc + self.stride
        return torch.tensor(input_ids), torch.tensor(attention_mask, dtype=bool)
