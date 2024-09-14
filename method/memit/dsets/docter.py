import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *


class DocTERDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        multi: bool = True,
        size: typing.Optional[int] = None,
        *args,
        **kwargs,
    ):
        # data_dir = Path(data_dir)
        
        data_dir = ''
        cf_loc = data_dir + 'dkee_sub.json'
        print(cf_loc)

        self.data = []
        cnt = 0 
        with open(cf_loc, "r") as f:
            tmp_data = json.load(f)["data"]
            sub_set = set({})
            for item in tmp_data:
                tmp_dict = {}
                tmp_dict["requested_rewrite"] = {}
                tmp_dict["requested_rewrite"]["prompt"] = item["prompt"]
                tmp_dict["requested_rewrite"]["subject"] = item["subject"]
                tmp_dict["requested_rewrite"]["target_new"] = {}
                tmp_dict["requested_rewrite"]["target_new"]["str"] = item["altered_target"]
                if tmp_dict["requested_rewrite"]["subject"] not in sub_set:
                    sub_set.add(tmp_dict["requested_rewrite"]["subject"])
                else:
                    continue
                
                tmp_dict["case_id"] = cnt
                cnt += 1
                self.data.append(tmp_dict)
            
        if size is not None:
            self.data = self.data[:size]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def get_len(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

