import os
import logging
import sys
sys.path.insert(0, '../')
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert_2.modeling import BertForNextSentencePrediction
from pytorch_pretrained_bert_2.tokenization import BertTokenizer
# from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear, SCHEDULES
from fastprogress import master_bar, progress_bar
import random
from feature_utils import convert_sentence_pair

# constants
SEED = 42
PYTORCH_PRETRAINED_BERT_CACHE = "/Users/julian/hackzurich/temp"

# Local network environment settings
os.environ["http_proxy"] = "127.0.0.1:11233"
os.environ["https_proxy"] = "127.0.0.1:11233"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger("bert")

class BERT_Recommender():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # %%

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if self.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(SEED)

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True,
            cache_dir='PYTORCH_PRETRAINED_BERT_CACHE')

        ## Model
        # %%
        print(PYTORCH_PRETRAINED_BERT_CACHE)
        self.model = BertForNextSentencePrediction.from_pretrained(
            "bert-base-uncased",
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE
        ).to(self.device)

        # correct_pairs = convert_sentence_pair(df_full.title.tolist(), df_full.desc.tolist(), max_seq_length=200,
        #                                       tokenizer=self.tokenizer)

    def get_recommendations(self, df_full):
        ## Prediction SEED
        idx = 102
        print('INPUT', df_full.iloc[idx].title, df_full.iloc[idx].desc)

        sentence_pairs = convert_sentence_pair(
            [df_full.iloc[idx]["title"]] * df_full.shape[0],
            df_full.desc.tolist(), max_seq_length=200, tokenizer=self.tokenizer)
        # %%
        BATCH_SIZE = 128
        logger.info("***** Running evaluation *****")
        all_input_ids = torch.tensor([f.input_ids for f in sentence_pairs], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in sentence_pairs], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in sentence_pairs], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=BATCH_SIZE)

        # logger.info("  Num examples = %d", len(correct_pairs))
        logger.info("  Batch size = %d", BATCH_SIZE)

        self.model.eval()

        res = []

        mb = progress_bar(eval_dataloader)
        for input_ids, input_mask, segment_ids in mb:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                res.append(nn.functional.softmax(
                    self.model(input_ids, segment_ids, input_mask), dim=1
                )[:, 0].detach().cpu().numpy())

        res = np.concatenate(res)
        # %%
        # _ = plt.hist(res, bins=100)
        # %%
        all_sorted_matches = np.argsort(res)[::-1]
        return all_sorted_matches







