'''
    skip_gram.py
'''
from typing import List, Dict
import random
import torch
from utils.config import Config

class Token():
    '''
        token with type define.
    '''

    def __init__(self, token_id: int, category_id: int):
        self.token_id = token_id
        self.category_id = category_id

class SkipGramWithType(Config):
    '''
        skip gram with type implementation.
          `corpus`: a list of context which consisted of token sequences.
    '''

    # skip gram half window size
    HALF_WINDOW_SIZE = 3

    # negative sample num
    NEGATIVE_SAMPLE_NUM = 5

    # table size, suitable for 10 times vocabulary size.
    TABLE_SIZE = 5000

    # neg token size: [category_num, table_size]
    neg_tokens: List[List[int]] = None
    corpus: List[List[Token]] = None

    def __init__(self, corpus: List[List[Token]], category_num: int, conf=None):
        Config.__init__(self, conf)
        self.corpus = corpus
        self.neg_tokens = []

        # statistical info initial.
        total_token_nums = category_num * [0]
        token_nums = []
        token_weights = []
        for i in range(0, category_num):
            token_nums.append(dict())
            token_weights.append([])
            self.neg_tokens.append(self.TABLE_SIZE * [0])

        # record token repeat num
        for context in corpus:
            for token in context:
                token_num = token_nums[token.category_id]
                token_num.setdefault(token.token_id, 0)
                token_num[token.token_id] += 1
                total_token_nums[token.category_id] += 1

        # normalize.
        total_token_nums = [num ** 0.75 for num in total_token_nums]
        for i, token_num in enumerate(token_nums):
            # skip-gram divide point.
            segment = 0
            for token_id, num in token_num.items():
                segment += (num ** 0.75) / total_token_nums[i]
                token_weights[i].append((token_id, segment))

        # generate neg table.
        for token_weight, neg_token in zip(token_weights, self.neg_tokens):
            step = token_weight[-1][1] / self.TABLE_SIZE
            weight_index = 0
            for j in range(0, self.TABLE_SIZE):
                cur_weight = step * j
                # push weight_index if its weight fall behind.
                while token_weight[weight_index][1] < cur_weight:
                    weight_index += 1
                # save cur neg token_id
                neg_token[j] = token_weight[weight_index][0]

    def _neg_sample(self, token: Token, exclude: List[Token]) -> List[int]:
        ''' sample negative for token, excluded tokens will not be sampled. '''
        exclude_token_ids = [t.token_id for t in exclude]
        neg_token_ids = []

        # remove excluded tokens.
        for nt in random.sample(self.neg_tokens[token.category_id], self.NEGATIVE_SAMPLE_NUM):
            if not nt in exclude_token_ids:
                neg_token_ids.append(nt)

        # len is self.NEGATIVE_SAMPLE_NUM for most most most situation.
        return neg_token_ids

    def generate_examples(self) -> Dict[str, torch.LongTensor]:
        '''
            using skip gram with type to convert corpus to
            2-classification tensor dict, using Negative Sampling.
        '''
        examples = []
        for context in self.corpus:
            for i, token in enumerate(context):
                window = context[max(i - self.HALF_WINDOW_SIZE, 0): i + self.HALF_WINDOW_SIZE]
                for ctoken in window:
                    # skip myself
                    if ctoken.token_id == token.token_id:
                        continue

                    # append this positive example.
                    examples.append((token.token_id, ctoken.token_id, 1))

                    # sample negative, exclude current token and center token.
                    for neg_token_id in self._neg_sample(ctoken, [ctoken, token]):
                        examples.append((token.token_id, neg_token_id, -1))

        return {
            'x1': torch.LongTensor([x[0] for x in examples]),
            'x2': torch.LongTensor([x[1] for x in examples]),
            'y': torch.LongTensor([x[2] for x in examples]),
        }
