'''
    model.py
'''
import torch
from torch.nn import Module
from torch.nn import Embedding
from utils.config import Config

class SimiModel(Module, Config):
    '''
        one line model.
        calc similarity between x1 and x2. then return.
    '''

    # embedding size config
    EMBEDDING_SIZE = 128

    # device
    DEVICE = torch.device('cpu')

    def __init__(self, id_num: int, conf=None):
        Module.__init__(self)
        Config.__init__(self, conf)
        self.embs = Embedding(id_num, self.EMBEDDING_SIZE).to(self.DEVICE)

    # pylint: disable=arguments-differ
    def forward(self, x1: torch.LongTensor, x2: torch.LongTensor):
        '''
            calc similarity.
              x1 shape: [*], * for any.
              x2 shape: [*], * for any.
        '''
        saved_size = x1.size()
        # x1, x2 shape: [*, hidden_size]
        x1 = self.embs(x1.view(-1))
        x2 = self.embs(x2.view(-1))
        return (x1 * x2).sum(-1).view(saved_size)
