'''
    trainer.py
'''
import torch
from pytorch_transformers import AdamW
from dataset import Dataset, Instance, Kpi, Alarm
from meta_path import MetaPath
from model import SimiModel
from skip_gram import SkipGramWithType
from tokenizer import Tokenizer
from utils.dataloader import DictDataLoader
from utils import deep_apply_dict

def main():
    ''' main '''
    # build graph
    graph = Dataset('先验知识表_v0.2.xlsx').build_graph(['topo', 'alarm', 'kpi'])

    # pre-define meta path
    meta_path = MetaPath(graph)
    meta_path.add([(Alarm, Kpi), ('*1-3', Instance), (Kpi, Alarm)], 20)

    # generate route using random walk
    routes = meta_path.random_walk()

    # build vocabulary, encode route.
    tokenizer = Tokenizer(graph.vertices)
    corpus = [tokenizer.encode(route) for route in routes]

    # generating examples using skip gram with type.
    train_tensors = SkipGramWithType(corpus, tokenizer.category_size(), {
        'HALF_WINDOW_SIZE': 3,
        'NEGATIVE_SAMPLE_NUM': 5,
        'TABLE_SIZE': 5000
    }).generate_examples()

    # create model.
    model = SimiModel(tokenizer.vocabulary_size(), {
        'EMBEDDING_SIZE': 128,
        'DEVICE': torch.device('cuda:0')
    })

    deep_apply_dict(train_tensors, lambda _, v: v.to(model.DEVICE))

    # create dataloader.
    dataloader = DictDataLoader(train_tensors, {
        'batch_size': 20000
    })

    # optimizer
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    # train start.
    for round_num in range(0, 200):
        for step, batch in enumerate(dataloader):
            # in fact, y is 1 or -1
            model.train()
            y = batch.pop('y')
            res = model.forward(**batch)
            loss = -torch.log(torch.sigmoid((y * res))).mean(-1)
            print(f'[round: {round_num}]: {step}/{len(dataloader)} end. loss: {loss}')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    torch.save({
        'model': model,
        'tokenizer': tokenizer,
        'graph': graph
    }, './saved2.bin')

if __name__ == '__main__':
    main()
