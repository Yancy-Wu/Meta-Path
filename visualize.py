'''
    visualize.py
'''

import json
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import SimiModel
from dataset import Instance, Kpi, Alarm
from tokenizer import Tokenizer
# from utils.graph import Graph

COLOR = {
    Alarm.__name__: 'r',
    Kpi.__name__: 'y',
    Instance.__name__: 'g'
}

def main():
    ''' main '''
    # load saved
    saved = torch.load('./saved.bin', map_location=torch.device('cpu'))
    # graph: Graph = saved['graph']
    tokenizer: Tokenizer = saved['tokenizer']
    model: SimiModel = saved['model']

    # fit embeddings.
    origin_embs = model.embs(torch.LongTensor(range(0, model.embs.num_embeddings)))
    origin_embs = origin_embs.cpu().detach().numpy()
    tsne = TSNE()
    embs = tsne.fit_transform(origin_embs)

    # draw dots.
    plt.switch_backend('agg')
    for i, dot in enumerate(embs):
        if abs(dot[0]) > 20 or abs(dot[1]) > 20:
            continue
        vertex = tokenizer.decode([i])[0]
        plt.scatter(*dot, c=COLOR[vertex.category])
    plt.savefig('visual.jpg')

    # dump embs to json.
    with open('./embs.json', 'w') as out:
        for i, emb in enumerate(origin_embs):
            vertex = tokenizer.decode([i])[0]
            info = {
                'id': vertex.vid,
                'category': vertex.category,
                'emb': emb.tolist()
            }
            out.write(json.dumps(info) + '\n')

if __name__ == '__main__':
    main()
