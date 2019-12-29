'''
    tokenizer.py
'''
from typing import List
from utils.graph import Vertex, Route
from skip_gram import Token

class Tokenizer():
    '''
        convert a group of graph route to id sequences.
        so make all routes like text.
          `vertices`: work like words. be used to build vocabulary
    '''

    vertices_num = 0
    category_num = 0

    def __init__(self, vertices: List[Vertex]):
        # initialize
        self.uuid_to_id = dict()
        self.category_to_id = dict()
        self.vocabulary: List[Vertex] = []

        # build vocabulary for encode and decode.
        for v in vertices:
            vid = self.uuid_to_id.get(v.uuid)
            cid = self.category_to_id.get(v.category)

            # new vertex(or called word)
            if vid is None:
                vid = self.vertices_num
                self.uuid_to_id.update({v.uuid:vid})
                self.vocabulary.append(v)
                self.vertices_num += 1

            # new category
            if cid is None:
                cid = self.category_num
                self.category_to_id.update({v.category:cid})
                self.category_num += 1

    def encode(self, routes: Route) -> List[Token]:
        '''
            tokenizer an route, re-index vertex to 0 ~ N,
            re-index vertex category to 0 ~ M.
              `routes`: graph route object.
              `return`: route vertex id and category id sequences.
        '''
        v: Vertex = None
        path: List[Token] = []
        for v in routes.vertices:
            vid = self.uuid_to_id.get(v.uuid)
            cid = self.category_to_id.get(v.category)
            path.append(Token(vid, cid))
        return path

    def decode(self, token_ids: List[int]) -> List[Vertex]:
        '''
            de-tokenizer an token sequences.
              `token_ids`: token id sequences.
              `return`: a graph vertex list.
        '''
        vertices = []
        for token_id in token_ids:
            vertex = self.vocabulary[token_id]
            vertices.append(vertex)
        return vertices

    def vocabulary_size(self):
        ''' acutally: num of vertices. '''
        return self.vertices_num

    def category_size(self):
        ''' acutally: num of categories. '''
        return self.category_num
