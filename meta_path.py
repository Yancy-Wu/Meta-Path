'''
    meta_path.py
'''
import random
import tqdm
from typing import List, Tuple
from utils.graph import Vertex, Route, Graph
from utils.config import Config

class MetaPath(Config):
    '''
        meta path sampler.
    '''

    # meta paths:
    meta_paths: List[Tuple[List[type], int]] = None

    def __init__(self, graph: Graph, conf=None):
        Config.__init__(conf)
        self.graph = graph
        self.meta_paths = []

    def _parse_meta_path(self, path: List[type or tuple]) -> List[type]:
        # parse path
        if not path:
            return [[]]

        # iteraterly call myself to decode.
        res = []
        node = path[0]
        rcontext_path = self._parse_meta_path(path[1:])

        # parse tuple.
        if isinstance(node, tuple):

            # for control cmd, such as *0-3 for repeat last node 0-3 times.
            if isinstance(node[0], str):
                cmd = node[0]
                if cmd[0] == '*':
                    repeat_node = node[-1]
                    for j in range(*map(int, cmd[1:].split('-'))):
                        res += [j * [repeat_node] + rp for rp in rcontext_path]
                else:
                    raise AssertionError('syntax error')

            # for or cmd, all node in tuple will send to path seperately.
            elif isinstance(node[0], type):
                for sub_node in node:
                    res += [[sub_node] + rp for rp in rcontext_path]

        # normal node.
        elif isinstance(node, type):
            res += [[node] + rp for rp in rcontext_path]

        # unknow cmd.
        else:
            raise AssertionError('syntax error')

        # return parse result.
        return res

    def add(self, path: List[type or tuple], sample_num=10):
        '''
            add a meta path, contains node type sequences
              `sample_num`: for each head node in this meta path,
              how many graph paths we will sample
        '''
        for parsed_path in self._parse_meta_path(path):
            self.meta_paths.append((parsed_path, sample_num))

    def _sample(self, vertex: Vertex, meta_path: List[type]) -> Route:
        # res path, save edges and vertices.
        route = Route()
        route.append(None, vertex)
        if vertex.category != meta_path[0].__name__:
            return None
        # sample next recursively.
        for category in meta_path[1:]:
            adjs = [*self.graph.adj(vertex, category)]
            if not adjs:
                return None
            random_adj = random.choice(adjs)
            route.append(*random_adj)
            vertex = random_adj[1]
        return route

    def _remove_route_duplicate(self, routes: List[Route]) -> List[Route]:
        # no route found. return null.
        routes = filter(lambda route: route, routes)
        # not append same route.
        res = []
        ruuids = set()
        for route in routes:
            uuid = str(route)
            if uuid in ruuids:
                continue
            res.append(route)
            ruuids.add(uuid)
        return res

    def random_walk(self) -> List[Route]:
        '''
            return random traversal route for each graph node.
        '''
        res: List[Route] = []
        vertex: Vertex = None
        # traversal all vertex, sample it.
        count = sum([path[1] for path in self.meta_paths], 0) * len(self.graph.vertices)
        with tqdm.trange(0, count) as progress:
            for vertex in self.graph.vertices:
                for meta_path, sample_num in self.meta_paths:
                    ruotes = [self._sample(vertex, meta_path) for _ in range(0, sample_num)]
                    res += self._remove_route_duplicate(ruotes)
                    progress.update(sample_num)
        return res
