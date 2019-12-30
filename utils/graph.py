'''
    graph.py
'''
from typing import List, Tuple
from uuid import uuid1

def uuid(classtype: type, uid: str = None):
    ''' myself uuid generater '''
    return classtype.__name__ + '|' + (uid if uid else str(uuid1()))

class Vertex:
    '''
        vertex define.
    '''
    def __init__(self, vid=None):
        self.uuid = uuid(self.__class__, vid)

    @property
    def category(self):
        ''' return vertex category '''
        return self.__class__.__name__

    @property
    def vid(self):
        ''' return vertex vid [NOTE: not uuid] '''
        return self.uuid.split('|')[1]

class Edge:
    '''
        edge define.
    '''
    def __init__(self, eid=None):
        self.uuid = uuid(self.__class__, eid)

    @property
    def category(self):
        ''' return edge category '''
        return self.__class__.__name__

    @property
    def mid(self):
        ''' return edge mid [NOTE: not uuid] '''
        return self.uuid.split('|')[1]

class Route:
    '''
        an route in graph
    '''
    def __init__(self):
        self.path = []

    @property
    def vertices(self):
        ''' return route vertex sequences '''
        return [x[1] for x in self.path]

    def append(self, edge: Edge, next_vertex: Vertex):
        ''' append new vertex to current route '''
        self.path.append((edge, next_vertex))

    def __str__(self):
        ''' convert route to str sequence '''
        res = ''
        for _, vertex in self.path:
            res += vertex.uuid
        return res

class Graph:
    '''
        graph struct define.
        contain vertices and edges.
    '''

    def __init__(self):
        self.vertices = []
        self.edges = []
        self.uuids_map = dict()

    def vertex(self, vid: str, category: type) -> Vertex:
        '''
            return vertex according to vid, None if not found
        '''
        vi = self.uuids_map.get(uuid(category, vid), None)
        return self.vertices[vi] if vi else None

    def adj(self, vertex: Vertex, category: type) -> List[Tuple[Edge, Vertex]]:
        '''
            return adjcents vertices for specific vertex whose category is
            same as given category.
        '''
        vid = self.uuids_map[vertex.uuid]
        adjs = [i for i in range(0, len(self.vertices)) if self.edges[vid][i]]
        # reserve nodes whose class name equal to category class name
        adjs = [*filter(lambda i: self.vertices[i].category == category.__name__, adjs)]
        return [(self.edges[vid][i], self.vertices[i]) for i in adjs]

    def add_vertex(self, vertex: Vertex):
        '''
            add new vertex, if new vertex uuid has existed, skip it.
        '''
        for exist_vertex in self.vertices:
            if exist_vertex.uuid == vertex.uuid:
                return

        # add vertex, update map dict.
        self.vertices.append(vertex)
        self.uuids_map.update({vertex.uuid:len(self.vertices) - 1})

        # expand edge array.
        self.edges.append((len(self.vertices) - 1) * [None])
        for edge_row in self.edges:
            edge_row.append(None)

    def add_edge(self, vh: Vertex, vt: Vertex, edge: Edge):
        '''
            add new edge, if vertex not exist, raise error.
        '''
        vi1 = self.uuids_map[vh.uuid]
        vi2 = self.uuids_map[vt.uuid]
        self.edges[vi1][vi2] = edge
