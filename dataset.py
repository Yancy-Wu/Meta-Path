'''
    dataset.py
'''
from typing import List
import json
import pandas as pd
from utils.graph import Graph, Vertex, Edge

class Instance(Vertex):
    ''' huawei instance vertex define. '''
    def __init__(self, vid: str):
        Vertex.__init__(self, vid)

class Kpi(Vertex):
    ''' huawei kpi vertex define '''
    def __init__(self, vid: str, classname: str, servicename: str, name: str):
        Vertex.__init__(self, vid)
        self.classname = classname
        self.servicename = servicename
        self.name = name

class Alarm(Vertex):
    ''' huawei alarm vertex define '''
    def __init__(self, vid: str, name: str):
        Vertex.__init__(self, vid)
        self.name = name

class InstanceToInstance(Edge):
    ''' huawei edge define between instance. '''
    def __init__(self, name: str):
        Edge.__init__(self, None)
        self.name = name

class KpiAlarmToInstance(Edge):
    ''' huawei edge define between instance and kpi, instance and alarm. '''

class KpiAlarmMutualEdge(InstanceToInstance):
    ''' huawei kpi to alarm, vice versa '''

class Dataset:
    '''
        in fact, a graph factory.
        use for generating a utils.graph object.
          `root`: dataset location.
    '''

    graph: Graph = None

    def __init__(self, root: str):
        self.root = root
        self.graph = Graph()

    def _load_sheet(self, sheet_name: str) -> pd.DataFrame:
        """ read a single sheet from excel file and do some preprocessing """
        # !!! missing values are represented as string 'nan' instead of NaN because of dtype=str
        data = pd.read_excel(self.root, sheet_name=sheet_name, dtype=str)
        # strip blank character
        for col in data:
            data[col] = data[col].apply(str.strip)
        # drop rows duplicated
        data = data.drop_duplicates()
        data = data.reset_index(drop=True)
        return data

    def _load_topo(self):
        # load instance topology to graph.
        for _, line in self._load_sheet('TOPO').iterrows():
            vh = Instance(line['首实体(*)'])
            vt = Instance(line['尾实体(*)'])
            edge = InstanceToInstance(line['关系名(*)'])
            self.graph.add_vertex(vh)
            self.graph.add_vertex(vt)
            self.graph.add_edge(vh, vt, edge)
            self.graph.add_edge(vt, vh, edge)

    def _load_kpi(self):
        # load kpi instance topology to graph.
        for _, line in self._load_sheet('KPI').iterrows():
            # NOTE: strange vid fetch approach.
            vid = line['INSTANCE_NAME'] if line['INSTANCE_NAME'] != 'nan' else line['ENTITY_NAME'].split('_')[0]
            vh = Kpi(line['Entity ID'], line['Class Name_CN'], line['Service Name_CN'], line['Name_CN'])
            vt = self.graph.vertex(vid, Instance)
            # isolate node. skip it.
            if not vt:
                continue
            edge = KpiAlarmToInstance()
            self.graph.add_vertex(vh)
            self.graph.add_edge(vh, vt, edge)
            self.graph.add_edge(vt, vh, edge)

    def _load_alarm(self):
        # load alarm instance topology to graph.
        for _, line in self._load_sheet('ALARM').iterrows():
            # NOTE: strange vid fetch approach.
            vid = line['INSTANCE_NAME'] if line['INSTANCE_NAME'] != 'nan' else line['ENTITY_NAME'].split('_')[0]
            vh = Alarm(line['ALARM ID'], line['Name_CN'])
            vt = self.graph.vertex(vid, Instance)
            # isolate node. skip it.
            if not vt:
                continue
            edge = KpiAlarmToInstance()
            self.graph.add_vertex(vh)
            self.graph.add_edge(vh, vt, edge)
            self.graph.add_edge(vt, vh, edge)

    def _load_rule(self) -> pd.DataFrame:
        # parse alarm/kpi id and type from a json-formatted string
        def _parse(s: str) -> pd.Series:
            jdata = json.loads(s)
            alarm_id = jdata.get('AlarmID', None)
            stats_id = jdata.get('StatsID', None)
            return (alarm_id, 'ALARM') if alarm_id else (stats_id, 'KPI')

        # load alarm instance topology to graph.
        for _, line in self._load_sheet('RULE').iterrows():
            hid, htype = _parse(line['事件B'])
            tid, ttype = _parse(line['事件A'])
            vh = self.graph.vertex(hid, Kpi if htype == 'KPI' else Alarm)
            vt = self.graph.vertex(tid, Kpi if ttype == 'KPI' else Alarm)
            # no such instance
            if not vt or not vh:
                continue
            edge = KpiAlarmMutualEdge(line['关系'])
            self.graph.add_edge(vh, vt, edge)
            self.graph.add_edge(vt, vh, edge)

    def build_graph(self, modules: List[str]) -> Graph:
        ''' convert source data to graph '''
        for module in modules:
            getattr(self, f'_load_{module}')()
        return self.graph
