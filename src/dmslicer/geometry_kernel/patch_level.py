from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import deque

@dataclass
class Patch:
    obj_pair: Tuple[int, int]
    _raw_g_pair: List[Dict]
    patch: List[List[int]]

    def __init__(self, obj1, obj2, g1: Dict, g2: Dict):
        self.obj_pair = (obj1.id, obj2.id)
        self._raw_g_pair = [g1, g2]
        self.patch = []
        self.component_detect(g1, g2)

    @staticmethod
    def bfs_search_patch(graph_raw_data: Dict[int, List[int]]) -> List[List[int]]:
        visited = set()
        connected_components: List[List[int]] = []
        queue = deque()
        for node in graph_raw_data:
            if node not in visited:
                component: List[int] = []
                visited.add(node)
                component.append(node)
                queue.append(node)
                while queue:
                    current_node = queue.popleft()
                    for neighbor in graph_raw_data.get(current_node, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            component.append(neighbor)
                            queue.append(neighbor)
                connected_components.append(component)
        return connected_components

    def component_detect(self, g1: Dict, g2: Dict):
        g1_mod = {tri: elem.get('adj', []) for tri, elem in g1.items()}
        g2_mod = {tri: elem.get('adj', []) for tri, elem in g2.items()}
        g1_res =[]
        g2_res= []
        if not g1_mod or not g2_mod:
            return g1_res, g2_res
        connected_components_g1 = Patch.bfs_search_patch(g1_mod)
        connected_components_g2 = Patch.bfs_search_patch(g2_mod)
        for component in connected_components_g1:
            g1_res.append({"component": component, "adj": []})
        for component in connected_components_g2:
            g2_res.append({"component": component, "adj": []})
        for id_g1,component_g1 in enumerate(g1_res):
            for id_g2,component_g2 in enumerate(g2_res):
                for elem in component_g2['component']:
                    status=False
                    for adj_g1 in g2[elem]['cover_area']['true']['adj_tri_ids']:
                        if adj_g1 in component_g1['component']:
                            status=True
                            break
                    if status==True:
                        component_g1['adj'].append(id_g2)
                        component_g2['adj'].append(id_g1)
                        break
        self.patch=[g1_res, g2_res]
        pass


                
        
