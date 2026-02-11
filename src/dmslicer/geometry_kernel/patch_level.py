from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import deque
import pandas as pd
from tqdm import tqdm
import numpy as np
def build_patch_graph(obj,tri_ids,df,tri_col):
            """
            构建单对象的“面片邻接统计图”并返回字典形式的图结构。
            
            该函数基于三角形列表 tri_ids 与三角形对统计表 df（由上一阶段的 pair-level 计算生成），
            为每个三角形计算：
            - adj：与其共享边的邻接三角形且同时出现在 tri_ids 中
            - deg：与邻接三角形法线的最大无向夹角（单位°，基于 abs(dot(n0, nj)) 计算；小于 1° 的角度视为 0；若无有效邻接则为 None）
            - cover_area：按照 area_pass 筛选后的覆盖统计（true/false 两类，包含累计面积与对端三角列表）
            同时在二阶统计中计算：
            - ddeg：所有邻接三角形的 deg 差值的最大值（单位°；小于 1° 的差值视为 0；若不存在可比较的邻接 deg，则为 None）
            
            参数:
                obj: 目标 Object，包含局部三角形数据与 topology 信息
                tri_ids: 当前对象参与统计的三角形 ID 列表（来自 df_true 的 tri1/tri2 去重集合）
                df: pandas.DataFrame，面片对统计表，至少包含 [tri_col, area_pass, cover_col] 等列
                tri_col: 字段名，指示当前对象对应列，如 "tri1" 或 "tri2"
                cover_col: 字段名，对应覆盖比例列，如 "cover1" 或 "cover2"
            
            返回:
                dict[int, dict]: 
                    {
                      tri_id: {
                        "adj": List[int],                # 邻接三角形 ID
                        "deg": float,                    # 与邻接的最大法线夹角（°）
                        "cover_area": {
                            "true":  {"area": float, "list": List[List[int]]},
                            "false": {"area": float, "list": List[List[int]]}
                        },
                        "ddeg": float                    # 与邻接的最大角差（°）
                      },
                      ...
                    }
            
            说明:
            - 法线夹角计算采用 abs(dot(n0, nj)) 并通过 np.clip 保证 arccos 入参在 [0,1]
            - 角度与角差的“门槛”为 1°，用于抑制数值抖动；低于阈值记为 0
            - 邻接集合来源于三角形 topology['edges'] 的值集合，并与 tri_ids 取交集以保持一致性
            """
            g = {}
            if tri_col == "tri1":
                cover_col = "cover1"
                adj_tri_col= "tri2"
            elif tri_col == "tri2":
                cover_col = "cover2"
                adj_tri_col= "tri1"
            else:
                raise ValueError(f"tri_col must be 'tri1' or 'tri2', but got {tri_col}")
            # ---------- 一阶：adj + deg + cover ----------
            for tri_id in tqdm(tri_ids,desc="build_patch_graph_deg",total=len(tri_ids),leave=False):
                df_tri = df[df[tri_col] == tri_id]

                df_true  = df_tri[df_tri['area_pass'] == True]
                df_false = df_tri[df_tri['area_pass'] == False]
                adj_tri_ids_in_df_true = list(set(df_true[adj_tri_col]))
                cover_true  = {"area_ratio": df_true[cover_col].sum(),
                            "area_acc": df_true['intersection_area'].sum(),
                            "adj_tri_ids": adj_tri_ids_in_df_true}
                adj_tri_ids_in_df_false = list(set(df_false[adj_tri_col]))
                cover_false = {"area_ratio": df_false[cover_col].sum(),
                            "area_acc": df_false['intersection_area'].sum(),
                            "adj_tri_ids": adj_tri_ids_in_df_false}

                tri = obj.triangles[tri_id]
                adj_all = list(tri.topology['edges'].values())
                adj = [t for t in adj_all if t in tri_ids]

                # --- deg ---
                n0 = tri.normal
                degs = []
                for j in adj:
                    nj = obj.triangles[j].normal
                    if n0 is None or nj is None:
                        continue
                    angle = np.degrees(
                        np.arccos(np.clip(np.abs(np.dot(n0, nj)), 0.0, 1.0))
                    )
                    angle = angle if angle >= 1.0 else 0.0
                    degs.append(angle)

                g[tri_id] = {
                    "adj": adj,
                    "deg": float(np.max(degs)) if degs else None,
                    "cover_area": {
                        "true": cover_true,
                        "false": cover_false
                    }
                }

            # ---------- 二阶：ddeg ----------
            for tri_id, elem in tqdm(g.items(),desc="build_patch_graph_ddeg",total=len(g),leave=False):
                dds = []
                for j in elem["adj"]:
                    if j in g:
                        dd = abs(elem["deg"] - g[j]["deg"])
                        dd = dd if dd >= 1.0 else 0.0
                        dds.append(dd)
                elem["ddeg"] = float(np.max(dds)) if dds else None
            return g
        
@dataclass
class Patch:
    df: pd.DataFrame
    obj_pair: Tuple[int, int]
    _raw_g_pair: List[Dict]
    patch: List[List[int]]

    def __init__(self, obj1, obj2, df):
        self.df=df
        df_true = df[df['area_pass'] == True]
        tri1=list(set(df_true['tri1']))
        tri2=list(set(df_true['tri2']))
        g1 = build_patch_graph(obj1, tri1, df, tri_col="tri1")
        g2 = build_patch_graph(obj2, tri2, df, tri_col="tri2")
        self.obj_pair = (obj1.id, obj2.id)
        self._raw_g_pair = {obj1.id: g1, obj2.id: g2}
        self.patch = {obj1.id: [], obj2.id: []}
        self.component_detect(g1, g2)
        self.split_patch_with_norm_deg_ddeg(obj1,obj2)

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
        obj1_id,obj2_id=self.obj_pair
        self.patch={obj1_id: g1_res, obj2_id: g2_res}
        pass
    
    def split_patch_with_norm_deg_ddeg(self,obj1,obj2):
        def turn_g_to_df(g):
            df_rows=[]
            for tri_id,attrs in g.items():
                # 基础字段
                row = {
                    'tri_id': tri_id,  # 把外层的数字key作为tri_id列
                    'adj': attrs['adj'],
                    'deg': attrs['deg'],
                    'ddeg': attrs['ddeg']
                }
                    # 展开cover_area的true/false子字段（避免嵌套）
                row['cover_area_true_ratio'] = attrs['cover_area']['true']['area_ratio']
                row['cover_area_true_acc'] = attrs['cover_area']['true']['area_acc']
                row['cover_area_true_adj_tri_ids'] = attrs['cover_area']['true']['adj_tri_ids']
                row['cover_area_false_ratio'] = attrs['cover_area']['false']['area_ratio']
                row['cover_area_false_acc'] = attrs['cover_area']['false']['area_acc']
                row['cover_area_false_adj_tri_ids'] = attrs['cover_area']['false']['adj_tri_ids']
                df_rows.append(row)
            if df_rows:
                df_component = pd.DataFrame(df_rows)
            else:
                df_component = pd.DataFrame(columns=[
                    'tri_id','adj','deg','ddeg',
                    'cover_area_true_ratio','cover_area_true_acc','cover_area_true_adj_tri_ids',
                    'cover_area_false_ratio','cover_area_false_acc','cover_area_false_adj_tri_ids'
                ])
            return df_component      
        g1=self._raw_g_pair[obj1.id]
        df_obj1=turn_g_to_df(g1)
        if 'ddeg' in df_obj1.columns and not df_obj1.empty:
            boundry_tri1=set(df_obj1[df_obj1['ddeg']>0]['tri_id'].tolist())
        else:
            boundry_tri1=set()
        for obj1_component in self.patch[obj1.id]:
            if boundry_tri1==set():
                continue
            obj1_component_set=set(obj1_component['component'])
            diff_obj1_component_set=obj1_component_set-boundry_tri1
            pass

        # obj1_component_set=set(self.patch[obj1.id][0]['component'])
        # g2=self._raw_g_pair[obj2.id]
        # df_obj2=turn_g_to_df(g2)
        pass
        
        
