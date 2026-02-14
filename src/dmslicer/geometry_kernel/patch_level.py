from typing import Any, Dict, List, Tuple
from collections import deque
import pandas as pd
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
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

                df_true  = df_tri[df_tri['area_pass']]
                df_false = df_tri[~df_tri['area_pass']]
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
        
def _stable_unique(seq: Any) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for x in seq:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi not in seen:
            seen.add(xi)
            out.append(xi)
    return out


def create_summary(df: pd.DataFrame, group_field: str, cover_field: str, tri_field: str, obj: Any) -> pd.DataFrame:
    summary_df = (
        df[df["area_pass"]]
        .groupby(group_field)
        .agg(
            cover_sum=(cover_field, "sum"),
            adj_obj_list=(tri_field, lambda x: _stable_unique(x)),
        )
        .reset_index()
    )
    summary_df_sorted = summary_df.sort_values(by="cover_sum", ascending=True).reset_index(drop=True)
    tri_list = list(summary_df_sorted[group_field])
    parametric_area_grade_list = [
        tri.parametric_area_grade for tri in obj.get_triangles_by_list(tri_list)
    ]
    summary_df_sorted["parametric_area_grade"] = parametric_area_grade_list
    first_col_name = summary_df_sorted.columns.tolist()[0]
    summary_df_sorted_renamed = summary_df_sorted.rename(columns={first_col_name: "tri_id"})
    summary_df_sorted_renamed = summary_df_sorted_renamed[
        ["tri_id", "cover_sum", "adj_obj_list", "parametric_area_grade"]
    ]
    return summary_df_sorted_renamed


def get_dynamic_threshold(parametric_area_grade: int) -> float:
    if parametric_area_grade == 4:
        return 0.10
    elif parametric_area_grade == 3:
        return 0.03
    elif parametric_area_grade == 2:
        return 0.01
    else:
        return 0.01


def _build_adj_eval(adj_list: Any, other_map: Dict[int, bool]) -> Dict[int, Any]:
    if adj_list is None or (isinstance(adj_list, float) and np.isnan(adj_list)):
        return {}
    if not isinstance(adj_list, (list, tuple, set)):
        return {}
    out: Dict[int, Any] = {}
    for t in adj_list:
        try:
            tid = int(t)
        except Exception:
            continue
        out[tid] = other_map.get(tid, None)
    return out


def validate_patch_with_dynamic_threshold(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    threshold_fn: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"tri_id", "cover_sum", "adj_obj_list", "parametric_area_grade"}
    missing1 = required - set(df1.columns)
    missing2 = required - set(df2.columns)
    if missing1:
        raise ValueError(f"df1 missing columns: {missing1}")
    if missing2:
        raise ValueError(f"df2 missing columns: {missing2}")

    df1_out = df1.copy()
    df2_out = df2.copy()
    df1_out["threshold"] = df1_out["parametric_area_grade"].apply(threshold_fn)
    df2_out["threshold"] = df2_out["parametric_area_grade"].apply(threshold_fn)
    df1_out["main_evaluation"] = df1_out["cover_sum"] > df1_out["threshold"]
    df2_out["main_evaluation"] = df2_out["cover_sum"] > df2_out["threshold"]
    df1_eval_map: Dict[int, bool] = dict(
        zip(df1_out["tri_id"].astype(int), df1_out["main_evaluation"].astype(bool))
    )
    df2_eval_map: Dict[int, bool] = dict(
        zip(df2_out["tri_id"].astype(int), df2_out["main_evaluation"].astype(bool))
    )
    df1_out["adj_obj_tri_evaluation"] = df1_out["adj_obj_list"].apply(lambda adj: _build_adj_eval(adj, df2_eval_map))
    df2_out["adj_obj_tri_evaluation"] = df2_out["adj_obj_list"].apply(lambda adj: _build_adj_eval(adj, df1_eval_map))
    df1_out["final_evaluation"] = df1_out.apply(
        lambda row: bool(row["main_evaluation"]) or any(v is True for v in dict(row["adj_obj_tri_evaluation"]).values()),
        axis=1,
    )
    df2_out["final_evaluation"] = df2_out.apply(
        lambda row: bool(row["main_evaluation"]) or any(v is True for v in dict(row["adj_obj_tri_evaluation"]).values()),
        axis=1,
    )
    return df1_out, df2_out

@dataclass
class Patch:
    df: pd.DataFrame
    obj_pair: Tuple[int, int]
    _raw_g_pair: Dict[int, Dict[int, List[int]]]
    patch: Dict[int, List[Dict[str, Any]]]

    def __init__(self, obj1: Any, obj2: Any, df: pd.DataFrame, show: str = "false"):
        self.df = df
        df_true = df[df["area_pass"]]
        self.obj_pair = (obj1.id, obj2.id)
        if df_true.empty:
            self._raw_g_pair = {obj1.id: {}, obj2.id: {}}
            self.patch = {obj1.id: [], obj2.id: []}
            return
        s1 = create_summary(df_true, "tri1", "cover1", "tri2", obj1)
        s2 = create_summary(df_true, "tri2", "cover2", "tri1", obj2)
        s1e, s2e = validate_patch_with_dynamic_threshold(s1, s2, get_dynamic_threshold)
        tri1 = list(s1e[s1e["final_evaluation"]]["tri_id"])
        tri2 = list(s2e[s2e["final_evaluation"]]["tri_id"])
        g1 = build_patch_graph(obj1, tri1, df, tri_col="tri1")
        g2 = build_patch_graph(obj2, tri2, df, tri_col="tri2")
        self._raw_g_pair = {obj1.id: g1, obj2.id: g2}
        self.patch = {obj1.id: [], obj2.id: []}
        self.component_detect(g1, g2)
        if show == "true":
            self._debug_show(obj1, obj2)

    # removed legacy __init__ and nested helpers

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
                    if status:
                        component_g1['adj'].append(id_g2)
                        component_g2['adj'].append(id_g1)
                        break
        obj1_id, obj2_id = self.obj_pair
        self.patch = {obj1_id: g1_res, obj2_id: g2_res}

    
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
            pass

        # obj1_component_set=set(self.patch[obj1.id][0]['component'])
        # g2=self._raw_g_pair[obj2.id]
        # df_obj2=turn_g_to_df(g2)
        pass

    def _debug_show(self,obj1,obj2) -> None:
        try:
            from ..visualizer.visualizer_interface import IVisualizer
        except Exception:
            return
        visualizer = IVisualizer.create()
        g1_res = self.patch[obj1.id]
        g2_res = self.patch[obj2.id]
        visualizer.addObj(obj1, opacity=0.1)
        visualizer.addObj(obj2, opacity=0.1)
        for component_g1 in g1_res:
            tris1 = component_g1["component"]
            visualizer.addObj(obj1, tris1)
        for component_g2 in g2_res:
            tris2 = component_g2["component"]
            visualizer.addObj(obj2, tris2)
        visualizer.show()
        
        
