"""材料分配与面向对象视图交互的控制模块。
Controls module for material assignment and object-focused viewport interactions.

本模块驱动 Streamlit 中的 Materialize/Processing 面板，提供颜色归一化、表格行构建、
材料状态同步、材料应用与持久化入口等工具函数。
This module drives the Materialize/Processing panels in Streamlit and provides helpers
for color normalization, table-row building, material-state syncing, material application,
and persistence entry points.
"""
from .materialization import Materialize

def render_controls():
    """以兼容方式渲染控制面板，委托给 Materialize。
    Render controls via backward-compatible Materialize wrapper.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Delegates directly to :func:`Materialize`.

    Raises
    ------
    streamlit.errors.StreamlitAPIException
        Propagated from nested Streamlit UI calls.

    Examples
    --------
    >>> render_controls()  # doctest: +SKIP
    """
    Materialize()
    
