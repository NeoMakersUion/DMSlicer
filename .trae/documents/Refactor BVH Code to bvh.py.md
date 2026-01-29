# Plan: Refactor BVH Logic to `bvh.py`

The user has moved the BVH-related code (`AABB`, `BVHNode`, `build_bvh`, `query_bvh`) to `d:\DMSlicer\src\dmslicer\geometry_kernel\bvh.py`. However, the file is currently missing necessary imports.

I need to:

1. **Supplement imports in** **`bvh.py`**: Add `numpy`, `typing` (List, Tuple, Optional), and potentially others used by the BVH logic.
2. **Clean up** **`canonicalize.py`**: Remove the `AABB`, `BVHNode`, `build_bvh`, and `query_bvh` definitions since they are now in `bvh.py`.
3. **Update** **`canonicalize.py`** **imports**: Ensure `canonicalize.py` imports these classes/functions from `.bvh`.
4. **Verify** **`test/test_bvh.py`**: Ensure the test script imports from the correct location.

## Detailed Steps:

### 1. Update `d:\DMSlicer\src\dmslicer\geometry_kernel\bvh.py`

Add the following imports to the top of the file:

```python
import numpy as np
from typing import List, Tuple, Optional
```

### 2. Update `d:\DMSlicer\src\dmslicer\geometry_kernel\canonicalize.py`

* **Remove**: `class AABB`, `class BVHNode`, `def build_bvh`, `def query_bvh`.

* **Add/Verify Import**: `from .bvh import AABB, BVHNode, build_bvh, query_bvh`. (It seems the user might have already added `from .bvh import BVHNode,query_bvh` in line 15 of `canonicalize.py`, but `AABB` and `build_bvh` are missing from the import list if they are used locally, or if `AABB` is used in type hints).

  * Wait, looking at `canonicalize.py` read result:
    Line 15: `from .bvh import BVHNode,query_bvh`
    Line 41-196: The definitions of `AABB`, `BVHNode`, `build_bvh`, `query_bvh` are **still present** in `canonicalize.py` in the version I just read.

  * **Correction**: The user said "I *have* moved... to this file", but the file content of `canonicalize.py` (lines 41-196) shows they are still there. The user might have copied them to `bvh.py` (which I saw has the code but no imports) but *not* deleted them from `canonicalize.py` yet, or they expect me to do the cleanup.

  * **Action**: I will delete these definitions from `canonicalize.py` and ensure the import at line 15 includes all necessary symbols: `from .bvh import AABB, BVHNode, build_bvh, query_bvh`.

### 3. Update `d:\DMSlicer\test\test_bvh.py`

* Change `from dmslicer.geometry_kernel.canonicalize import AABB, BVHNode, build_bvh, query_bvh`

* To `from dmslicer.geometry_kernel.bvh import AABB, BVHNode, build_bvh, query_bvh`.

## Execution Order

1. Write imports to `bvh.py`.
2. Remove code from `canonicalize.py` and update imports.
3. Update `test/test_bvh.py`.
4. Run tests to verify.

