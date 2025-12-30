# Code Review: Cleanup & Optimization Recommendations

## Executive Summary

This review identifies **unused code, duplicate code, and opportunities to reduce lines** throughout the codebase. The recommendations are organized by impact and effort required.

**Estimated Total Lines Removable**: ~500-700 lines
**Files That Can Be Deleted**: 3-5 files
**Functions That Can Be Removed**: 5-8 functions

---

## 1. UNUSED FILES (Can Delete Entirely)

### 1.1 `umapviz.py` (305 lines) - **DELETE**

**Status**: Standalone exploration script, never imported
**Evidence**:

- No imports of `umapviz` or `explore_umap` functions found in codebase
- Functions are defined but never called
- Contains bugs (line 65: `saveGraph()` call missing, line 133: undefined `n_neighbors_options`)

**Impact**: Remove 305 lines
**Risk**: Low - Not used anywhere

```python
# These functions are never called:
- explore_umap_parameters_2d()
- explore_umap_parameters_2d_min_dist()
- explore_umap_parameters_3d()
- explore_umap_parameters_3d_individual()
```

**Recommendation**: Delete file. If UMAP exploration is needed, it should be integrated into `kmeans/cluster_viz.py` which already has UMAP visualization functions.

---

### 1.2 `keyword_extraction.py` (28 lines) - **DELETE**

**Status**: Standalone example/test script
**Evidence**:

- Not imported anywhere
- Contains hardcoded example data
- Just demonstrates KeyBERT usage

**Impact**: Remove 28 lines
**Risk**: Low - Appears to be experimental/example code

**Recommendation**: Delete file. If keyword extraction is needed, integrate into `kmeans/cluster_titles.py` or create a proper module.

---

### 1.3 `dcscan/dbscan.py` (312 lines) - **CONSIDER ARCHIVING**

**Status**: Not imported in `main.py`, standalone DBSCAN exploration
**Evidence**:

- Not imported in main entry point
- Contains useful DBSCAN functions but not integrated into pipeline
- Functions like `test_dbscan_params()`, `final_dbscan()` are standalone

**Impact**: Remove 312 lines (if not needed)
**Risk**: Medium - May be useful for future DBSCAN experiments

**Recommendation**:

- If DBSCAN is not actively used, move to `archive/` or `experiments/` directory
- If keeping, integrate into main pipeline or create proper module structure

---

## 2. DUPLICATE CODE & REDUNDANCIES

### 2.1 `utils.py:31-38` - Duplicate `res = []` Assignment

**Location**: `utils.py` lines 32 and 34
**Issue**: Variable initialized twice unnecessarily

```python
def getAllVerses():
    res = []  # ❌ Line 32: First assignment
    data = getDataFromJson()
    res = []  # ❌ Line 34: Duplicate assignment (unnecessary)
    for verses in data.values():
        for text in verses.values():
            res.append(text)
    return res
```

**Fix**: Remove line 34
**Impact**: Remove 1 line, cleaner code

```python
def getAllVerses():
    res = []
    data = getDataFromJson()
    # Remove duplicate res = []
    for verses in data.values():
        for text in verses.values():
            res.append(text)
    return res
```

**Better Fix**: Simplify entire function:

```python
def getAllVerses():
    data = getDataFromJson()
    return [text for verses in data.values() for text in verses.values()]
```

**Impact**: Reduce from 7 lines to 3 lines (57% reduction)

---

### 2.2 Duplicate `getAllVerses()` Functions

**Location**:

- `utils.py:31-38` (used in codebase)
- `scripts/cleanup_proverbs.py:15-58` (different implementation, only used in that script)

**Issue**: Two different implementations of same function name
**Impact**: Confusion, maintenance burden

**Recommendation**:

- Keep `utils.py` version (used elsewhere)
- Rename `scripts/cleanup_proverbs.py` version to `_getAllVerses()` or inline it

---

### 2.3 `kmeans/kmeans.py:20-61` - `print_clusters()` Function

**Status**: Only used in legacy `run_kmeans_experiments()` function
**Issue**: Duplicates functionality of `utils.organize_clusters_data()` but with printing

**Current Usage**:

- Only called from `run_kmeans_experiments()` (line 137)
- `run_kmeans_experiments()` is commented out in `main.py` (line 128)

**Recommendation**:

- Since `organize_clusters_data()` exists in `utils.py`, `print_clusters()` can be removed
- If printing is needed, add optional print parameter to `organize_clusters_data()`
- **Impact**: Remove ~42 lines

---

## 3. UNUSED FUNCTIONS & IMPORTS

### 3.1 Legacy K-means Functions

**Location**: `kmeans/kmeans.py`

**Functions to Remove**:

- `print_clusters()` - Only used in legacy function (see 2.3)
- `run_kmeans_experiments()` - Commented out in `main.py`, replaced by `clustering_pipeline.py`

**Evidence**:

```python
# main.py:128
# run_kmeans_experiments(arr, verse_refs)  # ❌ Commented out
```

**Recommendation**:

- Keep file for reference but remove unused functions, OR
- Move entire `kmeans/kmeans.py` to `archive/` if not needed
- **Impact**: Remove ~140 lines

---

### 3.2 Unused Imports

#### `main.py`

```python
from kmeans.kmeans import run_kmeans_experiments  # ❌ Only used in commented code
from community_graph.community_graph import create_community_graph  # ❌ Commented out
```

**Fix**: Remove unused imports
**Impact**: Remove 2 lines

#### `kmeans/kmeans.py`

```python
from pathlib import Path  # ❌ Not used
import json  # ❌ Only used in print_clusters which can be removed
```

**Fix**: Remove if `print_clusters()` is removed
**Impact**: Remove 2 lines

#### `umapviz.py` (if keeping)

```python
from utils import saveGraph  # ❌ Not used in explore_umap_parameters_2d (bug on line 65)
```

---

## 4. CODE SIMPLIFICATIONS

### 4.1 `utils.py:31-38` - Simplify `getAllVerses()`

**Current**: 7 lines with nested loops
**Simplified**: 3 lines with list comprehension

```python
# Before (7 lines)
def getAllVerses():
    res = []
    data = getDataFromJson()
    res = []  # duplicate
    for verses in data.values():
        for text in verses.values():
            res.append(text)
    return res

# After (3 lines)
def getAllVerses():
    data = getDataFromJson()
    return [text for verses in data.values() for text in verses.values()]
```

**Impact**: Reduce 7 lines to 3 lines (57% reduction)

---

### 4.2 `scripts/getProverbsForSummarizing.py` - Simplify File Writing

**Location**: Lines 10-15
**Issue**: Opens file multiple times in append mode (inefficient)

```python
# Current (inefficient)
for cluster in data.values():
    with open(out_file, "a") as f:  # ❌ Opens file multiple times
        f.write(f"\n\nCluster {cluster['cluster_id']}\n")
    for verse in cluster["verses"]:
        with open(out_file, "a") as f:  # ❌ Opens file multiple times
            f.write(f"{verse['text']}\n")
```

**Fix**: Open once, write all at once

```python
# Better (efficient)
with open(out_file, "w") as f:
    for cluster in data.values():
        f.write(f"\n\nCluster {cluster['cluster_id']}\n")
        for verse in cluster["verses"]:
            f.write(f"{verse['text']}\n")
```

**Impact**: Reduce 6 lines to 5 lines, better performance

---

### 4.3 `scripts/makek20_clusters_titles_for_claude.py` - Remove Hardcoded Paths

**Location**: Lines 4, 13
**Issue**: Hardcoded experiment paths, appears to be one-off script

**Recommendation**:

- If this was a one-time script, delete it
- If keeping, use command-line arguments or config
- **Impact**: Remove 20 lines if deleting

---

### 4.4 `scripts/use_freq_for_titles.py` - Typo in Class Name

**Location**: Line 6
**Issue**: `FrequyencyTitleGenerator` should be `FrequencyTitleGenerator`

**Fix**: Rename class
**Impact**: 1 line change, better code quality

---

## 5. LEGACY/EXPERIMENTAL SCRIPTS

### 5.1 Scripts Directory Analysis

**Potentially Unused Scripts** (one-off utilities):

- `scripts/getProverbsForSummarizing.py` - Hardcoded paths, appears one-time
- `scripts/makek20_clusters_titles_for_claude.py` - Hardcoded experiment path
- `scripts/use_freq_for_titles.py` - Experimental frequency-based titles
- `scripts/use_claude_for_titles.py` - Legacy Claude API integration (replaced by `cluster_titles.py`)

**Recommendation**:

- Review each script - if it was a one-time utility, move to `archive/` or delete
- If keeping, add proper CLI arguments and documentation
- **Impact**: Remove 100-200 lines if archiving unused scripts

---

## 6. BUGS FOUND (Bonus Fixes)

### 6.1 `umapviz.py:65` - Missing `saveGraph()` Call

**Issue**: Function creates plot but doesn't save it

```python
plt.tight_layout()
# saveGraph(save_filename, plt)  # ❌ Missing!
plt.close(fig)
```

### 6.2 `umapviz.py:133` - Undefined Variable

**Issue**: `n_neighbors_options` used but not defined in `explore_umap_parameters_3d()`

```python
n_cols = len(n_neighbors_options)  # ❌ Variable not defined
```

---

## 7. SUMMARY OF RECOMMENDATIONS

### High Impact, Low Risk (Do First)

1. ✅ **Delete `umapviz.py`** - 305 lines, unused, has bugs
2. ✅ **Delete `keyword_extraction.py`** - 28 lines, example code
3. ✅ **Fix duplicate `res = []` in `utils.py`** - 1 line, simple fix
4. ✅ **Simplify `getAllVerses()`** - Reduce 7 lines to 3 lines
5. ✅ **Remove unused imports in `main.py`** - 2 lines

**Total**: ~340 lines removable

### Medium Impact, Medium Risk

6. ⚠️ **Archive or integrate `dcscan/dbscan.py`** - 312 lines, not integrated
7. ⚠️ **Remove legacy `print_clusters()` and `run_kmeans_experiments()`** - ~140 lines
8. ⚠️ **Review and archive unused scripts** - 100-200 lines

**Total**: ~550 lines potentially removable

### Low Impact, Low Risk (Polish)

9. 🔧 **Fix typo in `FrequyencyTitleGenerator`**
10. 🔧 **Simplify file writing in `getProverbsForSummarizing.py`**
11. 🔧 **Remove unused imports throughout**

---

## 8. IMPLEMENTATION PRIORITY

### Phase 1: Quick Wins (30 minutes)

- Fix duplicate `res = []` in `utils.py`
- Simplify `getAllVerses()` function
- Remove unused imports in `main.py`
- **Result**: ~10 lines removed, cleaner code

### Phase 2: Delete Unused Files (15 minutes)

- Delete `umapviz.py`
- Delete `keyword_extraction.py`
- **Result**: ~333 lines removed

### Phase 3: Legacy Code Cleanup (1 hour)

- Remove `print_clusters()` from `kmeans/kmeans.py`
- Comment out or remove `run_kmeans_experiments()` (keep file for reference)
- Review scripts directory
- **Result**: ~150-200 lines removed

### Phase 4: Archive Experimental Code (30 minutes)

- Move `dcscan/dbscan.py` to `archive/` if not actively used
- Archive one-off scripts
- **Result**: ~300-400 lines archived

---

## 9. ESTIMATED TOTAL IMPACT

| Category             | Lines Removable | Risk Level |
| -------------------- | --------------- | ---------- |
| Unused Files         | ~333            | Low        |
| Duplicate Code       | ~50             | Low        |
| Legacy Functions     | ~140            | Medium     |
| Experimental Scripts | ~200            | Medium     |
| Code Simplifications | ~20             | Low        |
| **TOTAL**            | **~743 lines**  |            |

**Percentage of Codebase**: Approximately 15-20% reduction in total lines

---

## 10. NOTES

- **Backup before deleting**: Some files may have historical value
- **Git history**: Deleted files remain in git history if needed
- **Documentation**: Consider adding README in `archive/` explaining why files were moved
- **Testing**: After cleanup, verify main pipeline still works:
  ```bash
  python main.py  # Should still work
  ```

---

## Questions to Consider

1. Is `dcscan/dbscan.py` needed for future experiments?
2. Are the scripts in `scripts/` one-time utilities or reusable tools?
3. Should `kmeans/kmeans.py` be kept for reference or fully removed?
4. Is there a need to keep `umapviz.py` for future UMAP exploration?
