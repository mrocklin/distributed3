from __future__ import annotations

from typing import Iterable

import pytest

import dask

from distributed.scheduler import TaskGroup, TaskState


@pytest.fixture()
def abcde():
    return "abcde"


def f(*args):
    return None


def dummy_dsk_to_taskstate(dsk: dict) -> tuple[list[TaskState], dict[str, TaskGroup]]:
    task_groups: dict[str, TaskGroup] = {}
    tasks = dict()
    priority = dask.order.order(dsk)
    for key in dsk:
        tasks[key] = ts = TaskState(key, None, "released")
        ts.group = task_groups.get(ts.group_key, TaskGroup(ts.group_key))
        task_groups[ts.group_key] = ts.group
        ts.group.add(ts)
        ts.priority = priority[key]
    for key, vals in dsk.items():
        stack = list(vals[1:])
        while stack:
            d = stack.pop()
            if isinstance(d, list):
                stack.extend(d)
                continue
            assert isinstance(d, (str, tuple, int))
            if d not in tasks:
                raise ValueError(f"Malformed example. {d} not part of dsk")
            tasks[key].add_dependency(tasks[d])
    return sorted(tasks.values(), key=lambda ts: ts.priority), task_groups


def _to_keys(prefix: str, suffix: Iterable[str]) -> list[str]:
    return list(prefix + "-" + i for i in suffix)


def test_tree_reduce(abcde):
    a, b, c, _, _ = abcde
    a_ = _to_keys(a, "123456789")
    b_ = _to_keys(b, "1234")
    dsk = {
        a_[0]: (f,),
        a_[1]: (f,),
        a_[2]: (f,),
        b_[0]: (f, a_[0], a_[1], a_[2]),
        a_[3]: (f,),
        a_[4]: (f,),
        a_[5]: (f,),
        b_[1]: (
            f,
            a_[6],
            a_[7],
            a_[8],
        ),
        a_[6]: (f,),
        a_[7]: (f,),
        a_[8]: (f,),
        b_[2]: (f, a_[6], a_[7], a_[8]),
        c: (f, b_[0], b_[1], b_[2]),
    }
    _, groups = dummy_dsk_to_taskstate(dsk)
    assert len(groups) == 3
    assert len(groups["a"]) == 9
    assert groups["a"].rootish
    assert not groups["b"].rootish
    assert not groups["c"].rootish


@pytest.mark.parametrize("num_Bs", [4, 5])
def test_nearest_neighbor(abcde, num_Bs):
    r"""
    a1  a2  a3  a4  a5  a6  a7 a8  a9
     \  |  /  \ |  /  \ |  / \ |  /
        b1      b2      b3     b4

    How these groups are classified depends on an implementation detail in the
    scheduler since we're defining a relatively artificial cutoff requiring
    root-ish groups to have at least 5 tasks.
    """
    a, b, c, _, _ = abcde
    a_ = _to_keys(a, "0123456789")
    aa_ = _to_keys(a, ["10", "11", "12"])
    b_ = _to_keys(b, "012345")

    dsk = {
        b_[1]: (f,),
        b_[2]: (f,),
        b_[3]: (f,),
        b_[4]: (f,),
        a_[1]: (f, b_[1]),
        a_[2]: (f, b_[1]),
        a_[3]: (f, b_[1], b_[2]),
        a_[4]: (f, b_[2]),
        a_[5]: (f, b_[2], b_[3]),
        a_[6]: (f, b_[3]),
        a_[7]: (f, b_[3], b_[4]),
        a_[8]: (f, b_[4]),
        a_[9]: (f, b_[4]),
    }
    if num_Bs == 5:
        dsk[b_[5]] = ((f,),)
        dsk[a_[9]] = ((f, b_[4], b_[5]),)
        dsk[aa_[0]] = ((f, b_[5]),)
        dsk[aa_[1]] = ((f, b_[5]),)
    _, groups = dummy_dsk_to_taskstate(dsk)
    assert len(groups) == 2

    # FIXME: This is an artifact of the magic numbers in the rootish
    # classification
    if num_Bs == 5:
        assert not groups["a"].rootish
    else:
        assert groups["a"].rootish
    assert groups["b"].rootish


@pytest.mark.parametrize("num_Bs", range(3, 8))
def test_base_of_reduce_preferred(abcde, num_Bs):
    r"""
                a4
               /|
             a3 |
            /|  |
          a2 |  |
         /|  |  |
       a1 |  |  |
      /|  |  |  |
    a0 |  |  |  |
    |  |  |  |  |
    b0 b1 b2 b3 b4
      \ \ / /  /
         c
    """
    a, b, c, d, e = abcde
    dsk = {(a, i): (f, (a, i - 1), (b, i)) for i in range(1, num_Bs)}
    dsk[(a, 0)] = (f, (b, 0))
    dsk.update({(b, i): (f, c) for i in range(num_Bs)})
    dsk[c] = (f,)

    _, groups = dummy_dsk_to_taskstate(dsk)
    assert len(groups) == 3
    assert not groups["a"].rootish
    assert groups["b"].rootish
    assert groups["c"].rootish
