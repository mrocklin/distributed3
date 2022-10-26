from __future__ import annotations

import pytest

import dask

from distributed.scheduler import TaskGroup, TaskState, coassignment_groups


def f(*args):
    return None


def assert_disjoint_sets(cogroups):
    seen = set()
    for values in cogroups.values():
        assert not (seen & values)
        seen.update(values)


@pytest.fixture(params=["abcde"])
def abcde(request):
    return request.param


def dummy_dsk_to_taskstate(dsk: dict) -> list[TaskState]:
    task_groups: dict[str, TaskGroup] = {}
    tasks = dict()
    priority = dask.order.order(dsk)
    for key in dsk:
        tasks[key] = ts = TaskState(key, None, "released")
        ts.group = task_groups.get(ts.group_key, TaskGroup(ts.group_key))
        task_groups[ts.group_key] = ts.group
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
    return sorted(tasks.values(), key=lambda ts: ts.priority)


def test_tree_reduce(abcde):
    r""" """
    a, b, c, _, _ = abcde
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = (a + i for i in "123456789")
    b1, b2, b3, b4 = (b + i for i in "1234")
    dsk = {
        a1: (f,),
        a2: (f,),
        a3: (f,),
        b1: (f, a1, a2, a3),
        a4: (f,),
        a5: (f,),
        a6: (f,),
        b2: (f, a4, a5, a6),
        a7: (f,),
        a8: (f,),
        a9: (f,),
        b3: (f, a7, a8, a9),
        c: (f, b1, b2, b3),
    }
    tasks = dummy_dsk_to_taskstate(dsk)
    assert isinstance(tasks, list)
    cogroups = coassignment_groups(tasks)
    assert_disjoint_sets(cogroups)
    assert len(cogroups) == 3


def test_nearest_neighbor(abcde):
    r"""
    a1  a2  a3  a4  a5  a6  a7 a8  a9
     \  |  /  \ |  /  \ |  / \ |  /
        b1      b2      b3     b4

    No co-groups
    """
    a, b, c, _, _ = abcde
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = (a + i for i in "123456789")
    b1, b2, b3, b4 = (b + i for i in "1234")

    dsk = {
        b1: (f,),
        b2: (f,),
        b3: (f,),
        b4: (f,),
        a1: (f, b1),
        a2: (f, b1),
        a3: (f, b1, b2),
        a4: (f, b2),
        a5: (f, b2, b3),
        a6: (f, b3),
        a7: (f, b3, b4),
        a8: (f, b4),
        a9: (f, b4),
    }
    tasks = dummy_dsk_to_taskstate(dsk)
    assert isinstance(tasks, list)
    cogroups = coassignment_groups(tasks)
    assert_disjoint_sets(cogroups)
    assert len(cogroups) == 0


def test_deep_bases_win_over_dependents(abcde):
    r"""
    It's not clear who should run first, e or d

    1.  d is nicer because it exposes parallelism
    2.  e is nicer (hypothetically) because it will be sooner released
        (though in this case we need d to run first regardless)

    Regardless of e or d first, we should run b before c.

            a
          / | \   .
         b  c |
        / \ | /
       e    d
    """
    a, b, c, d, e = abcde
    dsk = {a: (f, b, c, d), b: (f, d, e), c: (f, d), d: (f,), e: (f,)}

    tasks = dummy_dsk_to_taskstate(dsk)
    assert isinstance(tasks, list)
    cogroups = coassignment_groups(tasks)
    assert_disjoint_sets(cogroups)
    assert len(cogroups) == 1
    assert len(cogroups[0]) == 3


def test_base_of_reduce_preferred(abcde):
    r"""
             a3
            /|
          a2 |
         /|  |
       a1 |  |
      /|  |  |
    a0 |  |  |
    |  |  |  |
    b0 b1 b2 b3
      \ \ / /
         c

    """
    a, b, c, d, e = abcde
    dsk = {(a, i): (f, (a, i - 1), (b, i)) for i in [1, 2, 3]}
    dsk[(a, 0)] = (f, (b, 0))
    dsk.update({(b, i): (f, c) for i in [0, 1, 2, 3]})
    dsk[c] = (f,)

    tasks = dummy_dsk_to_taskstate(dsk)
    assert isinstance(tasks, list)
    cogroups = coassignment_groups(tasks)
    assert_disjoint_sets(cogroups)

    assert len(cogroups) == 1
    assert {ts.key for ts in cogroups[0]} == {
        c,
        (b, 0),
        (b, 1),
        (a, 0),
        (a, 1),
    }


def test_map_overlap(abcde):
    r"""
      b1      b3      b5
       |\    / | \  / |
      c1  c2  c3  c4  c5
       |/  | \ | / | \|
      d1  d2  d3  d4  d5
       |       |      |
      e1      e3      e5

    Want to finish b1 before we start on e5
    """
    a, b, c, d, e = abcde
    dsk = {
        (e, 1): (f,),
        (d, 1): (f, (e, 1)),
        (c, 1): (f, (d, 1)),
        (b, 1): (f, (c, 1), (c, 2)),
        (d, 2): (f,),
        (c, 2): (f, (d, 1), (d, 2), (d, 3)),
        (e, 3): (f,),
        (d, 3): (f, (e, 3)),
        (c, 3): (f, (d, 3)),
        (b, 3): (f, (c, 2), (c, 3), (c, 4)),
        (d, 4): (f,),
        (c, 4): (f, (d, 3), (d, 4), (d, 5)),
        (e, 5): (f,),
        (d, 5): (f, (e, 5)),
        (c, 5): (f, (d, 5)),
        (b, 5): (f, (c, 4), (c, 5)),
    }

    tasks = dummy_dsk_to_taskstate(dsk)
    assert isinstance(tasks, list)
    cogroups = coassignment_groups(tasks)
    assert_disjoint_sets(cogroups)
    assert len(cogroups) == 2

    assert {ts.key for ts in cogroups[0]} == {
        (e, 1),
        (d, 1),
        (c, 1),
        (c, 2),
        (d, 3),
        (d, 2),
        (e, 3),
    }
    assert {ts.key for ts in cogroups[1]} == {
        # Why is this not part of the group? Maybe linked to order output
        # (b, 5),
        (c, 5),
        (d, 5),
        (e, 5),
        (c, 4),
        (d, 4),
    }
    # Not all belong to a cogroup
    assert sum(map(len, cogroups.values())) != len(tasks)


def test_repartition():
    ddf = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-17",
        dtypes={"x": float, "y": float},
        freq="1d",
    )
    assert ddf.npartitions == 16
    ddf_repart = ddf.repartition(npartitions=ddf.npartitions // 2)
    dsk = ddf_repart.dask.to_dict()
    for k, _ in list(dsk.items()):
        if k[0].startswith("make-timeseries"):
            dsk[k] = (f,)
    tasks = dummy_dsk_to_taskstate(dsk)
    cogroups = coassignment_groups(tasks)
    assert_disjoint_sets(cogroups)

    assert len(cogroups) == ddf.npartitions // 2


def test_repartition_reduce(abcde):
    a, b, c, d, e = abcde
    a1, a2, a3, a4, a5, a6, a7, a8 = (a + i for i in "12345678")
    b1, b2, b3, b4 = (b + i for i in "1234")
    c1, c2, c3, c4 = (c + i for i in "1234")
    d1, d2 = (d + i for i in "12")

    dsk = {
        # Roots
        a1: (f,),
        a2: (f,),
        a3: (f,),
        a4: (f,),
        a5: (f,),
        a6: (f,),
        a7: (f,),
        a8: (f,),
        # Trivial reduce, e.g. repartition
        b1: (f, a1, a2),
        b2: (f, a3, a4),
        b3: (f, a5, a6),
        b4: (f, a7, a8),
        # A linear chain
        c1: (f, b1),
        c2: (f, b2),
        c3: (f, b3),
        c4: (f, b4),
        # Tree reduce
        d1: (f, c1, c2),
        d2: (f, c3, c4),
        e: (f, d1, d2),
    }
    tasks = dummy_dsk_to_taskstate(dsk)
    cogroups = coassignment_groups(tasks)
    assert_disjoint_sets(cogroups)

    assert all(
        [1 == sum([ts.key.startswith("b") for ts in gr]) for gr in cogroups.values()]
    )
    assert all(
        [1 == sum([ts.key.startswith("c") for ts in gr]) for gr in cogroups.values()]
    )
    assert all(
        [2 == sum([ts.key.startswith("a") for ts in gr]) for gr in cogroups.values()]
    )
    assert len(cogroups) == 4


def test_vorticity(abcde):
    # See https://gist.github.com/TomNicholas/fe9c6b6c415d4fa42523216c87e2fff2
    # https://github.com/dask/distributed/discussions/7128#discussioncomment-3910328
    a, b, c, d, e = abcde
    d1, d2, d3, d4, d5, d6, d7 = (d + i for i in "1234567")
    e1, e2, e3, e4, e5 = (e + i for i in "12345")

    def _gen_branch(ix):
        return {
            f"a{ix}": (f,),
            f"b{ix}": (f, f"a{ix}"),
            f"c{ix}": (f, f"a{ix}"),
            f"d{ix}": (f, f"b{ix}", f"c{ix}"),
        }

    dsk = {}
    for ix in range(1, 8):
        dsk.update(_gen_branch(ix))

    dsk.update(
        {
            e1: (f, d1, d2, d3, d4, d5),
            e2: (f, d3, d4, d5, d6, d7),
        }
    )
    from dask.base import visualize

    visualize(dsk, color="order")
    ordered = dask.order.order(dsk)
    tasks = dummy_dsk_to_taskstate(dsk)
    cogroups = coassignment_groups(tasks)
    assert_disjoint_sets(cogroups)

    assert len(cogroups) == 7
    assert all(len(group) == 4 for group in cogroups.values())
    assert not any(e1 in group or e2 in group for group in cogroups.values())
