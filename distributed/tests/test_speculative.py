import asyncio

from dask import delayed

from distributed.utils_test import gen_cluster, inc, dec, slowinc


@gen_cluster(client=True, config={"dask.optimization.fuse.active": False})
async def test_speculative_assignment_simple(c, s, a, b):
    x = delayed(slowinc)(1)
    y = delayed(inc)(x)
    z = delayed(dec)(y)

    result = c.compute(z)
    while not a.tasks:
        await asyncio.sleep(0.001)

    assert x.key in a.tasks
    assert y.key in a.tasks
    assert z.key in a.tasks

    assert a.tasks[y.key].state == "speculative"
    assert a.tasks[z.key].state == "speculative"

    assert await result == 2


@gen_cluster(client=True, config={"dask.optimization.fuse.active": False})
async def test_spec_assign_all_dependencies(c, s, a, b):
    x1 = await c.scatter([1], workers=a.address)
    x2 = await c.scatter([2], workers=a.address)
    # not spec assigned: data already present
    x1 = delayed(inc)(x1[0])
    # not spec assigned: data already present
    x2 = delayed(inc)(x2[0])
    # spec assigned (two dependencies on same worker)
    x1x2 = x1 + x2
    # spec assigned
    z = delayed(dec)(x1x2)

    zz = c.compute(z)
    result = await zz
    assert result == 4
    assert (x1.key, "waiting", "ready") in a.story(x1.key)
    assert (x2.key, "waiting", "ready") in a.story(x2.key)
    assert (x1x2.key, "speculative", "ready") in a.story(x1x2.key)
    assert (z.key, "speculative", "ready") in a.story(z.key)


@gen_cluster(client=True, config={"dask.optimization.fuse.active": False})
async def test_spec_assign_intermittent(c, s, a, b):
    """
        d
       / \
      e   h   # no spec
      |   |
      f   i   # both spec
      |   |
      g   j   # both spec
       \ /                                 # noqa
        k     # spec

    """

    d = await c.scatter([1])

    e = delayed(inc)(d[0])
    f = delayed(inc)(e)
    g = delayed(inc)(f)

    h = delayed(dec)(d[0])
    i = delayed(dec)(h)
    j = delayed(dec)(i)

    k = j + g

    l = c.compute(k)
    result = await l
    assert result == 2

    if b.story(h.key):
        worker = b
    else:
        worker = a

    assert (h.key, "waiting", "ready") in worker.story(h.key)
    assert (i.key, "speculative", "ready") in worker.story(i.key)
    assert (j.key, "speculative", "ready") in worker.story(j.key)

    if a.story(e.key):
        worker = a
    else:
        worker = b

    assert (e.key, "waiting", "ready") in worker.story(e.key)
    assert (f.key, "speculative", "ready") in worker.story(f.key)
    assert (g.key, "speculative", "ready") in worker.story(g.key)

    if a.story(k.key):
        assert (k.key, "speculative", "ready") in a.story(k.key)
    if b.story(k.key):
        assert (k.key, "speculative", "ready") in b.story(k.key)
