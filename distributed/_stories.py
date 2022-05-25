from __future__ import annotations

from typing import Iterable


def scheduler_story(
    keys_or_stimuli: set[str], transition_log: Iterable[tuple]
) -> list[tuple]:
    """Creates a story from the scheduler transition log given a set of keys
    describing tasks or stimuli.

    Parameters
    ----------
    keys_or_stimuli : set[str]
        Task keys or stimulus_id's
    log : iterable
        The scheduler transition log

    Returns
    -------
    story : list[tuple]
    """
    return [
        t
        for t in transition_log
        if t[0] in keys_or_stimuli or keys_or_stimuli.intersection(t[3])
    ]


def worker_story(keys_or_tags: set[str], log: Iterable[tuple]) -> list:
    """Creates a story from the worker log given a set of keys
    describing tasks or stimuli.

    Parameters
    ----------
    keys_or_tags : set[str]
        Task keys or arbitrary tags from the transition log, e.g. stimulus_id's
    log : iterable
        The worker log

    Returns
    -------
    story : list[str]
    """
    return [
        msg
        for msg in log
        if any(key in msg for key in keys_or_tags)
        or any(
            key in c
            for key in keys_or_tags
            for c in msg
            if isinstance(c, (tuple, list, set))
        )
    ]
