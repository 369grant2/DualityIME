"""What every GPU loop in tools/lm calls once per batch or step.

Batches of changing shapes (texts of every length, contexts of every length) make torch's cache keep blocks of every
size until the 24 GB are reserved; Windows' driver then spills into shared system memory instead of failing, and the
job crawls (the GPU "100% busy" at about 130 W) without a word. 2026-11-14 it stopped teacher_train.py at length 44 and
distill.py at step 26500, each unnoticed for half an hour. tidy() hands the cache back before that point.
(PYTORCH_CUDA_ALLOC_CONF=expandable_segments does nothing on Windows: torch says it is not supported.)
"""

LIMIT = 0.85  # (share of the GPU's memory torch may keep reserved before tidy() empties its cache)
IDLE = 0.10   # (...and only when at least this share of it is cached but unused: a big job's own working set stays)


def note():
    """For a progress line: what this process's torch holds, " mem 4.5/6.1 GiB" (peak allocated since the last note /
    reserved now). The GPU's whole memory full while these stay small = another program, not this job."""
    import torch
    if not torch.cuda.is_available():
        return ""
    peak, reserved = torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.memory_reserved() / 2 ** 30
    torch.cuda.reset_peak_memory_stats()
    return f" mem {peak:.1f}/{reserved:.1f} GiB"


def tidy(limit=LIMIT, idle=IDLE):
    """Empty torch's CUDA cache when what it reserves passes `limit` of the GPU's memory and at least `idle` of it is
    more than the job has needed at its peak (a job whose own step needs most of the GPU is left alone: emptying every
    step would only slow it - /code-review 2026-11-15). tidy() runs between steps, when the step's own memory is
    already freed, so the peak (torch.cuda.max_memory_allocated: since note() last reset it, else since the start)
    tells what the job needs, not what is allocated now. True when it emptied it."""
    import torch
    if not torch.cuda.is_available():
        return False
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved()
    if reserved <= limit * total or reserved - torch.cuda.max_memory_allocated() < idle * total:
        return False
    torch.cuda.empty_cache()
    return True
