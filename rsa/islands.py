"""Island logic for RSA: validation, merge scheduling, island boundaries."""

import math
from typing import List


def validate_island_params(M: int, N: int, K: int, T: int):
    """Validate island parameters.

    Args:
        M: Number of islands (must be power of 2)
        N: Population per island
        K: Candidates sampled per aggregation (must be <= N)
        T: Total RSA steps/loops

    Raises:
        ValueError on invalid configs.
    """
    if M < 1:
        raise ValueError(f"M (islands) must be >= 1, got {M}")
    if N < 1:
        raise ValueError(f"N (population) must be >= 1, got {N}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if K > N:
        raise ValueError(f"K ({K}) must be <= N ({N}), the initial island size")
    if T < 1:
        raise ValueError(f"T (loops) must be >= 1, got {T}")
    if M == 1:
        return
    if M & (M - 1) != 0:
        raise ValueError(f"M (islands) must be a power of 2, got {M}")
    num_merges = int(math.log2(M))
    num_phases = num_merges + 1
    if T % num_phases != 0:
        raise ValueError(
            f"T (loops={T}) must be divisible by log2(M)+1 = {num_phases} "
            f"(M={M}, log2(M)={num_merges}). "
            f"Valid T values: {', '.join(str(num_phases*i) for i in range(1, 6))}, ..."
        )


def get_num_islands(step: int, M: int, T: int) -> int:
    """Return the number of islands at a given step."""
    if M == 1:
        return 1
    num_merges = int(math.log2(M))
    phase_len = T // (num_merges + 1)
    phase = min(step // phase_len, num_merges)
    return M // (2 ** phase)


def get_merge_schedule(M: int, T: int) -> List[dict]:
    """Return a list of merge events: [{at_step, from, to}, ...]."""
    if M <= 1:
        return []
    num_merges = int(math.log2(M))
    phase_len = T // (num_merges + 1)
    events = []
    current = M
    for i in range(num_merges):
        merge_step = (i + 1) * phase_len
        new_count = current // 2
        events.append({"at_step": merge_step, "from": current, "to": new_count})
        current = new_count
    return events
