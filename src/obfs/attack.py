import random
from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean

from .bfs_plain import bfs
from .domain import (
    Adjacency,
    AttackMetrics,
    AttackResult,
    Burst,
    Fingerprint,
    FingerprintDatabase,
    Vertex,
)


# Spec §4.13, algorithm IZGRADI_BAZU_FINGERPRINTA. Without a budget the
# fingerprints are F_naive(s), with one they are F_mitigated(s) (Spec §3.2).
def build_fingerprint_database(
    adjacency: Adjacency, budget: int | None = None
) -> FingerprintDatabase:
    # Spec §4.13, step 1.
    database: defaultdict[Fingerprint, list[Vertex]] = defaultdict(list)

    # Spec §4.13, step 2.
    for s in range(len(adjacency)):
        order = bfs(adjacency, s).order
        if budget is None:
            fp = tuple(1 + len(adjacency[u]) for u in order)
        else:
            fp = (1 + budget,) * len(order)
        database[fp].append(s)

    # Spec §4.13, step 3.
    return dict(database)


# Spec §4.13, algorithm NAPADNI_UPIT.
def attack_query(
    observed: Sequence[Burst], database: FingerprintDatabase
) -> AttackResult:
    # Spec §4.13, steps 1-5.
    candidates = database[tuple(observed)]
    t = len(candidates)
    return AttackResult(list(candidates), t, 1 / t)


# Spec §4.13, algorithm IZRAČUNAJ_METRIKE_NAPADA.
def compute_attack_metrics(
    results: Sequence[AttackResult], database: FingerprintDatabase
) -> AttackMetrics:
    # Spec §4.13, steps 1-2.
    average_top1 = fmean(result.top1_credit for result in results)
    average_tie_size = fmean(result.tie_size for result in results)

    # Spec §4.13, steps 3-4.
    unique_candidates = sum(1 for group in database.values() if len(group) == 1)
    total_candidates = sum(len(group) for group in database.values())

    # Spec §4.13, step 5.
    return AttackMetrics(
        average_top1, average_tie_size, unique_candidates / total_candidates
    )


# Spec §5.2, step 2.
DEGREE_GROUPS = 10


# Spec §5.2, algorithm IZABERI_TESTNE_IZVORE.
def select_test_sources(
    adjacency: Adjacency, rng: random.Random, count: int = 100
) -> list[Vertex]:
    if not 0 <= count <= 100:
        raise ValueError(f"count must be in 0..100, got {count}")

    # Spec §5.2, step 1.
    ranked = sorted(range(len(adjacency)), key=lambda v: (len(adjacency[v]), v))

    # Spec §5.2, step 2.
    n = len(ranked)
    groups = [
        ranked[i * n // DEGREE_GROUPS : (i + 1) * n // DEGREE_GROUPS]
        for i in range(DEGREE_GROUPS)
    ]

    # Spec §5.2, steps 3-5. Up to count // 10 from each group, ten for the
    # default 100; smaller groups give all their vertices.
    per_group = count // DEGREE_GROUPS
    return [
        v for group in groups for v in rng.sample(group, min(per_group, len(group)))
    ]
