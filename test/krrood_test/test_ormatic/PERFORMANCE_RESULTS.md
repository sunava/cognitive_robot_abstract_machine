# ORMatic DAO Conversion — Performance Results

Measured with `test/krrood_test/test_ormatic/benchmark_dao_conversion.py`
(2026-06-11, in-memory SQLite, Python 3.12, 5 repetitions after warmup).

**Workload:** 2,200 domain objects / 400 conversion roots per repetition, built from the
krrood test dataset classes — a mix of scalar columns, single (one-to-one)
relationships, association-object collections, alternative mappings
(`EntityMapping`), and `__post_init__` back-references (`ContainerGeneration`).

Phases:

- `to_dao` — convert all domain roots to DAOs (shared `ToDataAccessObjectState`)
- `insert` — `session.add_all` + `commit`
- `query` — load all root DAOs back from the database
- `from_dao` — reconstruct all domain objects (shared `FromDataAccessObjectState`)

## Results (mean over 5 repetitions, milliseconds)

| phase    | before | after code opts (P1–P8) | after all (incl. P0) | total speedup |
|----------|-------:|------------------------:|---------------------:|--------------:|
| to_dao   |   55.2 |                    40.4 |                 41.1 |     **1.34×** |
| insert   |  215.6 |                   224.3 |                217.0 |         1.0×  |
| query    |   32.2 |                    28.4 |                 93.6 |         0.34× |
| from_dao |  619.4 |                   442.2 |                 42.6 |    **14.5×**  |

**End-to-end load path (`query` + `from_dao`): 651.6 ms → 136.2 ms = 4.8× faster.**

"before" is the state after the bug fixes from the review but before any
performance changes (commit state of the same session, so the comparison is
purely the optimizations).

## Where the time went (baseline profile)

A cProfile run of the baseline `from_dao` phase showed **~85 % of the time was
spent in SQLAlchemy lazy loads** (`_emit_lazyload`, ~1,600 single-row SELECTs):
the generated association objects' `target` relationship used the default
`lazy='select'`, so every collection item was fetched with its own query
(classic N+1). The remaining ~15 % was Python-level per-instance introspection
(mapper inspection, relationship classification, MRO walks, dataclass field
scans).

## Optimizations applied

| id | change | effect |
|----|--------|--------|
| P0 | `lazy='selectin'` on the association `target` relationship (`krrood/src/krrood/jinja_templates/sqlalchemy_model.py.jinja`). Eliminates the N+1 lazy loads; targets are now batch-loaded at query time. | `from_dao` 442 → 43 ms; `query` 28 → 94 ms (cost moved into batched queries, net 10× on the load path) |
| P1 | Per-DAO-class **conversion plan** (`_get_conversion_plan` in `dao.py`): cached tuples of data-column names, classified single/collection relationships with expected domain types and association classes, alternative-mapping base, plus the cached partition plan for alternatively mapped subclasses (`_get_alternative_partition_plan`). Replaces per-instance `sqlalchemy.inspect()`, `is_data_column` filtering, relationship classification, and MRO walks in both conversion directions. | bulk of the code-only gains: `to_dao` 55 → 40 ms, `from_dao` 619 → 442 ms together with P2–P8 |
| P2 | Cached allocation plan (`_get_allocation_plan` in `from_dao.py`): dataclass default values/factories computed once per class instead of per allocated instance. | part of code-only gains |
| P3 | Cached per-class set-field names (`_get_set_field_names`) and `__post_init__` presence (`_has_post_init`) instead of scanning all type hints / `hasattr` per object. | part of code-only gains |
| P4 | `convert_alternative_mappings_to_domain_objects` groups alternative-mapping instances by type once instead of scanning all instances per type (was O(types × instances)). | part of code-only gains |
| P5 | Parent/child column and relationship partition for alternatively mapped subclasses cached per (class, base) pair. | part of code-only gains |
| P6 | `TypeType.cache_ok = True` (was missing): SQLAlchemy no longer disables compiled-statement caching for every statement touching a `Type` column. | affects all queries involving `Type` columns (not isolated in this benchmark) |
| P7 | `get_python_type_from_sqlalchemy_column` no longer constructs two full `ORMatic` instances per call (cached default type mappings). | hot in `parametrization/feature_extractor.py`, not in this benchmark |
| P8 | Cached alternative-mapping class lookup in `apply_alternative_mapping_if_needed` (`to_dao.py`). | part of code-only gains |

## Trade-off note

P0 moves the relationship loading from `from_dao` (N+1 single-row SELECTs) to
`query` (a handful of batched `IN` SELECTs per relationship). The `query` phase
therefore appears ~3× slower, but the combined load path is 4.8× faster, and
the cost now scales with the number of relationship types instead of the number
of rows. Code that queries DAOs *without* reconstructing domain objects pays
the eager-loading cost too; if such a path becomes hot, switch specific
queries to `lazyload(...)` options rather than reverting the default.

## Reproducing

```bash
python test/krrood_test/test_ormatic/benchmark_dao_conversion.py \
    --groups 100 --positions 10 --repetitions 5
```

The "after code opts" column was measured by stripping `lazy="selectin"` from
the association `target` relationships in the generated interface; regenerate
the interface (any krrood pytest run does this) to restore the final state.

## Verification

Full ormatic test suite after all changes: **125 passed** (111 pre-existing
tests + 14 new bug-reproduction tests in `test_review_bugs.py`).
