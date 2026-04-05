# AGENTS

This file defines contributor and Codex rules for `trading-lab`.

## Working Mode

- build in phases only
- keep the repo runnable after each phase
- add or update tests with each implementation phase
- avoid speculative abstractions
- prefer explicit contracts and validation over implicit flexibility

## Architecture Rules

- do not modify engine semantics just to add a new signal
- new entry, exit, and risk logic must be added as separate node modules
- all nodes must be registered through a registry
- all nodes must declare a manifest/spec and capability requirements
- any node that requires unsupported engine behavior must fail compatibility checks
- analytics must consume canonical ledgers only, never private engine state
- unsupported complexity must not silently degrade into simpler v1 behavior
- if a node requires behavior outside declared engine capabilities, treat it as an engine feature request, not just a new node

## Core Rules

Rule 1

A new entry, exit, or risk idea must be implemented as a separate node module.

Rule 2

A node may not bypass engine contracts or write directly to orders, fills, trades, equity, or metrics.

Rule 3

If a node requires behavior not present in declared engine capabilities, that is an engine feature request, not a normal node addition.

Rule 4

All analytics must be computed from canonical ledgers only.

Rule 5

The contract layer must remain backward-conscious. New complexity should be introduced through declared capabilities, reserved fields, and versioned contracts.

Rule 6

Avoid script sprawl. Prefer a compact core:

- `contracts.py`
- `schemas.py`
- `data.py`
- `registry.py`
- `engine.py`
- `analytics.py`
- `nodes/`

Rule 7

Avoid giant files. If a core module becomes too large due to coherent new concerns, split only when justified by responsibility, not preemptively.

## Experiment Rules

- the engine must continue to simulate one variant and one fold correctly; do not move experiment orchestration into the engine
- experiment orchestration, fold handling, holdout handling, search, pruning, and reporting belong above the engine
- variant combinations must be built by the experiment layer, not by entry, exit, or risk nodes
- all concrete strategy combinations must be tracked by stable `variant_id` values
- holdout must never be used for search ranking or optimizer objectives
- deep dive outputs must be selected by `variant_id` and must remain fold-aware
- deep dive price entry and exit plots should default to one selected fold or holdout, not the full dataset
- search implementations must support both `max_variants` and `max_runtime_seconds`
- pruning is allowed only when every prune reason is logged and reproducible
- summary comparison metrics should be exported to Excel from the reporting layer

## File Layout Rules

- avoid too many scripts and tiny files
- do not create giant scripts or giant modules
- combine closely related core logic into a few coherent modules
- keep node implementations separate so they are easy to track independently
- prefer the experiment layer to remain compact and explicit:
  `experiments.py`, `runner.py`, `reporting.py`, `search.py`

## Contract Rules

- nodes must consume and produce only declared contract objects
- nodes must not mutate shared state
- nodes must not write directly to orders, fills, trades, equity, or analytics tables
- unsupported behavior must fail loudly, not degrade silently
- strict or audit diagnostics should explain when the missing behavior belongs in the engine boundary

## Extension Boundary

- a new node is not a new engine feature
- if a requested behavior changes execution, accounting, order lifecycle, fill semantics, position model, or ledger structure, treat it as an engine feature
- if a requested behavior needs capabilities such as partial exits, scale-in flows, same-bar reversal, stop/limit/bracket orders, multiple lots, target-position mode, or portfolio allocation, treat it as an engine feature request first
- if a requested behavior only changes strategy logic within existing engine contracts, treat it as a node
- future extensions should be capability-driven, not assumption-driven
