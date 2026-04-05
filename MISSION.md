# MISSION

`trading-lab` exists to provide a contract-first backtesting framework whose strategy logic is modular, whose execution logic is centralized, and whose analytics are ledger-driven.

## Architectural Source Of Truth

- the framework is contract-first
- the engine is the sole owner of execution, accounting, and evaluation
- canonical ledgers are the only source for analytics and reporting
- entry, exit, and risk logic live in separate node modules
- unsupported node behavior must fail loudly
- unsupported complexity must not silently degrade
- backward-compatible evolution is preferred
- future complexity must be introduced through declared capabilities, not silent engine assumptions

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

## Experiment Layer Boundary

The next layer above the engine is experiment orchestration.

Rules:

- the engine remains responsible only for simulating one variant and one fold correctly
- experiment orchestration, fold handling, holdout handling, search, pruning, and reporting live above the engine
- variant combinations are built by the experiment layer, not by nodes
- entry, exit, and risk nodes do not generate experiment grids or search combinations
- all concrete strategy combinations must be represented as stable `variant_id` values
- holdout data must never be used for search ranking, optimizer objectives, or model selection
- deep dive outputs must be selected by stable `variant_id` and remain fold-aware
- deep dive price entry and exit plots should default to one selected fold or holdout, not the full dataset
- search must support both `max_variants` and `max_runtime_seconds`
- pruning is allowed, but every prune reason must be logged and reproducible
- summary comparison metrics are exported to Excel from the experiment layer, not from private engine state

Preferred module boundary for this layer:

- `experiments.py`
- `runner.py`
- `reporting.py`
- `search.py`

## Current V1 Supported Behavior

- OHLCV input
- market orders only
- next-bar-open fills
- single position per symbol
- no pyramiding
- long and short support
- fees and simple slippage
- one central event loop

## Future Reserved Behavior

The first contract boundary must reserve space for:

- multiple entries
- partial exits
- scale-in behavior
- same-bar reversal
- stop, limit, and bracket order workflows
- multiple lots per symbol
- target-position mode
- portfolio allocators
- richer risk actions
- engine capability checks
- compatibility validation hooks
- metric dependency checks
- future lot-level accounting
- advanced metrics and report slices

These are reserved extension paths, not implied runtime support.

## Engine Ownership

The engine is the sole owner of:

- action resolution
- order generation
- fill simulation
- position updates
- accounting
- result production

Nodes do not place trades directly and do not own accounting.

## Canonical Ledgers

Canonical ledgers are the durable base layer for:

- metrics
- slicing and attribution
- walk-forward summaries
- robustness testing
- bootstrap analysis
- live-vs-backtest comparison

If a metric cannot be derived from canonical ledgers, the required base data belongs in a ledger rather than private engine state.

## Node Model

Entry, exit, and risk are separate node categories. New strategy logic must be added as new node modules rather than by editing the engine.

Every node should be treated as having a manifest/spec boundary that declares:

- name
- kind
- version
- contract version
- required capabilities
- emitted action types
- required history depth
- whether portfolio context is required

## New Node Vs New Engine Feature

A new node:

- changes strategy logic inside existing engine contracts
- does not require new execution semantics
- does not require new accounting semantics
- does not require new ledger semantics

A new engine feature:

- changes execution, fills, accounting, exposure, or evaluation semantics
- introduces new action types the engine must understand
- requires new compatibility rules or new canonical ledger fields
- is required whenever a node asks for behavior outside declared engine capabilities
- includes experiment orchestration behavior that changes fold handling, holdout policy, ranking, pruning, or reporting semantics above the engine

## Capability And Compatibility Principle

The framework should evolve through explicit capability declarations.

Required direction:

- the engine should declare supported capabilities
- nodes should declare required capabilities
- unsupported requirements must fail compatibility checks loudly
- diagnostics should explain when the missing behavior is an engine feature request rather than a node-only change
- the engine must not silently reinterpret unsupported requests
