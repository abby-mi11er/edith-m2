# Edith Architecture (v1)

This document defines the target runtime architecture for Edith and the current contracts to keep the app stable while features evolve.

## Design Goals

- Grounded answers with reliable citations.
- Local-first operation with explicit cloud opt-in.
- Tool execution behind a single policy gate.
- Consumer-grade desktop UX: simple defaults, advanced controls hidden.

## Runtime Components

1. UI layer
- Chat, Library, Reader, Runs, PhD OS, Settings.
- Displays state and captures user intent.
- Does not bypass policy or retrieval contracts.

2. Orchestration layer
- Classifies intent.
- Builds run settings.
- Routes to retrieval + synthesis pipeline.
- Enforces refusal and support-audit outcomes.

3. Retrieval layer
- Contract: `retrieve(query, scope, mode) -> sources[]`.
- Backends: Google File Search Store, Local Chroma.
- Optional controls: rewrite, rerank, diversity, breadth/depth routing.

4. TAL (Tool Access Layer)
- Single gate for write/web/scheduled actions.
- Enforces allowlist, mode policy, and capability tokens.
- Emits audit records for every tool execution.

5. Quality/safety layer
- Source gate.
- Multi-pass planning/answering.
- Recursive controller (long-context map/reduce).
- Claim-level support audit.
- Citation validation and sentence provenance.

6. Persistence layer
- App state in `~/Library/Application Support/Edith` by default.
- Run ledger, notebook, claim inventory, timeline, graph artifacts, cache.
- Index manifest and status as source of indexing truth.

## Policy Invariants

- Files-only mode cannot silently use web.
- `require_citations=true` cannot return unsupported factual claims.
- Action outputs requiring mutation must pass approval guardrails.
- Tools must execute through TAL, never directly from UI handlers.

## Data Flows

### Ask flow

1. User query enters orchestration.
2. TAL checks mode/capabilities.
3. Retrieval runs (possibly rewritten/distilled query).
4. Synthesis pipeline runs (multi-pass and/or recursive controller).
5. Support audit validates claims.
6. UI renders answer + evidence + citations.
7. Run record saved for replay/eval.

### Ingest flow

1. Files are discovered from library folder.
2. Hash/dedupe + metadata extraction.
3. Chunk + upload/embed by selected backend.
4. Manifest/index status updated.
5. UI status reflects queue and last run.

## Operational Guidance

- Keep repository code outside synced cloud folders for active development.
- Keep library content in a stable local folder; sync that folder separately if needed.
- Use one backend contract and keep backend-specific logic behind adapters.
- Add features by extending orchestration/TAL contracts first, then UI.

## Current v1 Long-Context Controller

- `Recursive controller v1` runs map/reduce across source batches.
- Guardrails:
  - minimum source threshold
  - bounded depth
  - bounded batches
  - bounded total model calls (`EDITH_RECURSIVE_MAX_CALLS`)
- On budget exhaustion or thin evidence, controller exits safely and main pipeline continues.
