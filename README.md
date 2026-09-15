# Crucible

An autonomous multi-agent QA platform. Agents recon an application under test,
synthesize a risk-weighted test plan, execute it against a real browser and API
surface, judge whether the results are genuine bugs, and triage what they find.

This is an internal tool. It is not multi-tenant and does not bill anyone.

## Status

Phase 1, in progress. What exists today:

| Component | State |
|---|---|
| Data classification + provider clearance | ✅ implemented, tested |
| Provider-agnostic model router with fail-closed escalation | ✅ implemented, tested |
| Run budgets with hard stops | ✅ implemented, tested |
| Cost ledger with per-agent attribution | ✅ implemented, tested |
| Structured logging (text + JSON) | ✅ implemented |
| Storage (SQLite + local artifacts) | ⏳ not started |
| Recon scout → app map | ⏳ not started |
| Execution fabric (Playwright) | ⏳ not started |
| Oracle signals → verdict | ⏳ not started |
| Guinea pig app + mutant injector | ⏳ not started |
| CLI | ⏳ not started |

## Why this exists

Most "AI testing" tools are a language model writing selectors for a
conventional test runner. The interesting problem is the *oracle*: deciding
whether an observed behaviour is a bug, without a human adjudicating every
result. Everything here is arranged around that problem, including refusing to
let a weak model authorise a verdict.

## Quickstart

Prerequisites: Python 3.12+ (3.14 in use here), and `uv`.

```bash
# 1. Create the virtualenv and install the foundation
uv sync --extra dev

# 2. Configure providers
cp .env.example .env
#    then edit .env and set GEMINI_API_KEY and/or NVIDIA_API_KEY

# 3. Run the test suite
uv run pytest
```

## Model providers

Three tiers, chosen per call by cost and stakes:

| Tier | Provider | Used for | Cleared data classes |
|---|---|---|---|
| `CHEAP` | Google Gemini (free tier) | Crawling, summarization, test generation | `PUBLIC` only |
| `FRONTIER` | NVIDIA NIM | Verdicts, critic, debugger | `PUBLIC`, `PROPRIETARY` |
| `LOCAL` | Ollama (optional) | Offline plumbing | `PUBLIC`, `PROPRIETARY` |

Tier 1 is free, which is the only reason it is usable here — and is also why it
cannot be trusted with real source code. Google's free tier may use prompts to
improve its products. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for how that is
enforced in code rather than by convention.

## Layout

```
src/crucible/
├─ core/     # configuration, budgets, logging, event bus
├─ llm/      # sensitivity gate, provider router, cost ledger
├─ store/    # SQLite models, local artifact store
├─ recon/    # scout: crawl the target, build the app map
├─ plan/     # test case synthesis and risk ranking
├─ execute/  # Playwright runner, record/replay
├─ oracle/   # independent verification signals -> verdict
└─ cli.py    # command line entry point
```

## Adding a provider

The router is provider-agnostic by design, because free tiers change monthly.
Adding one means editing `ProviderName` and `CLEARANCES` in
`llm/sensitivity.py`, adding a branch to `Settings.provider_config`, and adding
it to the relevant entries in `TIER_PREFERENCE`. No call site changes.
