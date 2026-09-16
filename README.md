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
| Storage (SQLite + local artifacts) | ✅ implemented, tested |
| Recon scout → app map | ✅ implemented, tested |
| Test plan synthesis with risk ranking | ✅ implemented, tested |
| API execution lane (checks + runner) | ✅ implemented, tested |
| Oracle signals → verdict | ✅ implemented, tested (deterministic signals only) |
| Triage → deduplicated findings | ✅ implemented, tested |
| Benchmark scoring (recall / precision) | ✅ implemented, tested |
| CLI (`doctor`, `recon`, `run`, `serve`) | ✅ implemented |
| Live dashboard | ✅ implemented |
| Guinea pig app + mutant registry | ✅ built, run manually |
| Browser lane (Playwright) | ⏳ not started |
| LLM-as-judge oracle signal | ⏳ not started — no reachable provider here |

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
#    `dev` and `api` are optional extras, so both must be named explicitly.
uv sync --extra dev --extra api

# 2. Configure providers
cp .env.example .env
#    then edit .env and set GEMINI_API_KEY and/or NVIDIA_API_KEY

# 3. Run the test suite
uv run pytest
```

## Watch a run live

The CLI prints one report when a run finishes. To watch a run *happen*:

```bash
uv run crucible serve            # dashboard on http://localhost:8000
```

Enter a URL, press **Start scan**, and the page streams every event as it
occurs — each page crawled, each check run, each invariant violated. This
matters because reconnaissance against a real application takes tens of seconds
and issues dozens of requests, and a terminal that prints nothing until the end
cannot distinguish *working* from *wedged*.

A run can also be driven headlessly: `POST /api/runs` starts one and returns a
stream id, and `GET /api/runs/{id}/events` is a server-sent event feed of the
same events, so another front end can be built on it.

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
├─ execute/  # API check lane: client, checks, runner (Playwright lane pending)
├─ oracle/   # independent verification signals -> verdict
├─ api/      # live dashboard: run registry, SSE stream, single-page UI
└─ cli.py    # command line entry point
```

## Adding a provider

The router is provider-agnostic by design, because free tiers change monthly.
Adding one means editing `ProviderName` and `CLEARANCES` in
`llm/sensitivity.py`, adding a branch to `Settings.provider_config`, and adding
it to the relevant entries in `TIER_PREFERENCE`. No call site changes.
