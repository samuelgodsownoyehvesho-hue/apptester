# Architecture

## The central problem

An agent that reports bugs is only useful if you can trust two numbers:
**recall** (of the bugs that exist, how many did it find?) and **precision**
(of the bugs it reported, how many were real?). Without ground truth, every
"the AI found a bug" demo is unfalsifiable.

So the repository ships its own target: a deliberately buggy application plus a
mutation injector that seeds known defects. Every run is scored against it.
The oracle subsystem is built before the dashboard, because the benchmark is
what makes the rest of the project meaningful.

## Pipeline

```
target ──▶ RECON ──▶ app_map ──▶ PLAN ──▶ test cases ──▶ EXECUTE
                                                             │
                                          artifacts ◀────────┘
                                                             │
                          ORAQUE (5 signals) ◀───────────────┘
                                                             │
                                        VERDICT ──▶ TRIAGE ──▶ finding
                                                             │
                                              CRITIC ◀───────┘
```

The pipeline is a state machine, not a free-form agent conversation. Agents run
*inside* nodes. A group chat with eight participants and no fixed topology
wanders, spends its budget on recrimination, and produces results that cannot
be reproduced.

## The oracle

A verdict requires agreement from independent signals, and the judge must cite
evidence or return `insufficient_evidence`:

| Signal | Mechanism | Strength |
|---|---|---|
| Spec | PRD / OpenAPI as the contract of record | Strong when specs exist |
| Differential | Same test against base ref vs HEAD | Strong; catches regressions |
| Metamorphic | Invariants that must hold (sort twice ≡ sort once) | Strongest; needs no spec |
| Visual | Screenshot / aria-tree diff against baseline | Catches presentation drift |
| Judge | LLM with a rubric, must cite DOM/network evidence | Necessary, not sufficient |

Disagreement between signals is recorded rather than discarded. That data is
what calibrates thresholds, and it is the only honest way to know whether the
oracle is improving.

## Data handling

The rule: **a payload derived from real source code, real application state, or
real customer data must never be sent to a provider that trains on its inputs.**

Encoded in `llm/sensitivity.py` as a clearance table, enforced by
`ModelRouter.select` *before a client is constructed*, so a violation would
have to be introduced in one place rather than at any of dozens of call sites.

Three design choices worth stating:

1. **Escalation is upward only.** When the preferred provider for a tier is not
   cleared, routing moves to a more constrained provider. No code path
   downgrades clearance to keep a call cheap.
2. **Secrets are cleared by nothing.** `DataClass.SECRET` raises for every
   provider including local inference. Credentials have no business in a
   prompt; this class exists so that a redaction failure is loud.
3. **The judge tier has no free fallback.** `TIER_PREFERENCE[FRONTIER]` omits
   Gemini deliberately. Silently judging with a different model would weaken
   every verdict while appearing to work.

`SensitivityViolation` subclasses `PermissionError` so that a broad
`except Exception` cannot swallow it.

## Budgets

Free tiers make budgets load-bearing. Documented daily caps are in the low
hundreds of thousands of tokens — roughly two or three unguarded agent runs
each — so a run that cannot stop itself cannot be iterated on.

`RunBudget` checks a conservative estimate *before* dispatch, records actuals
after, and latches `exhausted` if an estimate was too low, so the next call
stops rather than repeating the overshoot into a wall of 429s.

## Environment constraints

Measured with `scripts/net_check.py` on the development machine:

- **`generativelanguage.googleapis.com` is unreachable.** DNS resolves and TCP
  connects, but the TLS handshake fails with `UNEXPECTED_EOF_WHILE_READING`.
  Every other endpoint tested completes TLS, including `aistudio.google.com`
  and `build.nvidia.com`, so this is specific blocking of that hostname rather
  than a general network fault. Consequence: tier 1 is disabled and NVIDIA NIM
  currently carries both tiers. Re-run the probe on any new network before
  assuming portability.

## Open verification items

These are assumptions the current design depends on. Each must be confirmed
before the corresponding capability is trusted.

- [ ] **NVIDIA NIM training policy is unverified.** Free-tier access is granted
      for prototyping; the data-handling policy was not independently
      confirmed. `CLEARANCES[NVIDIA]` currently includes `PROPRIETARY` on this
      assumption. **Confirm before pointing the white-box triage path at a real
      repository.** If NIM does retain or train on inputs, NIM must be reduced
      to `PUBLIC` and the white-box path blocked pending a no-training
      provider.
- [ ] **NIM credit budget is unquantified.** Signup grants 1,000 inference
      credits (5,000 on request). Whether a credit corresponds to a request or
      a token volume is not documented anywhere we've found. Measure it with a
      single instrumented call and record the answer here.
- [ ] **Gemini free-tier rate limits are unpublished.** Google now shows limits
      per project in AI Studio rather than in the docs. `RateLimiter` uses a
      conservative local default; the real ceiling should be measured and
      recorded.
- [ ] **NVIDIA licensing.** Free catalogue access is limited to prototyping,
      research, development, and testing. Internal QA use is in scope; selling
      this as a product would require NVIDIA AI Enterprise licensing.

## Deferred decisions

Recorded so they are made deliberately later rather than by accident:

- **Docker.** Not installed. Phase 1 uses SQLite, the local filesystem for
  artifacts, and an in-process queue. Postgres + pgvector, Redis, and object
  storage arrive when a single-machine design actually becomes the constraint.
- **Multi-tenancy.** Out of scope. This is an internal tool; tenant isolation,
  billing, and hostile-input sandboxing are not designed for.
- **Mobile and desktop targets.** Out of scope. The execution fabric assumes a
  browser and an HTTP client.
- **Fix PRs.** Planned last, and never auto-merged. Branch plus pull request
  behind a recorded human approval.
