# Tidal — preprint / venue submission package

Everything below is paste-ready for a preprint server (Zenodo; TechRxiv when it
reopens) or a conference submission portal. Keep this file in sync with
`latex/main.tex` — the abstract here must match the compiled PDF verbatim
(minus LaTeX markup).

## Files

| Purpose | File |
|---|---|
| Preprint PDF (upload this) | `docs/paper/tidal-paper-arxiv.pdf` (21 pp) |
| IEEE conference version | `docs/paper/ieee/main-ieee.pdf` (15 pp, IEEEtran) |
| LaTeX sources | `docs/paper/latex/`, `docs/paper/ieee/` |

## Title

Tidal: A Deadline-Contracted Batch Tier for Unmodified LLM Serving Engines

## Author

Gurunath Lunkupali Venugopal — gurunathrajagopal@gmail.com

## Abstract (plain text)

A serving node sized for interactive traffic runs far below its throughput
ceiling most of the time: our online-only GPU baseline emits a third of its
node's offline ceiling, and scheduler budget left unspent in an iteration is
gone. Three capacities order the node: its throughput saturated with offline
work, co-serving batch behind live traffic, and serving live traffic alone. A
batch tier can lift the node from the third toward the first without forking
the engine. Commercial providers sell that gap as a second product, a batch API
at half price with a 24-hour completion window; self-hosted deployments have
none, and the systems that harvest idle capacity (HyGen, ConServe) fork the
engine and promise no completion time. Tidal is that contract over unmodified
vLLM: an OpenAI-wire /v1/batches front end, admission that refuses windows the
system's own observed rate cannot meet, and per-job least-laxity escalation. On
A6000 nodes serving Qwen2.5-7B, a gateway sidecar recovers 69.3% of the node's
steady-state offline ceiling at 1.18x/1.23x online p50/p99 (flat 20 req/s,
cross-node); under a diurnal trace at 12.1 req/s mean offered load it recovers
78.1%. We also implement the identical policy inside the engine as a scheduler
plugin and compare both placements. Code and evidence are open source.

## Keywords

LLM serving; batch processing; vLLM; deadline scheduling; least laxity;
GPU utilization; co-serving; cloud computing; inference systems; SLA

## Subject categories

- Preprint servers: Computer Science → Distributed / Cluster / Cloud Computing
  (arXiv: cs.DC; secondary cs.LG or cs.PF)
- IEEE taxonomy: Cloud computing; Resource management; Scheduling algorithms

## License (preprint)

CC BY 4.0 (retains copyright; permitted by IEEE preprint policy for later
venue submission)

## Related identifiers

- Code + evidence: https://github.com/rajagurunath/tidal
- Companion paper: LazyCode (batch-tier client; cite once it has a DOI)

## Notes / disclosures

- AI-assistance disclosure is already in the IEEE version's Acknowledgment
  section; include the same sentence in any venue form that asks.
- No funding; no conflicts of interest.
- Target IEEE venues when CFPs open (~late 2026 / early 2027):
  IC2E 2027, IEEE CLOUD 2027. Check each CFP for double-blind rules before
  submitting — the preprint is normally allowed, but the submitted PDF may
  need de-anonymization removed/added accordingly.
