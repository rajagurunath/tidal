# Tidal — preprint / venue submission package

**PUBLISHED preprint v2.3: DOI [10.5281/zenodo.22067989](https://doi.org/10.5281/zenodo.22067989) (Zenodo, 2026-08-23, CC BY 4.0). Concept DOI (all versions): 10.5281/zenodo.21905461; v2.2 was 10.5281/zenodo.21967536, v2.1 was 10.5281/zenodo.21905462.**

Everything below is paste-ready for a preprint server (Zenodo; TechRxiv when it
reopens) or a conference submission portal. Keep this file in sync with
`latex/main.tex` — the abstract here must match the compiled PDF verbatim
(minus LaTeX markup).

## Files

| Purpose | File |
|---|---|
| Preprint PDF (upload this) | `docs/paper/tidal-paper-arxiv.pdf` (26 pp) |
| IEEE conference version | `docs/paper/ieee/main-ieee.pdf` (15 pp, IEEEtran) |
| LaTeX sources | `docs/paper/latex/`, `docs/paper/ieee/` |

## Title

Tidal: A Deadline-Contracted Batch Tier for Unmodified LLM Serving Engines

## Author

Gurunath Lunkupali Venugopal — gurunathrajagopal@gmail.com

## Abstract (plain text)

**(v2.3 — published 2026-08-23. Abstract unchanged from v2.2; v2.3 adds the terminology convention and batching-lineage figure, fixes appendix float placement, trims Appendix A provenance.)**

A serving node sized for interactive traffic spends most of its life well below
its throughput ceiling. On our GPU testbed, a vLLM node serving only online
chat traffic emitted about a third of the output tokens per second that the
same node sustains when saturated with offline work — and capacity left unspent
in one scheduler iteration cannot be banked for the next. Commercial providers
sell exactly this slack as a second product: a batch API at half the online
price with a 24-hour completion window. A self-hosted deployment has no
equivalent product. vLLM's own batch tooling can drain an offline workload at
full speed, but only by dedicating the machine to it, and the research systems
that do harvest slack behind live traffic (HyGen, ConServe) fork the engine and
promise no completion time — no deadline a customer could plan around. Tidal
supplies the missing product over unmodified vLLM: an OpenAI-compatible
/v1/batches front end, an admission test that refuses any deadline window the
system's own observed completion rate cannot meet, and per-job escalation
governed by laxity — the classical real-time measure of how much time a job can
still afford to lose before its deadline is at risk. On RTX A6000 nodes serving
Qwen2.5-7B under a flat 20 req/s online load, a gateway sidecar recovers 69.3%
of the node's steady-state offline ceiling while online request latency rises
to 1.18x a matched node's online-only baseline at the median and 1.23x at the
99th percentile; under a diurnal trace at 12.1 req/s mean offered load it
recovers 78.1%. We also implement the identical policy inside the engine as a
scheduler plugin and compare the two placements of one policy — outside the
engine versus inside it — as an explicit design axis. Code and evidence are
open source.

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
