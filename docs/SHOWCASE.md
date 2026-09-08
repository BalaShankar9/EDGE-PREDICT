# Sharp-Edge · project guide

A Python sports prediction research system with collectors, feature pipelines, models, evaluation tools and a web interface.

**For:** Developers exploring sports data engineering and probabilistic modelling.<br>
**Current stage:** Research implementation · live pipeline validation pending<br>
**Reviewed:** 8 September 2026, from repository files and available GitHub workflow records. This is a source review, not a fresh application test or production certification.

## Start with the evidence

- [src/sharpedge/collectors](../src/sharpedge/collectors)
- [src/sharpedge/ml](../src/sharpedge/ml)
- [src/sharpedge/backtest](../src/sharpedge/backtest)
- [src/sharpedge/execution](../src/sharpedge/execution)
- [tests](../tests)

## A useful first demo

Run a fixed historical dataset through a documented model and compare predictions with a baseline on a strictly later time window. Show missing-data and stale-source behaviour.

## Next release checklist

These are proposed acceptance gates. An unchecked item does not imply its implementation is absent; it means fresh release evidence is still needed.

- [ ] Restore GitHub runner availability, then validate the scheduled data and retraining workflows and add separate CI that does not require live provider credentials.
- [ ] Lock a reproducible environment, verify model dependency declarations and create a small offline dataset with provenance.
- [ ] Publish time-separated evaluation with calibration, uncertainty and data exclusions; keep research results distinct from trading performance.

## What to measure

Out-of-time Brier score and calibration against a named baseline, with sample size and date range.

Publish the dataset or evaluation method, date range, sample size and limitations with each result. Code size, feature counts and agent counts do not measure product usefulness.

## What a finished showcase contains

Reproducible research notebook or command, source provenance and held-out evaluation results.

Keep one dated release record containing the commit, setup steps, required services, checks run, known limitations and rollback instructions. Add screenshots from that version using fictional or consented data; identify demo fixtures clearly.

## Three ways to evaluate this project

| Visitor | Start here | Evidence to look for |
| --- | --- | --- |
| Potential client | The demo scenario above | A repeatable workflow and a measurable outcome |
| Engineering team | Linked source and tests | Design decisions, failure handling and reproducibility |
| Product user or collaborator | README setup and release notes | A supported journey, current limitations and feedback route |

[Repository overview](../README.md) · [Issues](https://github.com/BalaShankar9/Sharp-Edge/issues) · [More projects](https://github.com/BalaShankar9)
