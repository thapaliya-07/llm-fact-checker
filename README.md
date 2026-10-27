# LLM Fact-Checker (with evaluation)

Takes a claim, gets the related Wikipedia article as evidence, and asks Claude (`claude-opus-5-5`)
to label it **Supported**, **Refuted**, or **Not Enough Info**. Then it measures accuracy on
30 hand-labeled claims, **with** evidence and **without** it.

## Setup

```bash
cd ~/fact-checker
source .venv/bin/activate          # already created; else: python3 -m venv .venv && pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...  # from console.anthropic.com
```

## Use

```bash
python factcheck.py "The Eiffel Tower is in Berlin."                 # searches Wikipedia for a page
python factcheck.py "The Eiffel Tower is in Berlin." --no-evidence   # model's own knowledge only
python evaluate.py --limit 3                                         # cheap trial run
python evaluate.py                                                   # full experiment
```

`evaluate.py` prints overall accuracy, accuracy per label, a confusion matrix, and each miss.
It saves per-claim CSVs and a summary to `results/`.

## Experiment design

| | |
|---|---|
| Dataset | `claims.json`: 30 claims, balanced 10 / 10 / 10 across the three labels |
| Evidence | Full plain-text English Wikipedia article for each claim's `page`, cached in `evidence_cache/` so every run uses identical text |
| Independent variable | Evidence given vs. not given (same model, same effort, same claims) |
| Metric | Accuracy against the gold labels (plus per-label accuracy) |

Each claim names its Wikipedia page. That holds retrieval fixed, so the experiment measures the
LLM's judgment and not search quality. The single-claim CLI searches Wikipedia when no page is given.

## Caveats to report honestly

- **n = 30 is small.** A single claim is worth 3.3 points of accuracy. Report the raw counts
  (for example 24/30), not just percentages.
- **The LLM isn't deterministic.** Run `evaluate.py` 2–3 times and report the mean and range.
- **The claims are easy, famous facts**, so the model often knows them without evidence. Expect
  the biggest difference in the *Not Enough Info* class. Harder, obscure, or recent claims would
  probably widen the gap. That would make a good follow-up.
- **The gold labels were written by hand.** NEI claims were checked by keyword search against
  their articles, but a quick manual review of `claims.json` is worthwhile.
- The "no evidence" prompt treats NEI as "can't be verified from widely known information". That
  definition is a design choice and should be stated in the write-up.

## Cost

About 500–600k input tokens per full run, almost all in the with-evidence condition (Wikipedia
articles average about 9k words). At Opus 5.5 pricing ($4 / M input) that is roughly $2–2.50 per run (measured: 3 claims ≈ 64k tokens).

## Results (first full run)

| Condition | Accuracy | Input tokens |
|---|---|---|
| With Wikipedia evidence | 29/30 (97%) | 574,212 |
| Without evidence | 30/30 (100%) | 11,526 |

The single miss (#28, Great Red Spot) is arguably a labeling issue: the Jupiter article says
models predict the storm "will be a permanent feature", which the model read as a refutation.
Per-claim outputs are in `results/`.

## Attribution

Evidence texts in `evidence_cache/` are from English Wikipedia and are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
