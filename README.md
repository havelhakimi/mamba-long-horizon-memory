# Long-Horizon State Tracking with Mamba-1

**HackerRank Deep Learning Research Engineer take-home assignment**

This project investigates whether a compact selective state-space model can preserve, overwrite, and retrieve key-value information as event sequences become longer and contain more distracting updates. We reproduce a Mamba-1 baseline and evaluate a focused change to its state-decay mechanism.

## Task

Each input timestep is a `SET(key, value)`, `QUERY(key)`, or `NOOP` event. There are eight keys (A–H), four possible stored values (1–4), and a fifth output class, `0` (`UNSET`), for keys without a previous assignment. The model predicts the latest value only at `QUERY` positions. The input vocabulary contains 41 logical event tokens.

The provided generator creates 4,000 training sequences (lengths 32–128) and 500 sequences in each validation/test split. Evaluation includes length-128 in-distribution testing, length-256 and length-512 out-of-distribution testing, and two length-512 stress tests for retention and overwriting.

## Model and Research Modification

The baseline reuses the Mamba-1 backbone from [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal), with `d_model=32`, `n_layer=1`, `d_state=8`, `expand=2`, and `d_conv=4`. A linear five-class head replaces the original language-model prediction head.

The proposed **learned retention gate** scales the state-decay exponent without changing the input-write term:

$$
 h_t = \exp(r_t\Delta_t A)\odot h_{t-1} + \Delta_t B_tu_t,
 \qquad r_t=\sigma(W_ru_t+b_r).
$$

The modification is implemented in `model_retention.py`. We compare four variants: original Mamba-1, learned gate, gate disabled (`r_t=1`), and fixed gate (`r_t=0.5`).

## Experimental Setup

- **Optimizer:** AdamW; learning rate `0.001`; FP32.
- **Training:** 2,000 optimizer updates per run; batch size 16.
- **Evaluation:** Batch size 4; fixed datasets shared by all variants.
- **Random seeds:** 42, 43, 44.
- **Reported hardware:** Tesla T4 GPU in Google Colab.
- **Metrics:** Query accuracy, accuracy by SET-to-QUERY delay, overwrite count, and cross-key interference; parameters, runtime, and GPU memory.

## Main Results

Mean query accuracy across three seeds (in percent):

| Model | ID-128 | OOD-256 | OOD-512 |
|---|---:|---:|---:|
| Original Mamba-1 | 83.72 | 80.30 | 78.15 |
| Learned retention gate | **89.63** | **87.23** | **85.43** |
| Disabled gate (`r_t=1`) | 83.72 | 80.30 | 78.15 |
| Fixed gate (`r_t=0.5`) | 85.77 | 83.26 | 81.26 |

The learned gate improves OOD-512 query accuracy by **7.28 percentage points** over the baseline. However, on the dedicated **511-step long-gap probe**, both models remain near the 25% chance level: **24.60%** for the baseline and **22.87%** for the learned gate. The modification improves general state tracking but does **not** solve extreme long-gap retrieval.

The learned gate typically takes values around 0.01–0.05 in the sampled analysis. Because the fixed-gate control uses `0.5`, the current experiments cannot conclusively separate input-dependent adaptation from stronger decay suppression. A useful follow-up is to compare matched long-gap sequences containing only NOOPs against sequences with intervening SET updates to other keys.

See `research_report.pdf` and the notebook for error bars, detailed ablations, grouped analyses, resource measurements, and limitations.

## Files

| File | Purpose |
|---|---|
| `HackerRank_Full_Notebook.ipynb` | End-to-end Colab workflow, experiments, checks, plots, and findings |
| `research_report.pdf` | Six-page research report |
| `generate_dataset.py` | HackerRank-provided synthetic data generator |
| `model_retention.py` | Retention-gated Mamba-1 modification |
| `requirements.txt` | Python dependencies |

The notebook also writes the training/evaluation scripts `run_baseline.py` and `run_retention.py` during execution.

## Reproducing the Experiments in Google Colab

1. Open `HackerRank_Full_Notebook.ipynb` in **Google Colab**.
2. Select **Runtime → Change runtime type → GPU** (a Tesla T4 was used for the reported measurements).
3. Select **Runtime → Run all**, following the notebook from top to bottom.

The notebook installs `einops`, clones the upstream `mamba-minimal` repository, records its commit, writes the dataset generator and modified/training scripts, creates all data splits, and runs the baseline, proposed model, controlled ablations, tests, and multi-seed analysis. Its generated outputs are stored in `results/` within the Colab runtime.

For a local environment, the dependencies listed in `requirements.txt` can be installed with:

```bash
python -m pip install -r requirements.txt
```

**CPU note:** The individual baseline and retention training scripts support `--device cpu`, but the notebook's full multi-seed resource-measurement cell explicitly requires CUDA. For a reduced CPU fallback experiment, use `d_model=16` and `d_state=4` consistently in both models and skip the GPU-only resource cell; label any such results as a fallback, not as reproductions of the reported GPU configuration.

## Attribution

- **Mamba-1 implementation:** [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal). The notebook clones and extends this upstream implementation rather than reimplementing the backbone from scratch.
- **Task and dataset generator:** Provided as part of the HackerRank Deep Learning Research Engineer assignment.

## AI Assistance Disclosure

ChatGPT was used to assist with research brainstorming, refining the proposed approach, code implementation and debugging, organizing the experimental workflow, and improving the writing and presentation of the research report.
