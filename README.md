# E2E Child TSE

End-to-end **child-speech TSE + SELECTION anonymization** on **900** two-speaker mixtures (AA / CA / CC × six overlap levels).

Code: [github.com/pranavtushar/E2E-CHILD-TSE](https://github.com/pranavtushar/E2E-CHILD-TSE) — large files on **Hugging Face**.

## Hugging Face assets

| Resource | Repo |
|----------|------|
| Data (`data/`) | [pranavtushar/e2e-child-tse-data](https://huggingface.co/datasets/pranavtushar/e2e-child-tse-data) |
| Weights (`vendor/`) | [pranavtushar/e2e-child-tse-vendor](https://huggingface.co/pranavtushar/e2e-child-tse-vendor) |
| Baseline `outputs/` (optional) | same dataset, folder `outputs/` on the Hub |

```bash
cd E2E_CHILD_TSE_2   # host: /mnt/hdd/pranav/E2E_CHILD_TSE_2  |  Docker: /app/E2E_CHILD_TSE_2

pip install -U huggingface_hub
hf auth login

hf download pranavtushar/e2e-child-tse-data --repo-type dataset --local-dir data
hf download pranavtushar/e2e-child-tse-vendor --repo-type model --local-dir vendor
```

Local copy from sibling trees (no HF): `bash scripts/install_self_contained.sh`

## Run

```bash
export PY=/root/miniconda3/envs/va_toolkit_py39/bin/python3
unset ADULT_REFERENCE_DIR PYTHON

bash scripts/preflight.sh
bash run.sh
```

Quick test (~30 items): `SMOKE=1 bash run.sh`

Skip remixing: `SKIP_CLEAN=1 SKIP_MIX=1 bash run.sh`

## Layout

```
E2E_CHILD_TSE_2/
  run.sh, scripts/, pipeline/, evaluation/
  data/      # HF — manifests + Libri + MyST
  vendor/    # HF — TSE + SELECTION weights
  mixtures/  # generated
  outputs/   # tse_hat, anon, recombine
  results/   # EER + WER
```

## Results

| Metric | Path |
|--------|------|
| EER | `results/eer/results/eer_summary.csv` |
| WER | `results/wer/s1_s2_whisper/s1_s2_wer_pack.xlsx` |

Mixtures are fixed by `data/manifests/mixture_mfa_catalog_docker.csv` (900 rows).
