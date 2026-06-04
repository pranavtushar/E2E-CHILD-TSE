# E2E Child TSE (code)

Pipeline code for child-speech TSE + SELECTION anonymization (900 mixtures).

**Assets (not in this repo):**

| Resource | Hugging Face |
|----------|----------------|
| Data (`data/`) | [pranavtushar/e2e-child-tse-data](https://huggingface.co/datasets/pranavtushar/e2e-child-tse-data) |
| Weights (`vendor/`) | [pranavtushar/e2e-child-tse-vendor](https://huggingface.co/pranavtushar/e2e-child-tse-vendor) |

Download locally:

```bash
hf download pranavtushar/e2e-child-tse-data --repo-type dataset --local-dir data
hf download pranavtushar/e2e-child-tse-vendor --repo-type model --local-dir vendor
```

Then: `bash scripts/preflight.sh && bash run.sh`
