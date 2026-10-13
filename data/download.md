# Getting MIND

Microsoft News Dataset (Wu et al., ACL 2020). Licence: Microsoft Research License Terms. You
have to accept them before downloading. Do not commit the data; `data/raw/` is gitignored.

## What is needed

Only the **labelled** splits. MIND's `test` split has no click labels, so it is never ingested.

| variant | files | zipped | use |
|---|---|---|---|
| large | `MINDlarge_train.zip`, `MINDlarge_dev.zip` | ~0.5 GB + ~0.1 GB | the real run |
| small | `MINDsmall_train.zip`, `MINDsmall_dev.zip` | ~50 MB total | laptop iteration only |

Downloads are linked from <https://msnews.github.io/> (the "Download" section). As of
2026-10-13 its buttons point at the gated Hugging Face dataset `yjw1029/MIND`; the old
`mind201910small.blob.core.windows.net` mirror returns `409 Public access is not permitted`.
Accept the dataset's terms on Hugging Face, then download with a read token:

```bash
cd data/raw
read -s HF_TOKEN   # paste a Hugging Face read token; not echoed, not saved
for f in MINDlarge_train MINDlarge_dev; do
  curl -fL -H "Authorization: Bearer $HF_TOKEN" -o $f.zip \
    https://huggingface.co/datasets/yjw1029/MIND/resolve/main/$f.zip
done
```

After unzipping (below), pin the bytes: `sha256sum data/raw/MINDlarge_{train,dev}/{behaviors,news}.tsv
> data/MANIFEST-mindlarge.sha256` from the repo root, and commit the manifest. `arc/pipeline.slurm`
refuses to ingest data that does not match it.

## Layout expected by `configs/config.yaml`

```
data/raw/
  MINDlarge_train/  behaviors.tsv  news.tsv  entity_embedding.vec  relation_embedding.vec
  MINDlarge_dev/    behaviors.tsv  news.tsv  ...
```

```bash
mkdir -p data/raw && cd data/raw
unzip MINDlarge_train.zip -d MINDlarge_train
unzip MINDlarge_dev.zip   -d MINDlarge_dev
```

For the small variant, set `data.variant: small` in the config. The directory names become
`MINDsmall_*`.

## Time coverage (what the split dates in the config assume)

- `train` behaviors: 2019-11-09 .. 2019-11-14
- `dev` behaviors: 2019-11-15

Timestamps carry no timezone and are parsed as UTC. `split.temporal_cutoff` defaults to
2019-11-15 00:00, so the test set is the dev day. `split.validation_cutoff` defaults to
2019-11-14 00:00, so tuning and ranker training use the last train day. Check
`results/dataset_stats.json` → `per_day` after ingest. If the days differ from the ones
above, change the cutoffs in the config, not in code.

## Sanity checks after `recsys ingest`

- `impressions` ≈ 2.2M (train) + 0.38M (dev) for MIND-large
- `ctr` should be about 0.04
- zero rows with unparseable timestamps or missing labels (ingest raises otherwise)
