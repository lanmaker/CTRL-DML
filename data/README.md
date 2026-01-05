# Data Manifest

`manifest.json` records file hashes and source URLs for datasets used in the paper.
If any checksum mismatches, re-download the dataset from the listed source.

Notes:
- HuggingFace artifacts (Yelp + BERT) are pinned via `CTRL_DML_YELP_REVISION` and
  `CTRL_DML_BERT_REVISION`. The local cache writes a small manifest with the
  dataset fingerprint.
- `data/yelp_cache/` is a cache directory and is not versioned.
