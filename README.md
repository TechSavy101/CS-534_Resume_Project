# Résumé AI-assistance classification dataset

Start with **training set.zip**. Extract it, then open the folder that contains **dataset/**. Load **dataset/manifest_all_complete.csv** for the full experiment. The handoff contains **208 source groups and 832 résumé documents**, with one version of each class for every source.

## Folder guide

| Path | Contents |
|---|---|
| `dataset/manifest_all_complete.csv` | Index for all 832 documents. Use it to load the files, labels, and original splits. |
| `dataset/human/` | De-identified source résumés, presumed human-authored. Class **0**. |
| `dataset/light_ai/` | Versions intended to preserve the original structure while making light copyedits. Class **1**. |
| `dataset/heavy_ai/` | Substantially rewritten versions of the source résumés. Class **2**. |
| `dataset/full_ai/` | AI drafts produced from structured fact sheets. Class **3**. |
| `manifest.csv`, if present | Approval-gated index. The reported handoff had one fully approved group in this index. |
| Repair, redaction, and unresolved-identifier logs | Audit notes for changes and flags. Keep them out of the text sent to an encoder. |
| Included Ollama vectors | Four 768-dimensional vectors generated with `nomic-embed-text`. They are usable only as outputs of that model. Generate MiniLM and BGE vectors directly from the corresponding text. |

The exact names and paths of the optional audit files and Ollama vectors can be listed with `unzip -Z1 "training set.zip"`. The raw source CSV, fact sheets, source previews, credentials, and checkpoints are outside the handoff.

## Read the manifest

| Column | How to use it |
|---|---|
| `source_id` | Shared identifier for the four versions of one source résumé. Read it as a string. |
| `class_id` | Numeric classification target, from 0 to 3 as listed above. |
| `class_name` | Human-readable label and class-folder name. |
| `split` | Existing `train`, `validation`, or `test` membership. |
| `relative_path` | File path relative to the directory containing the manifest. For example, `human/99244405.txt` resolves to `dataset/human/99244405.txt`. Read as UTF-8. |
| `qc_status` | Review status for the row. Keep it as metadata. |

| Split | Source groups | Documents per class | Total documents |
|---|---:|---:|---:|
| Train | 167 | 167 | 668 |
| Validation | 21 | 21 | 84 |
| Test | 20 | 20 | 80 |
| **Total** | **208** | **208** | **832** |

Keep the original splits. Each group contains four closely related documents. Splitting individual documents randomly could expose one version of a source during training and another during testing.

## Generate embeddings

1. **Choose a representation.** Run `sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5` locally via Sentence Transformers. Each encoder gets the same 832 documents. If you use OpenAI `text-embedding-3-small` for an additional experiment, use a privacy-cleared common subset and score both local models on that same subset for the head-to-head comparison.
2. **Use the résumé text alone.** Keep the source ID, filename, class, split, and audit flags out of the text passed to the encoder. Use `(source_id, class_id)` as the document key when saving vectors.
3. **Chunk consistently.** Break long texts at paragraph, bullet, or sentence boundaries where possible. Use the MiniLM tokenizer to cap chunks at roughly 240 content tokens. Split longer units further and check the final chunks against each encoder’s tokenizer and limit, including special tokens. MiniLM defaults to truncating above 256 word pieces. Use the same chunks across models so each sees the same content.
4. **Pool at the document level.** Encode each chunk, average the vectors for that résumé, then L2-normalize the result. Save one document vector per résumé. Record the encoder ID and revision, chunking rule, and pooling rule. Save a companion index in the same order as the vectors so the labels and vectors stay aligned.
5. **Train and compare.** Fit one classifier per encoder on the training vectors. Use the validation split to choose settings and evaluate the selected settings on test. Run a TF-IDF baseline on the same split, fitting its vocabulary on training text. Report macro-F1, accuracy, per-class recall, and a confusion matrix. The encoders generate features. The downstream classifier, such as logistic regression, is the model you fit to the four labels.

### Minimal loader

Run this from the extracted folder containing `dataset/`. It checks the split and class coverage before embedding. Point `manifest_path` at the extracted file if your working directory differs.

```python
import csv
from collections import Counter, defaultdict
from pathlib import Path

manifest_path = Path("dataset/manifest_all_complete.csv")
dataset_root = manifest_path.parent.resolve()
with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))

labels = dict(enumerate(("human", "light_ai", "heavy_ai", "full_ai")))
groups = defaultdict(list)
for row in rows:
    key = int(row["class_id"])
    if labels.get(key) != row["class_name"]:
        raise ValueError(f"Invalid label for source {row['source_id']}")
    path = (dataset_root / row["relative_path"]).resolve()
    if dataset_root not in path.parents:
        raise ValueError(f"Invalid relative_path: {row['relative_path']}")
    row["text"] = path.read_text(encoding="utf-8-sig")
    if not row["text"].strip():
        raise ValueError(f"Empty résumé: {path}")
    groups[row["source_id"]].append(row)

for source_id, variants in groups.items():
    if len(variants) != 4 or {int(r["class_id"]) for r in variants} != set(labels):
        raise ValueError(f"Incomplete or duplicate group: {source_id}")
    if len({r["split"] for r in variants}) != 1:
        raise ValueError(f"Group crosses splits: {source_id}")

counts = Counter(row["split"] for row in rows)
if len(groups) != 208 or counts != {"train": 668, "validation": 84, "test": 80}:
    raise ValueError(f"Unexpected export counts: {len(groups)} groups, {counts}")
print(counts)
```

For each résumé, once you have a list of validated chunk strings, the core local embedding step is:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_resume(chunks):
    vectors = encoder.encode(chunks, normalize_embeddings=False)
    vector = vectors.mean(axis=0)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid document vector")
    return vector / norm
```

Install the local encoder dependencies with `python3 -m pip install sentence-transformers numpy scikit-learn` in your project environment. Run the same code with the BGE model ID to produce a separate BGE feature file. This snippet assumes you have implemented the shared chunking step above.

## Quality and interpretation

The reported archive had **267 approved rows, 564 pending rows, and 1 unverified row**. **One complete source group passed all checks.** The full handoff intentionally includes the other groups so the team can run the experiment. Report results from the full set as exploratory. Pending reviews can reflect changed or missing facts as well as formatting-related false alarms. Check whether length, formatting, or recurring redaction markers predict the model’s decisions before interpreting scores as evidence of writing style.

Four variants were flagged for possible remaining names: `26480367/light_ai`, `26480367/heavy_ai`, `31602598/full_ai`, and `37560528/heavy_ai`. Use local embeddings for the full handoff. Clear any residual identifier flags before sending text to an external embedding service.

The source corpus is Snehaan Bhawal’s [Resume Dataset on Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset). The AI variants were generated with local Qwen2.5-3B through Ollama. Labels identify the intended writing workflow. Individual source authorship and every generated claim were not independently verified.

Model references: [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [BGE](https://huggingface.co/BAAI/bge-small-en-v1.5), and [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings).
