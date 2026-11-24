# Résumé version classification

This project compares two frozen sentence encoders on four existing résumé versions using a separate logistic regression classifier for each encoder. All résumé text and model inference stay on the local machine. The first run downloads publicly available model weights and configuration files from Hugging Face.

## Supplied dataset

The dataset contains 208 complete source groups and 832 UTF-8 text files. The inclusion manifest, `dataset/manifest_all_complete.csv`, provides the original `source_id`, `class_id`, `class_name`, and `relative_path`. Each path must agree with the category directory and the numeric filename stem. The script also checks for unlisted text files in the four category folders.

| Class ID | Folder | Output label |
| --- | --- | --- |
| 0 | `human` | `human_original` |
| 1 | `light_ai` | `light_ai_edit` |
| 2 | `heavy_ai` | `heavy_ai_edit` |
| 3 | `full_ai` | `ai_from_scratch` |

The handoff already contains staged copies and local redaction records. The manifest retains the supplied review statuses (1 approved, 585 pending, and 246 unverified). Handle the data locally. The experiment builds the requested 80/20 group split with seed 42 independently of the earlier split assignments in the manifest.

## Set up on macOS and in VS Code

Open the project folder in VS Code and run these commands from its integrated terminal. Use Python 3.10, 3.11, or 3.12. The example uses Python 3.12.

```bash
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For VS Code run and debug actions, use the Python extension's **Python: Select Interpreter** command and select `.venv/bin/python`.

## Prepare the supplied data

```bash
python prepare_data.py --dataset dataset --out resumes.csv
```

This reads the supplied manifest and text files, validates each group, and writes UTF-8 CSV with `source_id,label,resume_text`. CSV quoting preserves embedded punctuation, quotes, and line breaks. Each group must have exactly one nonblank résumé per class. Invalid groups are reported before any output is written.

For additional data with verified source IDs, the following input formats are supported. Use the existing shared source ID in every input. A wide table has one row per source and one text column per output label listed above. A standard long table has `source_id,label,resume_text`. Each separate class CSV has `source_id,resume_text`, and the command assigns its class from the argument order.

```bash
python prepare_data.py --wide wide.csv --out resumes.csv
python prepare_data.py --long existing_long.csv --out resumes.csv
python prepare_data.py --class-files human.csv light.csv heavy.csv full.csv --out resumes.csv
```

## Run the experiment

```bash
python run_experiment.py --data resumes.csv --seed 42 --out results/predictions.csv --plots results/plots
```

The split samples 42 test source IDs without replacement from the sorted list of 208 IDs using NumPy's seeded generator. The remaining 166 IDs form the training set. All four versions stay together, giving 664 training and 168 test texts. Both models receive the same texts in the same order.

- MiniLM uses `sentence-transformers/all-MiniLM-L6-v2`.
- BGE Small uses `BAAI/bge-small-en-v1.5`.

Both models run on CPU. The two tokenizers check each résumé against their own sequence limits, including special tokens. Longer texts are segmented into shared contiguous chunks. For each encoder, the script averages the normalized chunk embeddings with equal weights and then applies L2 normalization. Only résumé text is embedded. Each classifier is fitted separately on its own training embeddings and the training labels.

## View results

The terminal shows split counts, embedding dimensions, number of correct predictions, accuracy, macro F1, per-class recall, a confusion matrix, and individual predictions for each encoder. Use macro F1 as the primary comparison. The matrix uses the label order in the table for true rows and predicted columns.

`results/predictions.csv` contains `model,source_id,true_label,predicted_label,correct`. The supplied dataset produces 336 prediction rows, 168 for each model. The t-SNE PNGs use the corresponding test embeddings and color the points by true class after projection. The projections are for visualization only.

```bash
head -n 6 results/predictions.csv
wc -l results/predictions.csv
open results/plots/minilm_tsne.png results/plots/bge_tsne.png
```
