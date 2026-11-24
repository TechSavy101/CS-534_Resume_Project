import argparse
import math
import re
from bisect import bisect_right
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from threadpoolctl import threadpool_limits

from prepare_data import LABELS, check_output, load_long, write_csv

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}
PREDICTION_FIELDS = ("model", "source_id", "true_label", "predicted_label", "correct")


def split_rows(rows, seed):
    source_ids = sorted({row[0] for row in rows})
    if len(source_ids) < 3:
        raise ValueError("At least three complete source groups are required.")
    rng = np.random.default_rng(seed)
    test_ids = set(rng.choice(source_ids, size=math.ceil(len(source_ids) * 0.20), replace=False))
    train = [row for row in rows if row[0] not in test_ids]
    test = [row for row in rows if row[0] in test_ids]
    if {row[0] for row in train} & {row[0] for row in test}:
        raise RuntimeError("Training and test source IDs overlap.")
    for name, subset in (("Train", train), ("Test", test)):
        print(f"{name}: {len({row[0] for row in subset})} groups, {len(subset)} texts")
        for label in LABELS:
            print(f"  {label}: {sum(row[1] == label for row in subset)}")
    return train, test


def split_text(text, models):
    def fits(fragment):
        stripped = fragment.strip()
        within_limits = []
        for model in models:
            token_text = stripped.lower() if model[0].do_lower_case else stripped
            tokens = model.tokenizer(
                token_text, add_special_tokens=True, truncation=False, verbose=False
            )["input_ids"]
            within_limits.append(len(tokens) <= model.max_seq_length)
        return all(within_limits)

    if fits(text):
        return [text]

    boundaries = [match.end() for match in re.finditer(r"\S+\s*", text)]
    if not boundaries or boundaries[-1] != len(text):
        boundaries.append(len(text))

    def last_fitting(start, low, high, end_at):
        best = start
        while low <= high:
            mid = (low + high) // 2
            end = end_at(mid)
            if fits(text[start:end]):
                best = end
                low = mid + 1
            else:
                high = mid - 1
        return best

    start = 0
    chunks = []
    while start < len(text):
        first = bisect_right(boundaries, start)
        end = last_fitting(start, first, len(boundaries) - 1, boundaries.__getitem__)
        if end == start:
            end = last_fitting(start, start + 1, boundaries[first], lambda index: index)
        if end == start:
            raise ValueError(f"No shared tokenizer capacity at character offset {start}.")
        chunks.append(text[start:end])
        start = end
    return chunks


def encode_resumes(model, resumes):
    offsets = np.cumsum([0, *(len(chunks) for chunks in resumes)])
    flattened = [chunk for chunks in resumes for chunk in chunks]
    vectors = model.encode(flattened, normalize_embeddings=True, show_progress_bar=False)
    means = np.stack([
        vectors[start:end].mean(axis=0)
        for start, end in zip(offsets[:-1], offsets[1:])
    ])
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    return np.divide(means, norms, out=means, where=norms != 0)


def evaluate(name, train_vectors, train, test_vectors, test):
    classifier = LogisticRegression(C=1.0, max_iter=1000)
    classifier.fit(train_vectors, [row[1] for row in train])
    predicted = classifier.predict(test_vectors)
    actual = [row[1] for row in test]
    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)
    recalls = recall_score(actual, predicted, labels=LABELS, average=None, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=LABELS)
    correct = np.count_nonzero(predicted == actual)
    print(f"\n{name}: {test_vectors.shape[1]} dimensions, {correct}/{len(test)} correct")
    print(f"Accuracy: {accuracy:.6f} | Macro F1: {macro_f1:.6f}")
    for label, recall in zip(LABELS, recalls):
        print(f"  Recall {label}: {recall:.6f}")
    print("Confusion matrix (rows true; columns predicted)")
    print("Label order:", ", ".join(LABELS))
    print(matrix)
    print("source_id\ttrue_label\tpredicted_label")
    rows = []
    for (source_id, true_label, _), predicted_label in zip(test, predicted):
        print(f"{source_id}\t{true_label}\t{predicted_label}")
        rows.append((name, source_id, true_label, predicted_label, true_label == predicted_label))
    return rows


def save_tsne(name, vectors, test, seed, path):
    perplexity = min(30, max(2, (len(test) - 1) // 3))
    with threadpool_limits(limits=1):
        coordinates = TSNE(
            n_components=2, init="random", random_state=seed, perplexity=perplexity
        ).fit_transform(vectors)
    labels = np.asarray([row[1] for row in test])
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    try:
        for label in LABELS:
            points = coordinates[labels == label]
            ax.scatter(points[:, 0], points[:, 1], label=label, alpha=0.8)
        ax.set(title=f"{name} test embeddings", xlabel="t-SNE 1", ylabel="t-SNE 2")
        ax.legend()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, format="png")
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare MiniLM and BGE Small on grouped résumés.")
    parser.add_argument("--data", type=Path, default=Path("resumes.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/predictions.csv"))
    parser.add_argument("--plots", type=Path, default=Path("results/plots"))
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    paths = [args.data, args.out]
    try:
        check_output(args.out, [args.data])
        for name in MODELS:
            path = args.plots / f"{name}_tsne.png"
            check_output(path, paths)
            paths.append(path)
        rows = load_long(args.data)
        train, test = split_rows(rows, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    from sentence_transformers import SentenceTransformer

    models = {}
    for name, model_id in MODELS.items():
        print(f"Loading {model_id} on CPU", flush=True)
        models[name] = SentenceTransformer(model_id, device="cpu", trust_remote_code=False)
    chunks = [split_text(row[2], models.values()) for row in train + test]
    predictions = []
    for name, model in models.items():
        print(f"Encoding {len(chunks)} résumés with {name}", flush=True)
        vectors = encode_resumes(model, chunks)
        train_vectors, test_vectors = vectors[:len(train)], vectors[len(train):]
        predictions.extend(evaluate(name, train_vectors, train, test_vectors, test))
        save_tsne(name, test_vectors, test, args.seed, args.plots / f"{name}_tsne.png")
    write_csv(args.out, PREDICTION_FIELDS, predictions)
    print(f"Wrote {len(predictions)} predictions to {args.out}")
    print(f"Wrote two t-SNE plots to {args.plots}")


if __name__ == "__main__":
    main()
