import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("human_original", "light_ai_edit", "heavy_ai_edit", "ai_from_scratch")
FIELDS = ("source_id", "label", "resume_text")
FOLDERS = dict(zip(("human", "light_ai", "heavy_ai", "full_ai"), LABELS))


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        headers = reader.fieldnames or []
        missing = set(required) - set(headers)
        if missing or len(headers) != len(set(headers)):
            raise ValueError(f"{path}: invalid headers {headers!r}; missing {sorted(missing)}")
        rows = list(reader)
        bad = [i for i, row in enumerate(rows, 1) if None in row or None in row.values()]
        if bad:
            raise ValueError(f"{path}: wrong field count in data records {bad}")
        return rows


def check_output(path, inputs):
    for source in inputs:
        if path.resolve() == source.resolve() or (
            path.exists() and source.exists() and os.path.samefile(path, source)
        ):
            raise ValueError(f"Output would overwrite {source}: {path}")


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def validate(rows):
    groups = defaultdict(Counter)
    problems = []
    for index, (source_id, label, resume_text) in enumerate(rows, 1):
        groups[source_id][label] += 1
        if not source_id.strip() or source_id != source_id.strip():
            problems.append(f"Record {index}: invalid source_id {source_id!r}")
        if label not in LABELS:
            problems.append(f"Record {index}, group {source_id!r}: illegal label {label!r}")
        if not resume_text.strip():
            problems.append(f"Record {index}, group {source_id!r}, {label!r}: empty text")
    for source_id, counts in sorted(groups.items()):
        missing = [label for label in LABELS if not counts[label]]
        duplicates = {label: count for label, count in counts.items() if count > 1}
        if sum(counts.values()) != 4 or missing or duplicates:
            problems.append(
                f"Group {source_id!r}: {sum(counts.values())} rows, "
                f"missing={missing}, duplicates={duplicates}"
            )
    if not rows:
        problems.append("No résumé records found")
    if problems:
        raise ValueError("\n".join(problems))
    return sorted(rows, key=lambda row: (row[0], LABELS.index(row[1])))


def load_long(path):
    return validate([tuple(row[field] for field in FIELDS) for row in read_csv(path, FIELDS)])


def load_dataset(root):
    manifest = root / "manifest_all_complete.csv"
    records = read_csv(manifest, ("source_id", "class_id", "class_name", "relative_path"))
    files = {path for folder in FOLDERS for path in (root / folder).glob("*.txt")}
    used = set()
    rows = []
    problems = []
    for record in records:
        source_id = record["source_id"]
        folder = record["class_name"]
        relative = record["relative_path"]
        path = root / relative
        if (
            folder not in FOLDERS
            or record["class_id"] != str(tuple(FOLDERS).index(folder))
            or not source_id.isascii()
            or not source_id.isdecimal()
            or relative != f"{folder}/{source_id}.txt"
        ):
            problems.append(f"Group {source_id!r}: inconsistent manifest entry {relative!r}")
            continue
        used.add(path)
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                resume_text = stream.read()
        except (OSError, UnicodeError) as error:
            problems.append(f"Group {source_id!r}, {folder}: {error}")
            resume_text = ""
        rows.append((source_id, FOLDERS[folder], resume_text))
    problems.extend(f"Text file absent from manifest: {path}" for path in sorted(files - used))
    try:
        rows = validate(rows)
    except ValueError as error:
        problems.append(str(error))
    if problems:
        raise ValueError("\n".join(problems))
    return rows, [manifest, *files]


def main():
    parser = argparse.ArgumentParser(description="Validate and consolidate existing résumés.")
    formats = parser.add_mutually_exclusive_group()
    formats.add_argument("--dataset", type=Path, help="Directory containing the supplied manifest")
    formats.add_argument("--long", type=Path, help="CSV with source_id, label, resume_text")
    formats.add_argument("--wide", type=Path, help="CSV with source_id and one column per label")
    formats.add_argument("--class-files", type=Path, nargs=4, metavar="CSV",
                         help="Four CSVs in label order, each with source_id and resume_text")
    parser.add_argument("--out", type=Path, default=Path("resumes.csv"))
    args = parser.parse_args()
    try:
        if args.long:
            rows, inputs = load_long(args.long), [args.long]
        elif args.wide:
            inputs = [args.wide]
            rows = validate([
                (row["source_id"], label, row[label])
                for row in read_csv(args.wide, ("source_id", *LABELS))
                for label in LABELS
            ])
        elif args.class_files:
            inputs = args.class_files
            rows = validate([
                (row["source_id"], label, row["resume_text"])
                for label, path in zip(LABELS, inputs)
                for row in read_csv(path, ("source_id", "resume_text"))
            ])
        else:
            root = args.dataset or Path("dataset")
            rows, inputs = load_dataset(root)
            inputs.extend(root.glob("*.csv"))
        check_output(args.out, inputs)
        write_csv(args.out, FIELDS, rows)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"{error}\n")
    print(f"Wrote {len(rows)} texts in {len(rows) // len(LABELS)} groups to {args.out}")


if __name__ == "__main__":
    main()
