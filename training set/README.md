# Résumé training set handoff

This handoff contains **208 complete source groups** and **832 UTF-8 résumé documents**. Each source ID has exactly one version in each class. The frozen source ID list, category source, class folders, review statuses, and split assignments were verified before packaging.

## Class mapping

| class_id | class_name |
|---:|---|
| 0 | human |
| 1 | light_ai |
| 2 | heavy_ai |
| 3 | full_ai |

## Splits

| split | groups | documents |
|---|---:|---:|
| train | 167 | 668 |
| validation | 21 | 84 |
| test | 20 | 80 |

All four versions of a source remain in the same split. Fit models on train, tune on validation, and evaluate once on test. Read `dataset/manifest_all_complete.csv` as the inclusion manifest; its `relative_path` values are relative to the `dataset/` directory. This is intentionally independent of the approval-gated `manifest.csv`.

## Review status

Manifest row counts: approved=1, pending=585, unverified=246. Fully approved groups: 0; groups with at least one pending or unverified version: 208.

## Local redaction and unresolved checks

Staged copies were formatted consistently and checked/redacted locally using macOS NaturalLanguage person-name tagging, header-name checks, and local regexes for email, phone, URL, social handle, and street address. No résumé content was sent to a hosted service. Automated staged-copy redactions are recorded in `dataset/redaction_log.csv` by source, class, and reason code; `dataset/unresolved_identifiers.csv` contains only source IDs, classes, reason codes, and counts.

Residual identifier detections after redaction: 0. Reason counts: {}. These are potential or uncertain matches, not a claim that every detection is a real identifier. Do not send these records to hosted embedding or inference services; if these records need vector features, generate embeddings locally and retain them locally. Texts are included as required, with their original review state retained in the manifest.

## Provenance and exclusions

The existing seed-2026 split function was used for every frozen source ID. Source CSV, raw source previews, fact sheets, credentials, backups, and generation metadata are not included. The separate pre-repair backup is stored outside this handoff.
