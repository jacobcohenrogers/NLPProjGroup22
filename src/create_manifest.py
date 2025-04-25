# src/generate_manifest.py
"""
Generate a manifest CSV for the NLPProjGroup22 dataset layout.

Assumes the following directory structure:

NLPProjGroup22/
└── data/
    └── raw_text/
        ├── training/
        │   ├── Emilydickinson/
        │   │   ├── emilydickinson_0.txt
        │   │   └── ...
        │   ├── Frost/
        │   ├── Robertburns/
        │   └── Shakespeare/
        │   └── Waltwhitman/
        └── testing/
        │   ├── Emilydickinson/
        │   │   ├── emilydickinson_0.txt
        │   │   └── ...
        │   ├── Frost/
        │   ├── Robertburns/
        │   └── Shakespeare/
        │   └── Waltwhitman/

This script scans both `training/` and `testing/` splits, extracts author directories,
and writes a `manifest.csv` under `data/` with columns:
    file_path,author,prompt,split

Usage:
    python generate_manifest.py

"""
import csv
from pathlib import Path
import sys

def main():
    # Determine project root (two levels up from this script)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    raw_text_root = project_root / 'data' / 'raw_text'
    if not raw_text_root.exists():
        print(f"Error: raw_text directory not found at {raw_text_root}", file=sys.stderr)
        sys.exit(1)

    output_path = project_root / 'data' / 'manifest.csv'
    rows = []

    # Iterate over splits
    for split_dir in ['training', 'testing']:
        split_path = raw_text_root / split_dir
        if not split_path.exists():
            print(f"Warning: split directory not found: {split_path}", file=sys.stderr)
            continue

        # Each author is a subdirectory
        for author_dir in sorted(split_path.iterdir()):
            if not author_dir.is_dir():
                continue
            author = author_dir.name

            # Each .txt file in the author directory
            for txt_file in sorted(author_dir.glob('*.txt')):
                stem = txt_file.stem  # e.g. 'gomez_0'
                # use stem as prompt label (or customize mapping here)
                prompt = stem

                # relative path from project root
                rel_path = txt_file.relative_to(project_root)

                rows.append({
                    'file_path': str(rel_path),
                    'author': author,
                    'prompt': prompt,
                    'split': split_dir
                })

    # Write manifest
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['file_path', 'author', 'prompt', 'split'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote manifest with {len(rows)} entries to {output_path}")

if __name__ == '__main__':
    main()