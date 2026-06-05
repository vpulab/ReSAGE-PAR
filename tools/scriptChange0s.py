#!/usr/bin/env python3
"""Replace 0s with 1s (excluding first column) and write to folder.

Usage:
  python scriptChange0s.py --folder /path/to/folder --csv /path/to/input.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def transform_row(row: list[str]) -> list[str]:
	if not row:
		return row
	transformed = [row[0]]
	for value in row[1:]:
		transformed.append("1" if value == "0" else value)
	return transformed


def main() -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Given a folder path and a reference CSV, create a file named "
			"pseudolabels_with-1.csv in the folder, replacing 0s with 1s "
			"in each row (excluding the first column)."
		)
	)
	parser.add_argument(
		"--folder",
		required=True,
		help="Destination folder where pseudolabels_with-1.csv will be written.",
	)
	parser.add_argument(
		"--csv",
		required=True,
		help="Path to input CSV file.",
	)
	args = parser.parse_args()

	folder = Path(args.folder)
	input_csv = Path(args.csv)
	output_csv = folder / "pseudolabels_with-1.csv"

	if not input_csv.is_file() and not input_csv.is_absolute():
		candidate = folder / input_csv
		if candidate.is_file():
			input_csv = candidate

	if not input_csv.is_file():
		raise FileNotFoundError(
			f"Input CSV not found: {input_csv}. "
			"Use an absolute path or place the CSV inside --folder."
		)

	folder.mkdir(parents=True, exist_ok=True)

	with input_csv.open(newline="", encoding="utf-8") as infile, output_csv.open(
		"w", newline="", encoding="utf-8"
	) as outfile:
		reader = csv.reader(infile)
		writer = csv.writer(outfile)
		for row in reader:
			writer.writerow(transform_row(row))

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
