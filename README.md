# Data Cleaning Pipeline

A robust Python tool for automated data cleaning and validation. It handles common issues like missing values, duplicates, and type mismatches based on a configurable strategy.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the script directly:

```bash
python data_cleaner.py
```

It will generate default configuration files (`config.json`, `config.yaml`) if they do not exist. You can edits these files to customize the cleaning rules.

## Output

The script generates:
- `cleaned_output.csv`: The cleaned dataset.
- `cleaned_output.parquet`: Optimized binary format (if pyarrow is available).
- `cleaned_output_log.txt`: A detailed audit log of changes made.
