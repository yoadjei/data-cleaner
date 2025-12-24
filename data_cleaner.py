import pandas as pd
import numpy as np
import json
import yaml
import os

class DataCleaningPipeline:
    def __init__(self, config_path):
        # loads config file, supports json and yaml now
        # letting it fail if file missing so user knows
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                self.config = yaml.safe_load(f)
            else:
                self.config = json.load(f)
        self.log = []
        self.issues = {}

    def analyze(self, df):
        """Schema analysis - shows what we're dealing with"""
        print("\n--- Schema Analysis ---")
        print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
        
        # show types and missing data
        analysis = pd.DataFrame({
            'Type': df.dtypes,
            'Missing': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(1)
        })
        print(analysis)
        
        # sample values help catch weird data issues early
        print("\nSample values (first 3):")
        for col in df.columns:
            samples = df[col].dropna().head(3).tolist()
            print(f"  {col}: {samples}")
        
        self.log.append("Analyzed dataset schema")

    def validate(self, df):
        """check for problems before cleaning"""
        print("\n--- Validation ---")
        
        critical = False
        
        # required columns check
        required = self.config.get("required_columns", [])
        for col in required:
            if col not in df.columns:
                print(f"CRITICAL: Missing required column '{col}'")
                self.issues[col] = "missing_column"
                critical = True
            else:
                missing = df[col].isnull().sum()
                if missing > 0:
                    print(f"Warning: '{col}' has {missing} missing values")
                    self.issues[col] = f"missing_values_{missing}"
        
        # duplicates
        dupes = df.duplicated().sum()
        if dupes > 0:
            print(f"Found {dupes} duplicate rows")
            self.issues['duplicates'] = dupes
        
        # type mismatches - numeric data stored as strings happens all the time
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                try:
                    pd.to_numeric(non_null, errors='raise')
                    print(f"Note: '{col}' looks numeric but stored as text")
                    self.issues[col] = "type_mismatch"
                except:
                    pass
        
        if not self.issues:
            print("No issues found")
        
        self.log.append(f"Validation done - {len(self.issues)} issues found")
        return not critical

    def run_cleaning(self, df):
        """main cleaning logic"""
        print("\n--- Cleaning ---")
        df = df.copy()
        
        start_rows = len(df)

        # remove duplicates first
        before = len(df)
        df = df.drop_duplicates()
        if len(df) < before:
            removed = before - len(df)
            print(f"Removed {removed} duplicates")
            self.log.append(f"Removed {removed} duplicates")

        # apply cleaning strategies from config
        strategies = self.config.get("strategies", {})
        
        for col, rules in strategies.items():
            if col not in df.columns:
                continue

            # type conversion first - otherwise mean/median fails on strings
            if "convert_to" in rules:
                target = rules["convert_to"]
                try:
                    if target in ['int', 'float', 'int64', 'float64']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if 'int' in target.lower():
                            df[col] = df[col].astype('Int64')  # nullable int
                    else:
                        df[col] = df[col].astype(target)
                    print(f"Converted '{col}' to {target}")
                    self.log.append(f"Converted '{col}' to {target}")
                except Exception as e:
                    print(f"Warning: couldn't convert '{col}' - {e}")

            # imputation - fill missing values
            if "impute" in rules:
                miss_cnt = df[col].isnull().sum()
                if miss_cnt > 0:
                    method = rules["impute"]
                    
                    if method == "drop":
                        before_drop = len(df)
                        df = df.dropna(subset=[col])
                        dropped = before_drop - len(df)
                        print(f"Dropped {dropped} rows with missing '{col}'")
                        self.log.append(f"Dropped {dropped} rows missing '{col}'")
                    else:
                        # calculate fill value based on method
                        if method == "mean":
                            val = df[col].mean()
                        elif method == "median":
                            val = df[col].median()
                        elif method == "mode":
                            m = df[col].mode()
                            val = m[0] if len(m) > 0 else "UNKNOWN"
                        else:
                            print(f"Warning: unknown method '{method}'")
                            continue
                        
                        df[col] = df[col].fillna(val)
                        print(f"Filled {miss_cnt} missing in '{col}' using {method} (value: {val})")
                        self.log.append(f"Imputed '{col}' with {method}")

            # normalize strings - strip and lowercase
            # do this last so we don't mess up numbers
            if rules.get("normalize") and df[col].dtype == 'object':
                orig = df[col].copy()
                df[col] = df[col].astype(str).str.strip().str.lower()
                changed = (orig != df[col]).sum()
                if changed > 0:
                    print(f"Normalized {changed} values in '{col}'")
                    self.log.append(f"Normalized '{col}'")

        end_rows = len(df)
        print(f"\nSummary: {start_rows} -> {end_rows} rows ({start_rows - end_rows} removed)")
        return df

    def export(self, df, output_name):
        """save cleaned data and log"""
        print("\n--- Export ---")
        
        # csv always works
        csv_file = f"{output_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved {csv_file}")
        
        # parquet is optional but better
        try:
            pq_file = f"{output_name}.parquet"
            df.to_parquet(pq_file, index=False)
            print(f"Saved {pq_file}")
        except ImportError:
            print("Skipped parquet (need: pip install pyarrow)")
        except Exception as e:
            print(f"Parquet failed: {e}")
        
        # audit log
        log_file = f"{output_name}_log.txt"
        with open(log_file, "w") as f:
            f.write("="*50 + "\n")
            f.write("DATA CLEANING LOG\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Final: {df.shape[0]} rows x {df.shape[1]} cols\n\n")
            
            if self.issues:
                f.write("Issues Found:\n")
                f.write("-"*50 + "\n")
                for k, v in self.issues.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
            
            f.write("Actions:\n")
            f.write("-"*50 + "\n")
            for entry in self.log:
                f.write(f"- {entry}\n")
            
            f.write("\n" + "="*50 + "\n")
        
        print(f"Saved {log_file}")

# --- run it ---
if __name__ == "__main__":
    # config setup
    config = {
        "required_columns": ["id", "salary"],
        "strategies": {
            "name": {"normalize": True},
            "age": {"convert_to": "float", "impute": "median"},
            "salary": {"impute": "mean"},
            "dept": {"impute": "mode"}
        }
    }
    
    if not os.path.exists("config.json"):
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("Created default config.json")
    else:
        print("Found existing config.json")
    
    # yaml example too
    yaml_cfg = """required_columns:
  - id
  - salary

strategies:
  name:
    normalize: true
  age:
    convert_to: float
    impute: median
  salary:
    impute: mean
  dept:
    impute: mode
"""
    if not os.path.exists("config.yaml"):
        with open("config.yaml", "w") as f:
            f.write(yaml_cfg)
        print("Created default config.yaml")
    else:
        print("Found existing config.yaml")

    # test data with common issues
    data = {
        "id": [1, 2, 2, 4, 5],
        "name": [" Alice", "BOB", "bob", "Charlie ", "  dave  "],
        "age": ["25", "30", "30", None, "35"],
        "salary": [5000, 7000, 7000, None, 6200],
        "dept": ["HR", "IT", "IT", None, "Sales"]
    }
    df = pd.DataFrame(data)
    
    print("\nOriginal data:")
    print(df)

    # run pipeline
    pipeline = DataCleaningPipeline("config.json")
    
    pipeline.analyze(df)
    
    if pipeline.validate(df):
        cleaned = pipeline.run_cleaning(df)
        
        print("\nCleaned data:")
        print(cleaned)
        print("\nTypes:")
        print(cleaned.dtypes)
        
        pipeline.export(cleaned, "cleaned_output")
        print("\nDone!")
    else:
        print("\nCritical errors found - fix and rerun")