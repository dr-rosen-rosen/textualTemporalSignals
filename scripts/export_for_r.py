"""
Export processed data in formats optimized for R analysis.
Includes metadata and type preservation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import yaml
from typing import Dict, List, Optional


def create_metadata(df: pd.DataFrame, source_info: Dict) -> Dict:
    """
    Create metadata about the dataset for R users.
    
    Args:
        df: Processed dataframe
        source_info: Information about data sources
        
    Returns:
        Dictionary containing metadata
    """
    metadata = {
        'creation_date': pd.Timestamp.now().isoformat(),
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'sources': source_info,
        'columns': {},
        'date_range': {
            'min': df['event_date'].min().isoformat() if pd.notna(df['event_date'].min()) else None,
            'max': df['event_date'].max().isoformat() if pd.notna(df['event_date'].max()) else None
        }
    }
    
    # Add column information
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'n_missing': df[col].isna().sum(),
            'n_unique': df[col].nunique()
        }
        
        # Add summary statistics for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info['stats'] = {
                'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                'std': float(df[col].std()) if not df[col].isna().all() else None,
                'min': float(df[col].min()) if not df[col].isna().all() else None,
                'max': float(df[col].max()) if not df[col].isna().all() else None
            }
        
        metadata['columns'][col] = col_info
    
    return metadata


def prepare_for_r(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for smooth import into R.
    
    Args:
        df: Processed dataframe
        
    Returns:
        R-friendly dataframe
    """
    df = df.copy()
    
    # Convert datetime columns to string for better R compatibility
    # R can parse these back to dates easily
    date_columns = df.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        df[f'{col}_iso'] = df[col].dt.strftime('%Y-%m-%d')
    
    # Ensure no problematic column names for R
    df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
    
    # Convert any remaining object dtypes to string
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        df[col] = df[col].astype(str).replace('nan', '')
    
    return df


def export_for_modeling(
    input_path: str,
    output_dir: str,
    formats: List[str] = ['parquet', 'rds_prep'],
    include_samples: bool = True
):
    """
    Export processed data in multiple formats for R modeling.
    
    Args:
        input_path: Path to combined processed data
        output_dir: Directory for output files
        formats: List of output formats
        include_samples: Whether to create sample datasets
    """
    # Load processed data
    df = pd.read_parquet(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get source information
    source_info = df['source_system'].value_counts().to_dict()
    
    # Create metadata
    metadata = create_metadata(df, source_info)
    
    # Save metadata
    with open(output_dir / 'data_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Prepare data for R
    df_r = prepare_for_r(df)
    
    # Export in different formats
    if 'parquet' in formats:
        df_r.to_parquet(output_dir / 'events_for_r.parquet', index=False)
        print(f"Exported parquet file to {output_dir / 'events_for_r.parquet'}")
    
    if 'feather' in formats:
        df_r.to_feather(output_dir / 'events_for_r.feather')
        print(f"Exported feather file to {output_dir / 'events_for_r.feather'}")
    
    if 'csv' in formats or 'rds_prep' in formats:
        # For RDS prep, we save as CSV with type hints
        df_r.to_csv(output_dir / 'events_for_r.csv', index=False)
        print(f"Exported CSV file to {output_dir / 'events_for_r.csv'}")
        
        # Create R script for proper data loading
        if 'rds_prep' in formats:
            create_r_import_script(output_dir, metadata)
    
    # Create sample datasets if requested
    if include_samples:
        create_sample_datasets(df_r, output_dir)
    
    # Create summary report
    create_summary_report(df, output_dir)


def create_r_import_script(output_dir: Path, metadata: Dict):
    """
    Create an R script that properly loads the data with correct types.
    
    Args:
        output_dir: Output directory
        metadata: Data metadata dictionary
    """
    r_script = '''# Auto-generated R script for loading event data
# Generated: {date}

library(tidyverse)
library(arrow)  # For parquet files

# Function to load the event data with proper types
load_event_data <- function(format = "parquet") {{
  
  data_dir <- "{output_dir}"
  
  if (format == "parquet") {{
    df <- arrow::read_parquet(file.path(data_dir, "events_for_r.parquet"))
  }} else if (format == "csv") {{
    df <- readr::read_csv(
      file.path(data_dir, "events_for_r.csv"),
      col_types = cols(
        event_date = col_date(),
        event_num = col_character(),
        event_text = col_character(),
        source_system = col_character(),
        .default = col_character()
      )
    )
  }} else {{
    stop("Format must be 'parquet' or 'csv'")
  }}
  
  # Convert ISO date strings back to dates
  date_cols <- names(df)[str_detect(names(df), "_iso$")]
  for (col in date_cols) {{
    original_col <- str_remove(col, "_iso$")
    df[[original_col]] <- as.Date(df[[col]])
  }}
  
  return(df)
}}

# Load the data
event_data <- load_event_data()

# Display summary
cat("Data loaded successfully!\\n")
cat("Dimensions:", nrow(event_data), "rows x", ncol(event_data), "columns\\n")
cat("Sources:\\n")
print(table(event_data$source_system))
'''.format(
        date=metadata['creation_date'],
        output_dir=str(output_dir).replace('\\', '/')
    )
    
    with open(output_dir / 'load_event_data.R', 'w') as f:
        f.write(r_script)
    
    print(f"Created R import script at {output_dir / 'load_event_data.R'}")


def create_sample_datasets(df: pd.DataFrame, output_dir: Path):
    """
    Create smaller sample datasets for testing and development.
    
    Args:
        df: Full dataframe
        output_dir: Output directory
    """
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    # Random sample (1000 records)
    if len(df) > 1000:
        sample_random = df.sample(n=1000, random_state=42)
        sample_random.to_parquet(samples_dir / 'sample_random_1000.parquet', index=False)
    
    # Recent data sample (last 2 years)
    if 'event_date' in df.columns:
        recent_date = pd.Timestamp.now() - pd.DateOffset(years=2)
        sample_recent = df[df['event_date'] >= recent_date]
        if len(sample_recent) > 0:
            sample_recent.to_parquet(samples_dir / 'sample_recent_2years.parquet', index=False)
    
    # Sample by source (100 from each)
    samples_by_source = []
    for source in df['source_system'].unique():
        source_df = df[df['source_system'] == source]
        n_sample = min(100, len(source_df))
        samples_by_source.append(source_df.sample(n=n_sample, random_state=42))
    
    sample_by_source = pd.concat(samples_by_source, ignore_index=True)
    sample_by_source.to_parquet(samples_dir / 'sample_by_source.parquet', index=False)
    
    print(f"Created sample datasets in {samples_dir}")


def create_summary_report(df: pd.DataFrame, output_dir: Path):
    """
    Create a summary report of the processed data.
    
    Args:
        df: Processed dataframe
        output_dir: Output directory
    """
    report_lines = [
        "# Event Data Processing Summary Report",
        f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Overview",
        f"- Total records: {len(df):,}",
        f"- Total columns: {len(df.columns)}",
        f"- Date range: {df['event_date'].min()} to {df['event_date'].max()}",
        "\n## Records by Source",
    ]
    
    for source, count in df['source_system'].value_counts().items():
        report_lines.append(f"- {source}: {count:,} ({count/len(df)*100:.1f}%)")
    
    report_lines.extend([
        "\n## Data Quality",
        f"- Records with missing dates: {df['event_date'].isna().sum():,}",
        f"- Records with missing event numbers: {df['event_num'].isna().sum():,}",
        f"- Records with missing narratives: {df['event_text'].isna().sum():,}",
    ])
    
    if 'event_text' in df.columns:
        text_lengths = df['event_text'].str.len()
        report_lines.extend([
            "\n## Narrative Text Statistics",
            f"- Average length: {text_lengths.mean():.0f} characters",
            f"- Median length: {text_lengths.median():.0f} characters",
            f"- Shortest: {text_lengths.min():.0f} characters",
            f"- Longest: {text_lengths.max():.0f} characters",
        ])
    
    report_lines.extend([
        "\n## Output Files",
        f"- Main data file: events_for_r.parquet",
        f"- Metadata: data_metadata.json",
        f"- R import script: load_event_data.R",
        f"- Sample datasets: samples/",
    ])
    
    with open(output_dir / 'processing_summary.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Created summary report at {output_dir / 'processing_summary.md'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export processed event data for R modeling")
    parser.add_argument(
        'input_path',
        help='Path to combined processed data (parquet file)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/for_r_modeling',
        help='Output directory for R-ready files'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['parquet', 'rds_prep'],
        choices=['parquet', 'feather', 'csv', 'rds_prep'],
        help='Output formats to generate'
    )
    parser.add_argument(
        '--no-samples',
        action='store_true',
        help='Skip creating sample datasets'
    )
    
    args = parser.parse_args()
    
    export_for_modeling(
        args.input_path,
        args.output_dir,
        formats=args.formats,
        include_samples=not args.no_samples
    )