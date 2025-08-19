# Event Data Processing Pipeline

A Python-based data processing pipeline for harmonizing event reports from multiple safety reporting systems.

## Overview

This pipeline processes event data from four sources:
- **ASRS**: Aviation Safety Reporting System
- **NRC**: Nuclear Regulatory Commission
- **PHMSA**: Pipeline and Hazardous Materials Safety Administration  
- **Rail**: Rail safety reports

The pipeline handles data loading, cleaning, harmonization, and export for downstream analysis in R.

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
project/
├── config.yaml              # Main configuration file
├── requirements.txt         # Python dependencies
├── src/
│   ├── data_acquisition/    # Data loading modules
│   │   └── readers.py       # Source-specific readers
│   └── data_processing/     # Data processing modules
│       └── harmonizer.py    # Data harmonization
├── scripts/
│   ├── run_pipeline.py      # Main pipeline script
│   └── export_for_r.py      # Export data for R analysis
├── data/
│   ├── raw/                 # Raw data files
│   │   ├── asrs/
│   │   ├── nrc/
│   │   ├── phmsa/
│   │   └── rail/
│   └── processed/           # Processed output files
└── R/                       # R modeling scripts
```

## Usage

### Basic Pipeline Execution

Process all data sources:
```bash
python scripts/run_pipeline.py
```

Process specific sources:
```bash
python scripts/run_pipeline.py --sources asrs nrc
```

### Export for R Analysis

After running the pipeline, export data for R:
```bash
python scripts/export_for_r.py data/processed/all_events_combined.parquet
```

### Configuration

Edit `config.yaml` to customize:
- Data paths
- Processing parameters
- Output formats
- Date ranges

## Data Processing Steps

1. **Loading**: Each source has a custom reader that handles its specific format
2. **Cleaning**: Source-specific cleaning rules (date parsing, text cleaning, etc.)
3. **Harmonization**: Standardize column names and data types across sources
4. **Validation**: Check data quality and completeness
5. **Export**: Save in formats optimized for R analysis (parquet, feather, CSV)

## Key Features

- **Modular design**: Easy to add new data sources
- **Source-specific handling**: Custom logic for each data source's quirks
- **Data validation**: Automatic quality checks and reporting
- **R integration**: Seamless handoff to R for modeling
- **Comprehensive logging**: Track all processing steps

## Output Files

The pipeline generates:
- `{source}_harmonized.parquet`: Individual source files
- `all_events_combined.parquet`: Combined data from all sources
- `data_metadata.json`: Metadata about the processed data
- `processing_summary.md`: Summary report
- `load_event_data.R`: R script for loading the data

## Adding New Data Sources

1. Create a new reader class in `src/data_acquisition/readers.py`
2. Add source configuration to `config.yaml`
3. Update the harmonization mappings in `harmonizer.py`
4. Add the source to the pipeline script

## Future Enhancements

- Web scraping modules for NRC and PHMSA (to be added)
- PDF extraction for PHMSA reports
- Additional data quality metrics
- Automated testing suite

## License

[Your license here]

## Contact

[Your contact information]