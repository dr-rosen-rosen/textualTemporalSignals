# PHMSA PDF Processing Guide

## Overview

The PHMSA (Pipeline and Hazardous Materials Safety Administration) PDF processing module extracts structured data from incident report PDFs. It captures:

1. **Narrative Sections**:
   - Part VI: Description of events & package failure
   - Part VII: Recommendations/actions taken to prevent recurrence

2. **Consequences Data** (Part IV):
   - Incident results (spillage, fire, explosion, etc.)
   - Emergency response information
   - Damage costs
   - Casualties (fatalities and injuries)
   - Evacuation details
   - Transportation impacts

## Usage

### Basic Usage

Process PDFs from a directory:
```bash
python scripts/process_phmsa_pdfs.py /path/to/pdf/directory
```

Process PDFs from a CSV file listing:
```bash
python scripts/process_phmsa_pdfs.py file_list.csv
```

### With Existing Data

If you have existing PHMSA data from web scraping:
```bash
python scripts/process_phmsa_pdfs.py /path/to/pdfs --existing-data phmsa_web_data.csv
```

### Resume from Checkpoint

The scraper automatically saves progress. If interrupted, simply run the same command again:
```bash
python scripts/process_phmsa_pdfs.py /path/to/pdfs
```

### Skip PDF Extraction

If you've already extracted PDF data and just want to reprocess:
```bash
python scripts/process_phmsa_pdfs.py dummy --skip-extraction --extracted-file data/temp/phmsa/phmsa_extracted_flat_20240115_143022.parquet
```

## Input Formats

### PDF Directory
Simply point to a directory containing PDF files:
```
pdf_reports/
├── X-2023010218.pdf
├── E-2023010295.pdf
├── I-2023020043.pdf
└── ...
```

### CSV File List
Create a CSV with the following columns:
```csv
pdf_link,report_no
/path/to/X-2023010218.pdf,X-2023010218
/path/to/E-2023010295.pdf,E-2023010295
```

## Output Files

The process creates several output files in the specified output directory:

1. **phmsa_full.parquet**: Complete dataset with all extracted fields
2. **phmsa_consequences_summary.csv**: Summary table of incident consequences
3. **phmsa_text_analysis.parquet**: Subset for text analysis (narratives + incident types)
4. **processing_summary.csv**: Statistics about the processing run

### Temporary Files

During processing, checkpoint files are saved to allow resuming:
- `phmsa_scraping_checkpoint.pkl`: Tracks processed PDFs
- `phmsa_extracted_*.parquet`: Raw extraction results
- `failed_reports.csv`: List of PDFs that couldn't be processed

## Data Schema

### Main Fields

| Field | Type | Description |
|-------|------|-------------|
| report_no | string | Report identifier |
| narrative1 | string | Event description (Part VI) |
| narrative2 | string | Prevention recommendations (Part VII) |
| cmbd_narrative | string | Combined narrative text |
| narrative_word_count | int | Word count of combined narrative |

### Consequences Fields

**Incident Results:**
- incident_results_spillage
- incident_results_fire
- incident_results_explosion
- incident_results_material_entered_waterway
- incident_results_vapor_dispersion
- incident_results_environmental_damage
- incident_results_no_release

**Emergency Response:**
- emergency_response_fire_ems
- emergency_response_police
- emergency_response_in_house_cleanup
- emergency_response_other_cleanup

**Damages (if > $500):**
- damages_material_loss
- damages_carrier_damage
- damages_property_damage
- damages_response_cost
- damages_remediation_cleanup_cost

**Human Impact:**
- fatalities_caused_by_hazmat
- fatalities_employees
- fatalities_responders
- fatalities_general_public
- injuries_caused_by_hazmat
- injuries_hospitalized_*
- injuries_non_hospitalized_*

**Other Impacts:**
- evacuation_occurred
- evacuation_info_total_evacuated
- major_transportation_closed
- crash_or_derailment

## Troubleshooting

### PDF Extraction Errors

Some PDFs may fail to process due to:
- Corrupted PDF files
- Non-standard formatting
- Password protection

Check `failed_reports.csv` for a list of problematic PDFs.

### Memory Issues

For large batches of PDFs:
- Reduce `--save-interval` to save checkpoints more frequently
- Process in smaller batches
- Increase system memory allocation

### Missing Data

If certain fields are consistently missing:
- Check if the PDF format has changed
- Verify the regex patterns in `_extract_*` methods
- Enable debug logging: `--log-level DEBUG`

## Integration with Main Pipeline

After processing PDFs, integrate with the main pipeline:

```python
from src.data_acquisition.readers import PHMSAReader

# Initialize reader
reader = PHMSAReader('data/raw/phmsa')

# Read the integrated data
df = pd.read_parquet('data/processed/phmsa/phmsa_full.parquet')

# Continue with standard cleaning
df_clean = reader.clean_data(df)
```

## Extending the Scraper

To extract additional fields:

1. Add extraction logic to `PHMSAPDFScraper._extract_consequences()`
2. Update the flattening logic in `_flatten_nested_data()`
3. Add new fields to the integration module if needed

Example:
```python
def _extract_custom_field(self, text: str) -> str:
    pattern = re.compile(r"My Field:\s*(.+?)(?:\n|$)")
    match = pattern.search(text)
    return match.group(1).strip() if match else None
```