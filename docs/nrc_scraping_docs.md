# NRC Web Scraping Guide

## Overview

The NRC (Nuclear Regulatory Commission) web scraper extracts two types of data:

1. **LER Event Reports (Primary)**: Detailed event reports from the LER Search system
2. **Power Reactor Status Reports (Secondary)**: Daily status of all US nuclear reactors

## Data Sources

### LER Event Reports (Primary)
- URL: `https://lersearch.inl.gov/ENView.aspx?DOC::[EVENT_NUMBER]`
- Contains detailed incident reports with full narratives
- Requires Selenium for JavaScript-rendered content
- Input: Excel file with event numbers from ENSearchResults

### Power Reactor Status Reports (Secondary)
- URL Pattern: `https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status/YYYY/YYYYMMDD ps.html`
- Contains daily power levels, outage information, and status changes
- Available from 1999 to present
- Can be used to supplement LER data

## Prerequisites

### Selenium WebDriver Setup

The LER scraping requires a browser driver:

**Option 1: Firefox (Recommended)**
```bash
# macOS
brew install geckodriver

# Linux
sudo apt-get install firefox-geckodriver

# Windows
# Download from: https://github.com/mozilla/geckodriver/releases
```

**Option 2: Chrome**
```bash
# macOS
brew install chromedriver

# Linux
sudo apt-get install chromium-chromedriver

# Windows
# Download from: https://chromedriver.chromium.org/
```

### Getting Event Numbers

1. Go to [NRC Event Notification Reports Search](https://www.nrc.gov/reading-rm/doc-collections/event-status/event/)
2. Set your search criteria (date range, facility type, etc.)
3. Export results to Excel (ENSearchResults_power_reactor.xlsx)

## Usage

### Basic LER Event Scraping

Scrape event reports using event numbers file:
```bash
python scripts/scrape_nrc_data.py \
    --event-nums-file ENSearchResults_power_reactor.xlsx \
    --harmonize
```

### Headless Mode (No Browser Window)

Run scraping in background:
```bash
python scripts/scrape_nrc_data.py \
    --event-nums-file ENSearchResults_power_reactor.xlsx \
    --headless \
    --harmonize
```

### Resume Interrupted Scraping

The scraper automatically saves progress. To resume:
```bash
# Just run the same command again
python scripts/scrape_nrc_data.py \
    --event-nums-file ENSearchResults_power_reactor.xlsx
```

### Combine Multiple Data Sources

Scrape both LER events and power status:
```bash
python scripts/scrape_nrc_data.py \
    --event-nums-file ENSearchResults_power_reactor.xlsx \
    --scrape-years 2022 2023 \
    --extract-events \
    --harmonize
```

### Process Existing Scraped Data

Skip scraping and process existing files:
```bash
python scripts/scrape_nrc_data.py \
    --skip-scraping \
    --ler-events-file data/raw/nrc/nrc_event_reports.csv \
    --harmonize
```

## Output Files

The scraper produces several output files:

1. **LER Event Reports**: `nrc_event_reports.csv`
   - Complete event details including narratives
   - All fields from the LER system

2. **Failed Events**: `failed_events.csv`
   - List of event numbers that couldn't be scraped
   - Can be retried later

3. **Checkpoint File**: `nrc_scraping_checkpoint.pkl`
   - Tracks scraping progress
   - Allows resuming interrupted sessions

4. **Harmonized Data**: `nrc_events_harmonized.parquet`
   - Ready for main pipeline integration
   - Standardized column names and formats

## Data Schema

### LER Event Fields

| Field | Description |
|-------|-------------|
| event_num | Event notification number |
| facility | Facility name |
| event_date | Date of event |
| event_time | Time of event |
| notification_date | Date NRC was notified |
| notification_time | Time of notification |
| region | NRC region (1-4) |
| state | State abbreviation |
| emerg_class | Emergency classification |
| 10_cfr_sec | Applicable regulation section |
| rx_type | Reactor type |
| unit | Unit number(s) |
| scram_code | Scram code if applicable |
| rx_crit | Reactor criticality status |
| initial_pwr | Initial power level |
| current_pwr | Current power level |
| initial_rx_mode | Initial reactor mode |
| current_rx_mode | Current reactor mode |
| event_text | Full event narrative |
| last_update_date | Last update to report |

## Troubleshooting

### Selenium Issues

**"No suitable webdriver found"**
- Install Firefox or Chrome driver (see Prerequisites)
- Ensure driver is in system PATH

**Browser crashes or timeouts**
- Increase delay between requests: `--delay 2.0`
- Use headless mode to reduce memory usage
- Check system resources

### Scraping Failures

**Many failed events**
- Check internet connection
- Verify LER Search website is accessible
- Review failed_events.csv for patterns
- Increase delay between requests

**"Element not found" errors**
- Website structure may have changed
- Check if manual access works
- Report issue for script update

### Memory Issues

For large event lists:
- Process in batches (edit Excel file)
- Use checkpoint system (automatic)
- Monitor system memory

## Rate Limiting

The scraper includes polite crawling features:
- Default delay: 1.0 second between requests (LER)
- Configurable with `--delay` parameter
- Automatic saves every 50 events

## Integration with Main Pipeline

After scraping, the harmonized data integrates seamlessly:

```python
# The harmonized file is ready for the main pipeline
# Just run:
python scripts/run_pipeline.py --sources nrc
```

## Advanced Usage

### Custom Event Extraction

To modify which fields are extracted, edit the `_extract_event_data` method in `nrc_web_scraper.py`:

```python
# Add new fields
field_ids = {
    'event_num': "ContentPlaceHolderMainPageContent_Label1EventNumber",
    'new_field': "ContentPlaceHolderMainPageContent_LabelNewField",
    # ...
}
```

### Parallel Processing

The LER scraper uses a single browser instance for stability. For faster processing:
- Split event list into multiple Excel files
- Run multiple instances with different checkpoints
- Merge results afterward

### Debugging Specific Events

To debug a specific event that's failing:

```python
# In Python console
from src.data_acquisition.scrapers.nrc_web_scraper import NRCWebScraper

scraper = NRCWebScraper('debug_output', use_headless=False)
browser = scraper._get_webdriver()

# Navigate to specific event
event_num = "55555"  # Replace with actual event number
browser.get(f"https://lersearch.inl.gov/ENView.aspx?DOC::{event_num}")

# Manually inspect the page
# Check browser console for errors
```

## Workflow Example

Complete workflow from search to pipeline:

1. **Search for Events**:
   - Visit NRC Event Notification Reports Search
   - Set criteria (e.g., Power Reactors, 2020-2023)
   - Export to Excel

2. **Scrape Event Details**:
   ```bash
   python scripts/scrape_nrc_data.py \
       --event-nums-file ENSearchResults_power_reactor.xlsx \
       --output-dir data/raw/nrc
   ```

3. **Review Results**:
   - Check nrc_event_reports.csv for completeness
   - Review failed_events.csv for any issues

4. **Harmonize for Pipeline**:
   ```bash
   python scripts/scrape_nrc_data.py \
       --skip-scraping \
       --ler-events-file data/raw/nrc/nrc_event_reports.csv \
       --harmonize
   ```

5. **Run Main Pipeline**:
   ```bash
   python scripts/run_pipeline.py --sources nrc asrs rail phmsa
   ```

## Performance Considerations

Typical scraping rates:
- LER Events: ~30-60 events per minute (with 1s delay)
- Power Status: ~100-200 pages per minute (with 0.5s delay)

For 10,000 events, expect:
- Time: 3-5 hours
- Checkpoint saves: Every 50 events
- Resume capability: Automatic

## Data Quality Notes

### LER Events
- Most comprehensive event data
- Full narratives and regulatory details
- May have some HTML artifacts in text fields
- Dates are well-structured

### Power Status Reports
- Good for trend analysis
- Limited narrative information
- Useful for identifying unreported events
- Can supplement LER data

## Future Enhancements

Planned improvements:
1. Multi-threaded scraping with browser pool
2. Automatic retry for failed events
3. Direct API access when available
4. Real-time event monitoring
5. Integration with other NRC data sources# NRC Web Scraping Guide

## Overview

The NRC (Nuclear Regulatory Commission) web scraper extracts two types of data:

1. **Power Reactor Status Reports**: Daily status of all US nuclear reactors
2. **Event Notification Reports**: Specific safety events and incidents

## Data Sources

### Power Reactor Status Reports
- URL Pattern: `https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status/YYYY/YYYYMMDD ps.html`
- Contains daily power levels, outage information, and status changes
- Available from 1999 to present

### Event Notification Reports
- Search interface at NRC's Event Notification Report page
- Contains detailed incident reports with narratives
- Searchable by date range, facility, and event type

## Usage

### Basic Power Status Scraping

Scrape power status reports for specific years:
```bash
python scripts/scrape_nrc_data.py --scrape-years 2021 2022 2023
```

### Extract Events from Power Status

Convert power status reports to event records:
```bash
python scripts/scrape_nrc_data.py \
    --power-status-file data/raw/nrc/power_status_raw_2023.csv \
    --extract-events \
    --skip-scraping
```

### Full Pipeline Integration

Scrape, extract events, and harmonize for main pipeline:
```bash
python scripts/scrape_nrc_data.py \
    --scrape-years 2023 \
    --extract-events \
    --harmonize
```

### Combine with Existing Data

Add newly scraped data to existing NRC dataset:
```bash
python scripts/scrape_nrc_data.py \
    --scrape-years 2023 \
    --existing-data data/raw/nrc/nrc_events.csv \
    --extract-events
```

## Output Files

The scraper produces several output files:

1. **Raw Power Status Data**: `power_status_raw_YYYY.csv`
   - Direct scrape of power reactor status tables
   - One row per reactor per day

2. **Extracted Events**: `power_status_events.csv`
   - Events extracted from power status (shutdowns, scrams, etc.)
   - Filtered to include only notable occurrences

3. **Combined Events**: `nrc_events_combined.csv`
   - All NRC events from various sources
   - Deduplicated by event number

4. **Harmonized Data**: `nrc_events_harmonized.parquet`
   - Ready for main pipeline integration
   - Standardized column names and formats

## Event Extraction Logic

Events are extracted from power status reports when:
- Reactor is down (down days > 0)
- Scrams occurred (num_scrams > 0)
- Changes reported in status
- Power level below 90% (significant reduction)

Each extracted event includes:
- Event date and facility
- Power level and days down
- Narrative from reason/comment field
- Generated event number (format: `PS-YYYYMMDD-UNIT_NAME`)

## Checkpoint System

The scraper automatically saves progress to resume interrupted scraping:
- Checkpoint file: `nrc_scraping_checkpoint.pkl`
- Tracks processed URLs and events
- Automatically resumes from last position

To restart from scratch, delete the checkpoint file:
```bash
rm data/raw/nrc/nrc_scraping_checkpoint.pkl
```

## Rate Limiting

The scraper includes polite crawling features:
- Default delay: 0.5 seconds between requests
- Retry logic for failed requests
- Adjustable delay with `--delay` parameter

## Data Schema

### Power Status Fields

| Field | Description |
|-------|-------------|
| url | Source URL |
| report_date | Date of status report |
| region | NRC region (1-4) |
| unit | Reactor unit name |
| power | Power level (%) |
| down | Days down |
| reason_comment | Reason for status/comment |
| change_in_report | Changes from previous |
| num_scrams | Number of scrams |

### Event Fields (after extraction)

| Field | Description |
|-------|-------------|
| event_num | Unique event identifier |
| event_date | Date of event |
| facility | Facility name |
| region | NRC region |
| event_type | Type of event |
| power_level | Power level at time |
| days_down | Duration of outage |
| event_text | Event narrative |
| source_url | Original data source |

## Troubleshooting

### Connection Errors

If experiencing connection issues:
- Increase delay between requests: `--delay 2.0`
- Check NRC website availability
- Review bad_links_power_status.csv for patterns

### Memory Issues

For large date ranges:
- Process one year at a time
- Use checkpoint system to process in batches
- Monitor memory usage with system tools

### Data Quality

Common issues and solutions:
- **Invalid dates**: Script automatically skips invalid dates (e.g., Feb 31)
- **Missing data**: Check bad_links file for failed URLs
- **Duplicate events**: Deduplication is automatic based on event_num

## Integration with Main Pipeline

After scraping, integrate with the main pipeline:

```python
# In your main pipeline script
from src.data_acquisition.readers import NRCReader

# Read the scraped and harmonized data
nrc_data = pd.read_parquet('data/raw/nrc/nrc_events_harmonized.parquet')

# Continue with standard pipeline processing
```

## Advanced Usage

### Custom Event Extraction

To modify event extraction criteria, edit the `process_power_status_to_events` function:

```python
# Example: Only extract complete shutdowns
if row['power'] == '0' or row['down'] != '0':
    # Create event record
```

### Parallel Scraping

For faster scraping (use responsibly):

```python
# In nrc_web_scraper.py, modify to use ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {executor.submit(self._scrape_power_status_page, url): url 
               for url in urls}
```

## Future Enhancements

Planned improvements:
1. Implement event notification search and scraping
2. Add support for other NRC data sources
3. Automatic detection of new report formats
4. Enhanced event categorization
5. Integration with NRC's API (when available)