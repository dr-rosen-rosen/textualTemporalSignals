#!/usr/bin/env python
"""
Script to scrape NRC data and prepare it for the main pipeline.

This script handles:
1. Scraping LER (Licensee Event Report) event data (primary)
2. Scraping power reactor status reports (optional)
3. Combining with existing data
4. Preparing for main pipeline integration
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import argparse
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_acquisition.scrapers.nrc_web_scraper import NRCWebScraper
from src.data_acquisition.readers import NRCReader
from src.data_processing.harmonizer import DataHarmonizer


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('nrc_scraping.log')
        ]
    )


def process_power_status_to_events(power_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert power status reports to event-like format.
    
    Power status reports contain daily reactor status, but we want to
    extract notable events (shutdowns, power reductions, scrams).
    
    Args:
        power_df: DataFrame with power status data
        
    Returns:
        DataFrame with event-like records
    """
    logger = logging.getLogger(__name__)
    
    events = []
    
    # Group by unit and date
    grouped = power_df.groupby(['unit', 'report_date'])
    
    for (unit, date), group in grouped:
        for _, row in group.iterrows():
            # Check if this represents an event
            if (row['down'] != '0' or 
                row['num_scrams'] != '0' or 
                row['change_in_report'] != '' or
                int(row['power']) < 90):  # Significant power reduction
                
                # Create event record
                event = {
                    'event_date': date,
                    'facility': unit,
                    'region': row['region'],
                    'event_type': 'power_status',
                    'power_level': row['power'],
                    'days_down': row['down'],
                    'num_scrams': row['num_scrams'],
                    'event_text': row['reason_comment'],
                    'change_note': row['change_in_report'],
                    'source_url': row['url']
                }
                
                # Generate event number
                event['event_num'] = f"PS-{date.replace('-', '')}-{unit.replace(' ', '_')}"
                
                events.append(event)
    
    events_df = pd.DataFrame(events)
    logger.info(f"Extracted {len(events_df)} events from power status reports")
    
    return events_df


def main():
    parser = argparse.ArgumentParser(
        description="Scrape NRC data and prepare for pipeline"
    )
    
    # LER Event scraping options (primary)
    parser.add_argument(
        '--event-nums-file',
        help='Excel file with event numbers from ENSearchResults (primary data source)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run browser in headless mode for LER scraping'
    )
    
    # Power status scraping options (secondary)
    parser.add_argument(
        '--scrape-years',
        nargs='+',
        type=int,
        help='Years to scrape power status reports (e.g., 2020 2021 2022)'
    )
    
    # Data handling options
    parser.add_argument(
        '--existing-data',
        help='Path to existing NRC data file to combine with'
    )
    parser.add_argument(
        '--output-dir',
        default='data/raw/nrc',
        help='Output directory for scraped data'
    )
    parser.add_argument(
        '--skip-scraping',
        action='store_true',
        help='Skip scraping and just process existing files'
    )
    parser.add_argument(
        '--ler-events-file',
        help='Existing LER events file to process'
    )
    parser.add_argument(
        '--power-status-file',
        help='Existing power status file to process'
    )
    
    # Processing options
    parser.add_argument(
        '--extract-events',
        action='store_true',
        help='Extract events from power status reports'
    )
    parser.add_argument(
        '--harmonize',
        action='store_true',
        help='Harmonize data for main pipeline'
    )
    
    # Other options
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between web requests in seconds'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Scrape or load data
    ler_events_df = None
    power_df = None
    events_df = None
    
    if not args.skip_scraping:
        # Initialize scraper
        scraper = NRCWebScraper(
            output_dir, 
            delay_between_requests=args.delay,
            use_headless=args.headless
        )
        
        # Scrape LER events (primary data source)
        if args.event_nums_file:
            logger.info(f"Scraping LER events from {args.event_nums_file}")
            ler_events_df = scraper.scrape_ler_events(args.event_nums_file)
            
            # Save raw scraped data
            filename = "nrc_event_reports.csv"
            ler_events_df.to_csv(output_dir / filename, index=False)
            logger.info(f"Saved LER event data to {filename}")
            
            # This is our primary event data
            events_df = ler_events_df
        
        # Scrape power status reports (secondary/supplementary)
        if args.scrape_years:
            logger.info(f"Scraping power status reports for years: {args.scrape_years}")
            power_df = scraper.scrape_power_status_reports(args.scrape_years)
            
            # Save raw scraped data
            filename = f"power_status_raw_{'_'.join(map(str, args.scrape_years))}.csv"
            power_df.to_csv(output_dir / filename, index=False)
            logger.info(f"Saved raw power status data to {filename}")
    
    else:
        # Load existing files
        if args.ler_events_file:
            ler_events_df = pd.read_csv(args.ler_events_file)
            logger.info(f"Loaded {len(ler_events_df)} LER events from {args.ler_events_file}")
            events_df = ler_events_df
            
        if args.power_status_file:
            power_df = pd.read_csv(args.power_status_file)
            logger.info(f"Loaded {len(power_df)} power status records from {args.power_status_file}")
    
    # Step 2: Extract events from power status if requested
    if args.extract_events and power_df is not None:
        logger.info("Extracting events from power status reports...")
        power_events_df = process_power_status_to_events(power_df)
        
        # Save extracted events
        filename = "power_status_events.csv"
        power_events_df.to_csv(output_dir / filename, index=False)
        logger.info(f"Saved {len(power_events_df)} extracted events to {filename}")
        
        # Add to events dataframe
        if events_df is None:
            events_df = power_events_df
        else:
            events_df = pd.concat([events_df, power_events_df], ignore_index=True)
    
    # Step 3: Combine with existing data if provided
    if args.existing_data and events_df is not None:
        logger.info(f"Combining with existing data from {args.existing_data}")
        existing_df = pd.read_csv(args.existing_data)
        
        # Combine dataframes
        combined_df = pd.concat([existing_df, events_df], ignore_index=True)
        
        # Remove duplicates
        if 'event_num' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['event_num'], keep='last')
        
        events_df = combined_df
        logger.info(f"Combined dataset has {len(events_df)} total events")
    
    # Step 4: Clean and harmonize if requested
    if args.harmonize and events_df is not None:
        logger.info("Cleaning and harmonizing NRC data...")
        
        # Use NRC reader for cleaning
        reader = NRCReader(output_dir)
        events_df = reader.clean_data(events_df)
        
        # Harmonize for pipeline
        harmonizer = DataHarmonizer()
        events_df = harmonizer.harmonize(events_df, 'nrc')
        
        # Save harmonized data
        filename = "nrc_events_harmonized.parquet"
        events_df.to_parquet(output_dir / filename, index=False)
        logger.info(f"Saved harmonized data to {filename}")
    
    # Final save of all events
    if events_df is not None and not args.harmonize:
        filename = "nrc_events_combined.csv"
        events_df.to_csv(output_dir / filename, index=False)
        logger.info(f"Saved combined events to {filename}")
    
    # Print summary
    print("\n" + "="*50)
    print("NRC DATA SCRAPING COMPLETE")
    print("="*50)
    
    if ler_events_df is not None:
        print(f"LER Event Records: {len(ler_events_df):,}")
        
    if power_df is not None:
        print(f"Power Status Records: {len(power_df):,}")
        if args.extract_events:
            print(f"Events Extracted: {len(power_events_df):,}")
    
    if events_df is not None:
        print(f"Total Event Records: {len(events_df):,}")
        if 'event_date' in events_df.columns:
            print(f"Date Range: {events_df['event_date'].min()} to {events_df['event_date'].max()}")
    
    print("="*50)
    print(f"\nOutput files saved to: {output_dir}")
    
    # Print instructions for next steps
    if ler_events_df is not None and not args.harmonize:
        print("\nNext steps:")
        print("1. Review the scraped data in nrc_event_reports.csv")
        print("2. Run with --harmonize flag to prepare for main pipeline")
        print("3. Or proceed directly to the main pipeline")
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()