"""
NRC (Nuclear Regulatory Commission) Web Scraper

Scrapes event data from NRC's LER Search system and other sources:
1. Event notification reports from LER Search (requires Selenium)
2. Power reactor status reports (optional)
"""

import re
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests

logger = logging.getLogger(__name__)


class NRCWebScraper:
    """Scrape event data from NRC websites."""
    
    def __init__(self, 
                 output_dir: Union[str, Path],
                 checkpoint_file: str = "nrc_scraping_checkpoint.pkl",
                 delay_between_requests: float = 1.0,
                 use_headless: bool = True):
        """
        Initialize the NRC scraper.
        
        Args:
            output_dir: Directory to save scraped data
            checkpoint_file: File to save progress for resuming
            delay_between_requests: Delay in seconds between requests
            use_headless: Whether to run browser in headless mode
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / checkpoint_file
        self.delay = delay_between_requests
        self.use_headless = use_headless
        
        # Load checkpoint if exists
        self.checkpoint = self._load_checkpoint()
        
    def _load_checkpoint(self) -> Dict:
        """Load checkpoint data if it exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return {'processed_events': set(), 'failed_events': set(), 'data': []}
    
    def _save_checkpoint(self):
        """Save current progress."""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(self.checkpoint, f)
    
    def _get_webdriver(self):
        """Create and configure a Selenium webdriver."""
        # Try Firefox first, then Chrome
        try:
            options = webdriver.FirefoxOptions()
            if self.use_headless:
                options.add_argument('--headless')
            return webdriver.Firefox(options=options)
        except Exception as e:
            logger.warning(f"Firefox driver failed: {e}. Trying Chrome...")
            try:
                options = webdriver.ChromeOptions()
                if self.use_headless:
                    options.add_argument('--headless')
                return webdriver.Chrome(options=options)
            except Exception as e2:
                logger.error(f"Chrome driver also failed: {e2}")
                raise RuntimeError("No suitable webdriver found. Please install Firefox or Chrome driver.")
    
    def scrape_ler_events(self, 
                         event_nums_file: Union[str, Path],
                         save_interval: int = 50) -> pd.DataFrame:
        """
        Scrape event details from LER Search system using event numbers.
        
        Args:
            event_nums_file: Excel file with event numbers (from ENSearchResults)
            save_interval: Save checkpoint every N events
            
        Returns:
            DataFrame with scraped event data
        """
        logger.info(f"Loading event numbers from {event_nums_file}")
        
        # Load event numbers
        event_nums_df = pd.read_excel(event_nums_file, skiprows=4, usecols=['Event Number'])
        event_nums = event_nums_df['Event Number'].tolist()
        
        logger.info(f"Found {len(event_nums)} event numbers to scrape")
        
        # Filter out already processed events
        events_to_process = [e for e in event_nums if e not in self.checkpoint['processed_events']]
        logger.info(f"{len(events_to_process)} events remaining to process")
        
        events = []
        browser = None
        
        try:
            # Initialize browser once
            browser = self._get_webdriver()
            wait = WebDriverWait(browser, 10)
            
            for i, event_num in enumerate(events_to_process):
                try:
                    # Add delay to be respectful
                    time.sleep(self.delay)
                    
                    logger.debug(f"Scraping event {event_num} ({i+1}/{len(events_to_process)})")
                    
                    # Navigate to event page
                    url = f"https://lersearch.inl.gov/ENView.aspx?DOC::{event_num}"
                    browser.get(url)
                    
                    # Wait for page to load
                    wait.until(EC.presence_of_element_located((By.ID, "ContentPlaceHolderMainPageContent_Label1EventNumber")))
                    
                    # Extract event data
                    event_data = self._extract_event_data(browser)
                    event_data['url'] = url
                    events.append(event_data)
                    
                    # Update checkpoint
                    self.checkpoint['processed_events'].add(event_num)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(events_to_process)} events")
                    
                    # Save checkpoint periodically
                    if (i + 1) % save_interval == 0:
                        self._save_checkpoint()
                        # Also save intermediate results
                        self._save_intermediate_results(events)
                        
                except TimeoutException:
                    logger.error(f"Timeout loading event {event_num}")
                    self.checkpoint['failed_events'].add(event_num)
                except Exception as e:
                    logger.error(f"Error scraping event {event_num}: {str(e)}")
                    self.checkpoint['failed_events'].add(event_num)
            
        finally:
            if browser:
                browser.quit()
            
            # Save final checkpoint and results
            self._save_checkpoint()
            if events:
                self._save_intermediate_results(events)
        
        # Convert to DataFrame
        df = pd.DataFrame(events)
        
        # Save failed events
        if self.checkpoint['failed_events']:
            failed_df = pd.DataFrame({'event_num': list(self.checkpoint['failed_events'])})
            failed_df.to_csv(self.output_dir / 'failed_events.csv', index=False)
            logger.warning(f"Failed to scrape {len(self.checkpoint['failed_events'])} events. See failed_events.csv")
        
        return df
    
    def _extract_event_data(self, browser) -> Dict:
        """Extract all event data from the current page."""
        soup = BeautifulSoup(browser.page_source, features="html.parser")
        
        # Define field mappings
        field_ids = {
            'event_num': "ContentPlaceHolderMainPageContent_Label1EventNumber",
            'facility': "ContentPlaceHolderMainPageContent_Label1Facility",
            'emerg_class': "ContentPlaceHolderMainPageContent_Label1EmergencyClass",
            '10_cfr_sec': "ContentPlaceHolderMainPageContent_Label110CFRSection",
            'last_update_date': "ContentPlaceHolderMainPageContent_Label1LastUpdateDate",
            'event_date': "ContentPlaceHolderMainPageContent_Label1EventDate",
            'event_time': "ContentPlaceHolderMainPageContent_Label1EventTime",
            'notification_date': "ContentPlaceHolderMainPageContent_Label1NotificationDate",
            'notification_time': "ContentPlaceHolderMainPageContent_Label1NotificationTime",
            'region': "ContentPlaceHolderMainPageContent_Label1Region",
            'state': "ContentPlaceHolderMainPageContent_Label1State",
            'rx_type': "ContentPlaceHolderMainPageContent_Label1RXType",
            'unit': "ContentPlaceHolderMainPageContent_LabelUnit1",
            'scram_code': "ContentPlaceHolderMainPageContent_LabelSCRAMCode1",
            'rx_crit': "ContentPlaceHolderMainPageContent_LabelRXCrit1",
            'initial_pwr': "ContentPlaceHolderMainPageContent_LabelInitialPower1",
            'initial_rx_mode': "ContentPlaceHolderMainPageContent_LabelInitialRXMode1",
            'current_pwr': "ContentPlaceHolderMainPageContent_LabelCurrentPower1",
            'current_rx_mode': "ContentPlaceHolderMainPageContent_LabelCurrentRXMode1",
            'event_text': "ContentPlaceHolderMainPageContent_LabelEventText"
        }
        
        # Extract data
        event_data = {}
        for field_name, element_id in field_ids.items():
            element = soup.find(id=element_id)
            if element:
                event_data[field_name] = element.text.strip()
            else:
                event_data[field_name] = None
                logger.warning(f"Could not find element {element_id} for field {field_name}")
        
        return event_data
    
    def _save_intermediate_results(self, events: List[Dict]):
        """Save intermediate results to avoid data loss."""
        df = pd.DataFrame(events)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"nrc_events_intermediate_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.debug(f"Saved intermediate results to {filename}")
    
    def scrape_power_status_reports(self, 
                                   years: List[int], 
                                   save_interval: int = 100) -> pd.DataFrame:
        """
        Scrape power reactor status reports for specified years.
        This is kept for backward compatibility but is not the primary data source.
        
        Args:
            years: List of years to scrape
            save_interval: Save checkpoint every N URLs
            
        Returns:
            DataFrame with scraped power status data
        """
        logger.info(f"Starting to scrape power status reports for years: {years}")
        
        # Generate URLs
        urls = self._generate_power_status_urls(years)
        logger.info(f"Generated {len(urls)} URLs to scrape")
        
        rows = []
        bad_links = []
        
        session = requests.Session()
        
        for i, url in enumerate(urls):
            # Skip if already processed
            if url in self.checkpoint.get('processed_urls', set()):
                continue
                
            try:
                # Add delay to be respectful
                time.sleep(self.delay)
                
                # Scrape the page
                response = session.get(url, timeout=30)
                response.raise_for_status()
                
                data = self._scrape_power_status_page(url, response.text)
                rows.extend(data)
                
                # Update checkpoint
                if 'processed_urls' not in self.checkpoint:
                    self.checkpoint['processed_urls'] = set()
                self.checkpoint['processed_urls'].add(url)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(urls)} URLs")
                
                # Save checkpoint periodically
                if (i + 1) % save_interval == 0:
                    self._save_checkpoint()
                    
            except Exception as e:
                logger.error(f"Error scraping {url}: {str(e)}")
                bad_links.append(url)
        
        # Final save
        self._save_checkpoint()
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Save bad links
        if bad_links:
            bad_links_df = pd.DataFrame({'url': bad_links})
            bad_links_df.to_csv(self.output_dir / 'bad_links_power_status.csv', index=False)
            logger.warning(f"Failed to scrape {len(bad_links)} URLs. See bad_links_power_status.csv")
        
        return df
    
    def _generate_power_status_urls(self, years: List[int]) -> List[str]:
        """Generate URLs for power reactor status reports."""
        root = 'https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status/'
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        days = [f"{d:02d}" for d in range(1, 32)]
        
        urls = []
        for year in years:
            for month in months:
                for day in days:
                    # Skip invalid dates
                    try:
                        datetime(year, int(month), int(day))
                        url = f"{root}{year}/{year}{month}{day}ps.html"
                        urls.append(url)
                    except ValueError:
                        # Invalid date (e.g., Feb 30)
                        continue
        
        return urls
    
    def _scrape_power_status_page(self, url: str, html: str) -> List[Dict]:
        """Scrape a single power status page."""
        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all('table')
        
        rows = []
        for table in tables:
            # Get table summary (region info)
            summary = table.get('summary', '')
            
            # Extract date from URL
            date_match = re.search(r'/(\d{4})(\d{2})(\d{2})ps\.html', url)
            if date_match:
                year, month, day = date_match.groups()
                report_date = f"{year}-{month}-{day}"
            else:
                report_date = None
            
            # Parse table rows
            for row in table.find_all('tr'):
                cols = row.find_all('td')
                if len(cols) == 6:  # Valid data row
                    data = {
                        'url': url,
                        'report_date': report_date,
                        'region': summary,
                        'unit': cols[0].text.strip(),
                        'power': cols[1].text.strip(),
                        'down': cols[2].text.strip(),
                        'reason_comment': cols[3].text.strip(),
                        'change_in_report': cols[4].text.strip(),
                        'num_scrams': cols[5].text.strip()
                    }
                    rows.append(data)
        
        return rows
    
    def export_results(self, df: pd.DataFrame, filename: str, format: str = 'csv'):
        """
        Export scraped results to file.
        
        Args:
            df: DataFrame with scraped data
            filename: Output filename (without extension)
            format: Output format ('parquet', 'csv', 'json')
        """
        output_file = self.output_dir / f"{filename}.{format}"
        
        if format == 'parquet':
            df.to_parquet(output_file, index=False)
        elif format == 'csv':
            df.to_csv(output_file, index=False)
        elif format == 'json':
            df.to_json(output_file, orient='records', indent=2)
        
        logger.info(f"Exported {len(df)} records to {output_file}")


def main():
    """Example usage of the NRC scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape data from NRC websites")
    parser.add_argument('--output-dir', default='data/raw/nrc', 
                       help='Output directory for scraped data')
    
    # LER Search options
    parser.add_argument('--event-nums-file', 
                       help='Excel file with event numbers from ENSearchResults')
    parser.add_argument('--headless', action='store_true',
                       help='Run browser in headless mode')
    
    # Power status options
    parser.add_argument('--years', nargs='+', type=int,
                       help='Years to scrape power status reports')
    
    # General options
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between requests in seconds')
    parser.add_argument('--format', choices=['parquet', 'csv', 'json'], 
                       default='csv', help='Output format')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize scraper
    scraper = NRCWebScraper(
        args.output_dir, 
        delay_between_requests=args.delay,
        use_headless=args.headless
    )
    
    # Scrape LER events if file specified
    if args.event_nums_file:
        logger.info(f"Scraping LER events from {args.event_nums_file}")
        events_df = scraper.scrape_ler_events(args.event_nums_file)
        
        # Export results
        scraper.export_results(
            events_df,
            "nrc_event_reports",
            format=args.format
        )
        
        print(f"\nScraped {len(events_df)} event reports")
    
    # Scrape power status reports if years specified
    if args.years:
        logger.info(f"Scraping power status reports for years: {args.years}")
        power_df = scraper.scrape_power_status_reports(args.years)
        
        # Export results
        scraper.export_results(
            power_df, 
            f"nrc_power_status_{'_'.join(map(str, args.years))}", 
            format=args.format
        )
        
        print(f"\nScraped {len(power_df)} power status records")
    
    logger.info("Scraping complete!")


if __name__ == "__main__":
    main() filename: str, format: str = 'parquet'):
        """
        Export scraped results to file.
        
        Args:
            df: DataFrame with scraped data
            filename: Output filename (without extension)
            format: Output format ('parquet', 'csv', 'json')
        """
        output_file = self.output_dir / f"{filename}.{format}"
        
        if format == 'parquet':
            df.to_parquet(output_file, index=False)
        elif format == 'csv':
            df.to_csv(output_file, index=False)
        elif format == 'json':
            df.to_json(output_file, orient='records', indent=2)
        
        logger.info(f"Exported {len(df)} records to {output_file}")
    
    def combine_with_existing(self, 
                            scraped_df: pd.DataFrame,
                            existing_file: Union[str, Path]) -> pd.DataFrame:
        """
        Combine newly scraped data with existing NRC data file.
        
        Args:
            scraped_df: Newly scraped data
            existing_file: Path to existing NRC data
            
        Returns:
            Combined DataFrame
        """
        existing_df = pd.read_csv(existing_file)
        
        # Standardize column names if needed
        if 'event_text' not in scraped_df.columns and 'reason_comment' in scraped_df.columns:
            scraped_df['event_text'] = scraped_df['reason_comment']
        
        # Combine the dataframes
        combined = pd.concat([existing_df, scraped_df], ignore_index=True)
        
        # Remove duplicates based on key fields
        if 'event_num' in combined.columns:
            combined = combined.drop_duplicates(subset=['event_num'], keep='last')
        
        logger.info(f"Combined data: {len(combined)} total records")
        
        return combined


def main():
    """Example usage of the NRC scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape data from NRC websites")
    parser.add_argument('--output-dir', default='data/raw/nrc', 
                       help='Output directory for scraped data')
    parser.add_argument('--years', nargs='+', type=int,
                       help='Years to scrape power status reports')
    parser.add_argument('--start-date', help='Start date for event notifications (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for event notifications (YYYY-MM-DD)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between requests in seconds')
    parser.add_argument('--format', choices=['parquet', 'csv', 'json'], 
                       default='csv', help='Output format')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize scraper
    scraper = NRCWebScraper(args.output_dir, delay_between_requests=args.delay)
    
    # Scrape power status reports if years specified
    if args.years:
        logger.info(f"Scraping power status reports for years: {args.years}")
        power_df = scraper.scrape_power_status_reports(args.years)
        
        # Export results
        scraper.export_results(
            power_df, 
            f"nrc_power_status_{'_'.join(map(str, args.years))}", 
            format=args.format
        )
        
        print(f"\nScraped {len(power_df)} power status records")
    
    # Scrape event notifications if dates specified
    if args.start_date and args.end_date:
        logger.info(f"Scraping event notifications from {args.start_date} to {args.end_date}")
        events_df = scraper.scrape_event_notifications(args.start_date, args.end_date)
        
        # Export results
        scraper.export_results(
            events_df,
            f"nrc_events_{args.start_date}_{args.end_date}",
            format=args.format
        )
        
        print(f"\nScraped {len(events_df)} event notifications")
    
    logger.info("Scraping complete!")


if __name__ == "__main__":
    main()