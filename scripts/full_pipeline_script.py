#!/usr/bin/env python
"""
Full pipeline orchestration script.

This script coordinates all data acquisition, processing, and export steps:
1. Web scraping (NRC)
2. PDF processing (PHMSA)
3. Data loading and cleaning (all sources)
4. Harmonization and combination
5. Export for R modeling
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import argparse
from datetime import datetime
import yaml
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f'full_pipeline_{timestamp}.log')
        ]
    )


def run_subprocess(command: list, description: str) -> bool:
    """
    Run a subprocess and capture output.
    
    Args:
        command: Command and arguments as list
        description: Description for logging
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting: {description}")
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Completed: {description}")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run full event data processing pipeline"
    )
    
    # Pipeline stages
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['scrape', 'pdf', 'process', 'export', 'all'],
        default=['all'],
        help='Pipeline stages to run'
    )
    
    # Data sources
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=['nrc', 'phmsa', 'asrs', 'rail', 'all'],
        default=['all'],
        help='Data sources to process'
    )
    
    # Scraping options
    parser.add_argument(
        '--scrape-years',
        nargs='+',
        type=int,
        help='Years to scrape for NRC data'
    )
    parser.add_argument(
        '--phmsa-pdfs',
        help='Directory containing PHMSA PDF files'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed',
        help='Output directory for final data'
    )
    
    # Options
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip stages if output files already exist'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without executing'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("STARTING FULL EVENT DATA PIPELINE")
    logger.info("="*60)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine stages to run
    stages = set(args.stages)
    if 'all' in stages:
        stages = {'scrape', 'pdf', 'process', 'export'}
    
    # Determine sources to process
    sources = set(args.sources)
    if 'all' in sources:
        sources = {'nrc', 'phmsa', 'asrs', 'rail'}
    
    # Track success
    success = True
    
    # Stage 1: Web Scraping (NRC)
    if 'scrape' in stages and 'nrc' in sources:
        if args.scrape_years:
            logger.info("Stage 1: Scraping NRC data")
            
            command = [
                sys.executable,
                'scripts/scrape_nrc_data.py',
                '--scrape-years'] + [str(y) for y in args.scrape_years] + [
                '--extract-events',
                '--harmonize'
            ]
            
            if args.dry_run:
                logger.info(f"Would run: {' '.join(command)}")
            else:
                success &= run_subprocess(command, "NRC web scraping")
        else:
            logger.warning("Skipping NRC scraping - no years specified")
    
    # Stage 2: PDF Processing (PHMSA)
    if 'pdf' in stages and 'phmsa' in sources:
        if args.phmsa_pdfs:
            logger.info("Stage 2: Processing PHMSA PDFs")
            
            command = [
                sys.executable,
                'scripts/process_phmsa_pdfs.py',
                args.phmsa_pdfs,
                '--output-dir', 'data/processed/phmsa'
            ]
            
            if args.dry_run:
                logger.info(f"Would run: {' '.join(command)}")
            else:
                success &= run_subprocess(command, "PHMSA PDF processing")
        else:
            logger.warning("Skipping PHMSA PDF processing - no PDF directory specified")
    
    # Stage 3: Main Processing Pipeline
    if 'process' in stages:
        logger.info("Stage 3: Running main processing pipeline")
        
        command = [
            sys.executable,
            'scripts/run_pipeline.py',
            '--config', args.config
        ]
        
        if sources != {'nrc', 'phmsa', 'asrs', 'rail'}:
            command.extend(['--sources'] + list(sources))
        
        if args.dry_run:
            logger.info(f"Would run: {' '.join(command)}")
        else:
            success &= run_subprocess(command, "Main data processing")
    
    # Stage 4: Export for R
    if 'export' in stages:
        logger.info("Stage 4: Exporting data for R modeling")
        
        # Find the combined data file
        combined_file = Path(config['data_paths']['processed_data']) / 'all_events_combined.parquet'
        
        if combined_file.exists():
            command = [
                sys.executable,
                'scripts/export_for_r.py',
                str(combined_file),
                '--output-dir', 'data/for_r_modeling'
            ]
            
            if args.dry_run:
                logger.info(f"Would run: {' '.join(command)}")
            else:
                success &= run_subprocess(command, "R export preparation")
        else:
            logger.error(f"Combined data file not found: {combined_file}")
            success = False
    
    # Summary
    logger.info("="*60)
    if success:
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    else:
        logger.error("PIPELINE COMPLETED WITH ERRORS")
    logger.info("="*60)
    
    # Print summary statistics if not dry run
    if not args.dry_run and 'process' in stages:
        try:
            # Load final combined data
            combined_file = Path(args.output_dir) / 'all_events_combined.parquet'
            if combined_file.exists():
                df = pd.read_parquet(combined_file)
                
                print("\n" + "="*50)
                print("FINAL DATASET SUMMARY")
                print("="*50)
                print(f"Total Records: {len(df):,}")
                
                if 'source_system' in df.columns:
                    print("\nRecords by Source:")
                    for source, count in df['source_system'].value_counts().items():
                        print(f"  {source}: {count:,}")
                
                if 'event_date' in df.columns:
                    print(f"\nDate Range: {df['event_date'].min()} to {df['event_date'].max()}")
                
                print("="*50)
        except Exception as e:
            logger.error(f"Could not load summary statistics: {e}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())