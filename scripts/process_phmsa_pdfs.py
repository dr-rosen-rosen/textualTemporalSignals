#!/usr/bin/env python
"""
Script to process PHMSA PDF reports and integrate with the main pipeline.

This script:
1. Extracts data from PHMSA PDF reports
2. Combines with existing PHMSA data (if available)
3. Prepares the data for the main pipeline
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_acquisition.scrapers.phmsa_pdf_scraper import PHMSAPDFScraper
from src.data_processing.phmsa_integration import integrate_phmsa_data


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('phmsa_processing.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process PHMSA PDF reports and prepare for main pipeline"
    )
    
    # Required arguments
    parser.add_argument(
        'pdf_source',
        help='Either a directory containing PDFs or a CSV file with pdf_link and report_no columns'
    )
    
    # Optional arguments
    parser.add_argument(
        '--existing-data',
        help='Path to existing PHMSA data CSV (from web scraping)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed/phmsa',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--temp-dir',
        default='data/temp/phmsa',
        help='Temporary directory for PDF extraction checkpoints'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=50,
        help='Save checkpoint every N PDFs'
    )
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip PDF extraction and use existing extracted data'
    )
    parser.add_argument(
        '--extracted-file',
        help='Path to existing extracted data file (if using --skip-extraction)'
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
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract data from PDFs (unless skipped)
    if not args.skip_extraction:
        logger.info("Starting PDF extraction...")
        
        # Determine PDF source
        pdf_source_path = Path(args.pdf_source)
        
        if pdf_source_path.is_dir():
            # Directory of PDFs
            pdf_files = list(pdf_source_path.glob('*.pdf'))
            report_ids = None
            logger.info(f"Found {len(pdf_files)} PDF files in {pdf_source_path}")
            
        elif pdf_source_path.suffix == '.csv':
            # CSV file with pdf paths and report numbers
            file_list = pd.read_csv(pdf_source_path)
            
            if 'pdf_link' not in file_list.columns or 'report_no' not in file_list.columns:
                raise ValueError("CSV must contain 'pdf_link' and 'report_no' columns")
            
            pdf_files = file_list['pdf_link'].tolist()
            report_ids = file_list['report_no'].tolist()
            logger.info(f"Loaded {len(pdf_files)} PDF paths from CSV")
            
        else:
            raise ValueError(f"pdf_source must be a directory or CSV file, got: {pdf_source_path}")
        
        # Initialize scraper
        scraper = PHMSAPDFScraper(output_dir=temp_dir)
        
        # Process PDFs
        results_df = scraper.process_pdf_list(
            pdf_files,
            report_ids=report_ids,
            save_interval=args.save_interval
        )
        
        # Export extracted data
        scraper.export_results(results_df, format='parquet')
        
        # Get the path to the flattened extracted file
        extracted_files = list(temp_dir.glob('phmsa_extracted_flat_*.parquet'))
        if not extracted_files:
            raise FileNotFoundError("No extracted data file found")
        
        extracted_file = sorted(extracted_files)[-1]  # Get most recent
        logger.info(f"Using extracted data file: {extracted_file}")
        
    else:
        # Use existing extracted file
        if not args.extracted_file:
            raise ValueError("--extracted-file must be provided when using --skip-extraction")
        
        extracted_file = Path(args.extracted_file)
        if not extracted_file.exists():
            raise FileNotFoundError(f"Extracted file not found: {extracted_file}")
        
        logger.info(f"Using existing extracted data: {extracted_file}")
    
    # Step 2: Integrate with existing PHMSA data (if provided)
    logger.info("Integrating PHMSA data...")
    
    final_df = integrate_phmsa_data(
        pdf_extracted_file=str(extracted_file),
        existing_phmsa_file=args.existing_data,
        output_dir=args.output_dir
    )
    
    # Step 3: Create summary report
    logger.info("Creating summary report...")
    
    summary_stats = {
        'Total Records': len(final_df),
        'Records with Narratives': final_df['has_narrative'].sum() if 'has_narrative' in final_df else 0,
        'Records with PDF Data': final_df['has_pdf_extraction'].sum() if 'has_pdf_extraction' in final_df else 0,
        'Records with Consequences': final_df['has_consequences_data'].sum() if 'has_consequences_data' in final_df else 0,
    }
    
    # Add consequence statistics if available
    if 'incident_results_spillage' in final_df.columns:
        consequence_stats = {
            'Incidents with Spillage': final_df['incident_results_spillage'].sum(),
            'Incidents with Fire': final_df.get('incident_results_fire', pd.Series()).sum(),
            'Incidents with Explosion': final_df.get('incident_results_explosion', pd.Series()).sum(),
            'Incidents with Injuries': final_df.get('injuries_caused_by_hazmat', pd.Series()).sum(),
            'Incidents with Fatalities': final_df.get('fatalities_caused_by_hazmat', pd.Series()).sum(),
            'Incidents with Evacuation': final_df.get('evacuation_occurred', pd.Series()).sum(),
        }
        summary_stats.update(consequence_stats)
    
    # Save summary report
    summary_df = pd.DataFrame([summary_stats]).T
    summary_df.columns = ['Count']
    summary_df.to_csv(output_dir / 'processing_summary.csv')
    
    # Print summary
    print("\n" + "="*50)
    print("PHMSA PDF PROCESSING COMPLETE")
    print("="*50)
    for key, value in summary_stats.items():
        print(f"{key}: {value:,}")
    print("="*50)
    print(f"\nOutput files saved to: {output_dir}")
    print("- phmsa_full.parquet: Complete dataset")
    print("- phmsa_consequences_summary.csv: Summary of incident consequences")
    print("- phmsa_text_analysis.parquet: Data for text analysis")
    print("- processing_summary.csv: Processing statistics")
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()