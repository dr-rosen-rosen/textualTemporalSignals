"""
Main pipeline script for loading, cleaning, and harmonizing event data.
"""

import sys
import logging
from pathlib import Path
import yaml
import pandas as pd
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_acquisition.readers import create_reader
from src.data_processing.harmonizer import DataHarmonizer


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log')
        ]
    )


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_source(source: str, config: dict) -> pd.DataFrame:
    """
    Process a single data source: load, clean, and harmonize.
    
    Args:
        source: Source name ('asrs', 'nrc', 'rail', 'phmsa')
        config: Configuration dictionary
        
    Returns:
        Harmonized dataframe
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {source} data...")
    
    # Get source-specific configuration
    source_config = config['data_paths'][source]
    data_dir = Path(source_config['data_dir'])
    
    # Create reader
    reader = create_reader(source, data_dir)
    
    # Load data
    if source == 'asrs':
        # ASRS requires combining multiple files
        df = reader.read_multiple_files()
    else:
        # Other sources read single file
        df = reader.read_data(source_config['events_file'])
    
    logger.info(f"Loaded {len(df)} raw records from {source}")
    
    # Clean data
    df_clean = reader.clean_data(df)
    logger.info(f"Cleaned data: {len(df_clean)} records remaining")
    
    # Harmonize data
    harmonizer = DataHarmonizer()
    df_harmonized = harmonizer.harmonize(df_clean, source)
    
    return df_harmonized


def main(sources: Optional[list] = None, config_path: str = "config.yaml"):
    """
    Main pipeline execution.
    
    Args:
        sources: List of sources to process. If None, process all.
        config_path: Path to configuration file
    """
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting event data processing pipeline")
    
    # Load configuration
    config = load_config(config_path)
    
    # Define sources to process
    if sources is None:
        sources = ['asrs', 'nrc', 'rail', 'phmsa']
    
    # Process each source
    harmonized_data = {}
    for source in sources:
        try:
            df = process_source(source, config)
            harmonized_data[source] = df
            
            # Save individual harmonized files
            output_path = Path(config['data_paths']['processed_data']) / f"{source}_harmonized.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved {source} harmonized data to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing {source}: {str(e)}")
            continue
    
    # Combine all sources
    if harmonized_data:
        harmonizer = DataHarmonizer()
        combined_df = harmonizer.combine_sources(harmonized_data)
        
        # Validate combined data
        validation_results = harmonizer.validate_harmonized_data(combined_df)
        logger.info(f"Validation results: {validation_results}")
        
        # Save combined data
        output_format = config['processing']['output_format']
        output_path = Path(config['data_paths']['processed_data']) / f"all_events_combined.{output_format}"
        
        if output_format == 'parquet':
            combined_df.to_parquet(output_path, index=False)
        elif output_format == 'feather':
            combined_df.to_feather(output_path)
        elif output_format == 'csv':
            combined_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved combined data to {output_path}")
        
        # Print summary statistics
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        for source, count in validation_results['sources'].items():
            print(f"{source}: {count} records")
        print(f"Total: {validation_results['total_records']} records")
        print(f"Date range: {validation_results['date_range']['min']} to {validation_results['date_range']['max']}")
        print("="*50)
    
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    # Allow command line arguments for specific sources
    import argparse
    
    parser = argparse.ArgumentParser(description="Process event data from multiple sources")
    parser.add_argument(
        '--sources', 
        nargs='+', 
        choices=['asrs', 'nrc', 'rail', 'phmsa'],
        help='Sources to process (default: all)'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    main(sources=args.sources, config_path=args.config)