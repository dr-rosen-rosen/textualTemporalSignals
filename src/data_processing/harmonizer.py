"""
Data harmonization module to standardize column names and data types across sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataHarmonizer:
    """Harmonize data from different sources to a common schema."""
    
    # Define column mappings for each source
    COLUMN_MAPPINGS = {
        'asrs': {
            'event_date': 'date',
            'event_num': 'acn',
            'event_text': 'cmbd_narrative'
        },
        'rail': {
            'event_date': 'Date',
            'event_num': 'Accident.Number',
            'event_text': 'Narrative'
        },
        'nrc': {
            'event_date': 'event_date',
            'event_num': 'event_num',
            'event_text': 'event_text'
        },
        'phmsa': {
            'event_date': 'date',
            'event_num': 'report_no',
            'event_text': 'cmbd_narrative'
        }
    }
    
    def __init__(self):
        self.key_vars = ['event_date', 'event_num', 'event_text', 'source_system']
    
    def harmonize(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Harmonize dataframe columns to standard schema.
        
        Args:
            df: Source-specific dataframe
            source: Source system name
            
        Returns:
            Harmonized dataframe with standardized columns
        """
        if source not in self.COLUMN_MAPPINGS:
            raise ValueError(f"Unknown source: {source}")
        
        df = df.copy()
        
        # Rename columns according to mapping
        mapping = self.COLUMN_MAPPINGS[source]
        df = df.rename(columns={v: k for k, v in mapping.items()})
        
        # Add source system column
        df['source_system'] = source
        
        # Ensure event_num is string type
        df['event_num'] = df['event_num'].astype(str)
        
        # Add unique identifier combining source and row number
        df['dataSet_num'] = source + '_' + (df.index + 1).astype(str)
        
        # Handle multiple events with same event_num (as in original R code)
        if source in ['asrs', 'rail']:
            df = self._handle_duplicate_event_nums(df)
        
        # Standardize date format
        df = self._standardize_dates(df, source)
        
        # Reorder columns with key variables first
        cols = self.key_vars + [col for col in df.columns if col not in self.key_vars]
        df = df[[col for col in cols if col in df.columns]]
        
        logger.info(f"Harmonized {len(df)} records from {source}")
        
        return df
    
    def _handle_duplicate_event_nums(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle cases where multiple records have the same event_num.
        Creates new event_num with suffix for duplicates.
        """
        df['event_num_orig'] = df['event_num']
        
        # Group by event_num and add row number within group
        df['event_num'] = (df.groupby('event_num_orig').cumcount() + 1).astype(str)
        df['event_num'] = df['event_num_orig'] + '_' + df['event_num']
        
        return df
    
    def _standardize_dates(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Standardize date formats across sources."""
        if source == 'rail':
            # Rail dates are in M/D/Y format
            df['event_date'] = pd.to_datetime(df['event_date'], format='%m/%d/%Y', errors='coerce')
        elif source == 'nrc':
            # NRC dates should already be parsed in cleaning
            if 'event_date2' in df.columns:
                df['event_date'] = df['event_date2']
                df = df.drop(columns=['event_date2'])
        elif source in ['asrs', 'phmsa']:
            # These should already have 'date' column properly formatted
            pass
        
        # Ensure date column is datetime type
        if 'event_date' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        
        return df
    
    def combine_sources(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine harmonized dataframes from multiple sources.
        
        Args:
            dataframes: Dictionary of source name to harmonized dataframe
            
        Returns:
            Combined dataframe with all sources
        """
        if not dataframes:
            raise ValueError("No dataframes provided to combine")
        
        # Ensure all dataframes have the same columns
        all_columns = set()
        for df in dataframes.values():
            all_columns.update(df.columns)
        
        # Add missing columns as NaN
        for source, df in dataframes.items():
            for col in all_columns:
                if col not in df.columns:
                    df[col] = np.nan
        
        # Combine all dataframes
        combined = pd.concat(dataframes.values(), ignore_index=True)
        
        # Sort by source and date
        combined = combined.sort_values(['source_system', 'event_date'])
        
        logger.info(f"Combined {len(combined)} total records from {len(dataframes)} sources")
        
        return combined
    
    def validate_harmonized_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate harmonized data for completeness and quality.
        
        Args:
            df: Harmonized dataframe
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_records': len(df),
            'sources': df['source_system'].value_counts().to_dict(),
            'missing_dates': df['event_date'].isna().sum(),
            'missing_event_nums': df['event_num'].isna().sum(),
            'missing_narratives': df['event_text'].isna().sum(),
            'duplicate_event_nums': df.duplicated(subset=['source_system', 'event_num']).sum(),
            'date_range': {
                'min': df['event_date'].min(),
                'max': df['event_date'].max()
            }
        }
        
        # Check for empty narratives
        if 'event_text' in df.columns:
            validation_results['empty_narratives'] = (df['event_text'] == '').sum()
            validation_results['avg_narrative_length'] = df['event_text'].str.len().mean()
        
        return validation_results