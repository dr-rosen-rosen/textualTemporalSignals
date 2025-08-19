"""
Data readers for different event reporting systems.
Each reader handles the specific format and idiosyncracies of its data source.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ASRSReader:
    """Reader for Aviation Safety Reporting System (ASRS) data."""
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        
    def read_multiple_files(self, file_pattern: str = "*.xlsx") -> pd.DataFrame:
        """
        Read and combine multiple ASRS Excel files.
        ASRS data often comes in multiple downloads that need to be combined.
        
        Args:
            file_pattern: Glob pattern for finding files
            
        Returns:
            Combined dataframe with all ASRS events
        """
        files = list(self.data_dir.glob(file_pattern))
        if not files:
            raise FileNotFoundError(f"No files matching {file_pattern} found in {self.data_dir}")
        
        logger.info(f"Found {len(files)} ASRS files to process")
        
        dfs = []
        for file_path in files:
            logger.debug(f"Reading {file_path}")
            
            # ASRS files have a complex header structure
            # Read the first two rows to construct proper column names
            header_row1 = pd.read_excel(file_path, nrows=0).columns
            header_row2 = pd.read_excel(file_path, skiprows=1, nrows=0).columns
            
            # Combine headers (handling the multi-level structure)
            if len(header_row1) == len(header_row2):
                headers = [f"{h1}.{h2}" for h1, h2 in zip(header_row1, header_row2)]
            else:
                headers = header_row1
            
            # Read the actual data
            df = pd.read_excel(
                file_path, 
                skiprows=3, 
                names=headers,
                usecols="A:DU"  # Based on the R code range
            )
            
            dfs.append(df)
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates based on ACN (report number)
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['ACN'])
        logger.info(f"Removed {initial_count - len(combined_df)} duplicate records")
        
        return combined_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean ASRS data according to the original R logic.
        
        Args:
            df: Raw ASRS dataframe
            
        Returns:
            Cleaned dataframe
        """
        # Remove rows where ACN is null or equals 'ACN' (header row)
        df = df[df['ACN'].notna() & (df['ACN'] != 'ACN')].copy()
        
        # Parse date from YYYYMM format
        df['date'] = pd.to_datetime(
            df['Time.Date'].astype(str).str[:6] + '01',
            format='%Y%m%d',
            errors='coerce'
        )
        
        # Combine narrative columns
        narrative_cols = [col for col in df.columns if col.startswith('Report.')]
        df['cmbd_narrative'] = df[narrative_cols].fillna('').agg(' '.join, axis=1).str.strip()
        
        # Calculate word count
        df['tot_wc'] = df['cmbd_narrative'].str.split().str.len()
        
        # Clean column names (lowercase and replace dots with underscores)
        df.columns = df.columns.str.lower().str.replace('.', '_', regex=False)
        
        return df


class NRCReader:
    """Reader for Nuclear Regulatory Commission (NRC) event data."""
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
    
    def read_data(self, filename: str = "nrc_events.csv") -> pd.DataFrame:
        """Read NRC data from CSV file."""
        file_path = self.data_dir / filename
        return pd.read_csv(file_path)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean NRC data according to the original R logic.
        
        Args:
            df: Raw NRC dataframe
            
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        # Fix known date errors
        df['event_date'] = df['event_date'].replace('11/20/2103', '11/20/2013')
        
        # Parse dates
        df['event_date2'] = pd.to_datetime(df['event_date'], format='%m/%d/%Y', errors='coerce')
        df['notification_date2'] = pd.to_datetime(df['notification_date'], format='%m/%d/%Y', errors='coerce')
        
        # Clean event text (lowercase)
        df['event_text'] = df['event_text'].str.lower()
        
        # Standardize facility names
        facility_mapping = {
            'columbia generating statiregion:': 'columbia generating station',
            'davis-besse': 'davis besse',
            'washington nuclear (wnp-2region:': 'washington nuclear',
            'vogtle 1/2': 'vogtle',
            'vogtle 3/4': 'vogtle',
            'fort calhoun': 'ft calhoun',
            'summer construction': 'summer'
        }
        
        df['facility'] = df['facility'].str.lower().replace(facility_mapping)
        
        # Remove any columns starting with 'Unnamed' (like R's X column)
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        
        return df


class RailReader:
    """Reader for Rail safety data."""
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
    
    def read_data(self, filename: str = "rail_events.csv") -> pd.DataFrame:
        """Read Rail data from CSV file."""
        file_path = self.data_dir / filename
        return pd.read_csv(file_path)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Rail data according to the original R logic.
        
        Args:
            df: Raw Rail dataframe
            
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        # Clean narrative field
        df['Narrative'] = (df['Narrative']
                          .str.replace('NoneNone', '', regex=False)
                          .str.replace('None$', '', regex=True))
        
        # Filter out invalid narratives
        df = df[
            df['Narrative'].notna() & 
            (df['Narrative'] != '') & 
            (df['Narrative'] != 'None')
        ]
        
        # Ensure valid UTF-8 (similar to R's utf8::utf8_valid)
        df = df[df['Narrative'].apply(lambda x: self._is_valid_utf8(str(x)))]
        
        return df
    
    @staticmethod
    def _is_valid_utf8(text: str) -> bool:
        """Check if text is valid UTF-8."""
        try:
            text.encode('utf-8')
            return True
        except UnicodeEncodeError:
            return False


class PHMSAReader:
    """Reader for Pipeline and Hazardous Materials Safety Administration (PHMSA) data."""
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
    
    def read_data(self, filename: str = "phmsa_events.csv") -> pd.DataFrame:
        """Read PHMSA data from CSV file."""
        file_path = self.data_dir / filename
        return pd.read_csv(file_path)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean PHMSA data according to the original R logic.
        
        Args:
            df: Raw PHMSA dataframe
            
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        # Filter out invalid narratives
        df = df[
            df['cmbd_narrative'].notna() & 
            (df['cmbd_narrative'] != '')
        ]
        
        # Ensure valid UTF-8
        df = df[df['cmbd_narrative'].apply(lambda x: self._is_valid_utf8(str(x)))]
        
        # Ensure cmbd_narrative is string type
        df['cmbd_narrative'] = df['cmbd_narrative'].astype(str)
        
        return df
    
    @staticmethod
    def _is_valid_utf8(text: str) -> bool:
        """Check if text is valid UTF-8."""
        try:
            text.encode('utf-8')
            return True
        except UnicodeEncodeError:
            return False


def create_reader(source: str, data_dir: Union[str, Path]) -> Union[ASRSReader, NRCReader, RailReader, PHMSAReader]:
    """
    Factory function to create appropriate reader based on source.
    
    Args:
        source: Data source name ('asrs', 'nrc', 'rail', 'phmsa')
        data_dir: Directory containing source data
        
    Returns:
        Appropriate reader instance
    """
    readers = {
        'asrs': ASRSReader,
        'nrc': NRCReader,
        'rail': RailReader,
        'phmsa': PHMSAReader
    }
    
    if source not in readers:
        raise ValueError(f"Unknown source: {source}. Valid sources: {list(readers.keys())}")
    
    return readers[source](data_dir)