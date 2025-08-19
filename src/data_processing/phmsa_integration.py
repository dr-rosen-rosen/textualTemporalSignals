"""
Integration module to combine PHMSA PDF extracted data with the main pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PHMSADataIntegrator:
    """Integrate PHMSA PDF extracted data into the main pipeline."""
    
    def __init__(self):
        self.required_columns = ['report_no', 'narrative1', 'narrative2']
        
    def prepare_phmsa_data(self, 
                          pdf_extracted_file: Union[str, Path],
                          existing_phmsa_file: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Prepare PHMSA data by combining PDF extracted data with existing event data.
        
        Args:
            pdf_extracted_file: Path to the flattened PDF extraction results
            existing_phmsa_file: Optional path to existing PHMSA event data (from web scraping)
            
        Returns:
            Combined PHMSA dataframe ready for the main pipeline
        """
        # Load PDF extracted data
        pdf_data = pd.read_parquet(pdf_extracted_file)
        logger.info(f"Loaded {len(pdf_data)} records from PDF extraction")
        
        # Combine narratives into single field
        pdf_data['cmbd_narrative'] = pdf_data.apply(
            lambda row: self._combine_narratives(row['narrative1'], row['narrative2']), 
            axis=1
        )
        
        # If we have existing PHMSA data, merge with it
        if existing_phmsa_file:
            existing_data = pd.read_csv(existing_phmsa_file)
            logger.info(f"Loaded {len(existing_data)} records from existing PHMSA data")
            
            # Merge on report_no
            combined = existing_data.merge(
                pdf_data,
                on='report_no',
                how='left',
                suffixes=('', '_pdf')
            )
            
            # Update narrative if extracted from PDF
            if 'cmbd_narrative_pdf' in combined.columns:
                mask = combined['cmbd_narrative_pdf'].notna()
                combined.loc[mask, 'cmbd_narrative'] = combined.loc[mask, 'cmbd_narrative_pdf']
                combined = combined.drop('cmbd_narrative_pdf', axis=1)
            
            # Add consequence data columns
            consequence_cols = [col for col in pdf_data.columns if col.startswith(
                ('incident_results_', 'emergency_response_', 'damages_', 
                 'fatalities_', 'injuries_', 'evacuation_', 'crash_')
            )]
            
            for col in consequence_cols:
                if col in pdf_data.columns and col not in combined.columns:
                    # Add the column from pdf_data
                    col_data = pdf_data.set_index('report_no')[col]
                    combined[col] = combined['report_no'].map(col_data)
            
            df = combined
            
        else:
            # Use PDF data as the base
            df = pdf_data.copy()
            
            # Add date column if not present (will need to be filled from another source)
            if 'date' not in df.columns:
                logger.warning("Date column not found in PDF data. Will need to be added from another source.")
                df['date'] = pd.NaT
        
        # Calculate narrative statistics
        df['narrative_word_count'] = df['cmbd_narrative'].str.split().str.len()
        df['has_narrative'] = df['narrative_word_count'] > 0
        
        # Add extraction metadata
        df['has_pdf_extraction'] = df['report_no'].isin(pdf_data['report_no'])
        df['has_consequences_data'] = df[[col for col in df.columns if col.startswith('incident_results_')]].notna().any(axis=1)
        
        # Log summary statistics
        logger.info(f"Final PHMSA dataset: {len(df)} records")
        logger.info(f"Records with narratives: {df['has_narrative'].sum()}")
        logger.info(f"Records with PDF extraction: {df['has_pdf_extraction'].sum()}")
        logger.info(f"Records with consequences data: {df['has_consequences_data'].sum()}")
        
        return df
    
    def _combine_narratives(self, narrative1: str, narrative2: str) -> str:
        """Combine two narrative fields into one."""
        narratives = []
        
        if pd.notna(narrative1) and narrative1.strip():
            narratives.append(narrative1.strip())
            
        if pd.notna(narrative2) and narrative2.strip():
            narratives.append(narrative2.strip())
            
        return ' '.join(narratives)
    
    def create_consequences_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a summary of consequences data for analysis.
        
        Args:
            df: PHMSA dataframe with consequences data
            
        Returns:
            Summary dataframe with key metrics
        """
        summary_data = []
        
        for idx, row in df.iterrows():
            summary = {
                'report_no': row['report_no'],
                'date': row.get('date', pd.NaT),
                'had_release': row.get('incident_results_no_release') == False if pd.notna(row.get('incident_results_no_release')) else None,
                'had_spillage': row.get('incident_results_spillage', None),
                'had_fire': row.get('incident_results_fire', None),
                'had_explosion': row.get('incident_results_explosion', None),
                'had_environmental_damage': row.get('incident_results_environmental_damage', None),
                'total_damage_cost': None,
                'had_fatalities': row.get('fatalities_caused_by_hazmat', False),
                'had_injuries': row.get('injuries_caused_by_hazmat', False),
                'had_evacuation': row.get('evacuation_occurred', False),
                'had_transportation_closure': row.get('major_transportation_closed', False),
                'severity_score': 0  # Will calculate below
            }
            
            # Calculate total damage cost if available
            damage_cols = ['damages_material_loss', 'damages_carrier_damage', 
                          'damages_property_damage', 'damages_response_cost', 
                          'damages_remediation_cleanup_cost']
            
            damage_values = []
            for col in damage_cols:
                if col in row and pd.notna(row[col]):
                    damage_values.append(row[col])
            
            if damage_values:
                summary['total_damage_cost'] = sum(damage_values)
            
            # Calculate severity score (simple scoring system)
            if summary['had_fire']:
                summary['severity_score'] += 3
            if summary['had_explosion']:
                summary['severity_score'] += 4
            if summary['had_fatalities']:
                summary['severity_score'] += 5
            if summary['had_injuries']:
                summary['severity_score'] += 2
            if summary['had_evacuation']:
                summary['severity_score'] += 2
            if summary['had_environmental_damage']:
                summary['severity_score'] += 2
            if summary['total_damage_cost'] and summary['total_damage_cost'] > 100000:
                summary['severity_score'] += 3
            elif summary['total_damage_cost'] and summary['total_damage_cost'] > 10000:
                summary['severity_score'] += 1
                
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Add severity categories
        summary_df['severity_category'] = pd.cut(
            summary_df['severity_score'],
            bins=[-1, 0, 2, 5, 10, 100],
            labels=['None', 'Low', 'Medium', 'High', 'Very High']
        )
        
        return summary_df
    
    def export_for_analysis(self, df: pd.DataFrame, output_dir: Union[str, Path]):
        """
        Export PHMSA data in formats suitable for analysis.
        
        Args:
            df: Processed PHMSA dataframe
            output_dir: Directory to save output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full dataset
        df.to_parquet(output_dir / 'phmsa_full.parquet', index=False)
        
        # Save consequences summary
        summary_df = self.create_consequences_summary(df)
        summary_df.to_csv(output_dir / 'phmsa_consequences_summary.csv', index=False)
        
        # Save subset with just narratives and basic info for text analysis
        text_cols = ['report_no', 'date', 'cmbd_narrative', 'narrative_word_count']
        text_df = df[text_cols + [col for col in df.columns if col.startswith('incident_results_')]]
        text_df.to_parquet(output_dir / 'phmsa_text_analysis.parquet', index=False)
        
        logger.info(f"Exported PHMSA analysis files to {output_dir}")


# Example usage function
def integrate_phmsa_data(pdf_extracted_file: str, 
                        existing_phmsa_file: Optional[str] = None,
                        output_dir: str = 'data/processed/phmsa'):
    """
    Main function to integrate PHMSA PDF data.
    
    Args:
        pdf_extracted_file: Path to PDF extraction results
        existing_phmsa_file: Optional path to existing PHMSA data
        output_dir: Output directory for processed files
    """
    integrator = PHMSADataIntegrator()
    
    # Prepare the data
    df = integrator.prepare_phmsa_data(pdf_extracted_file, existing_phmsa_file)
    
    # Export for analysis
    integrator.export_for_analysis(df, output_dir)
    
    return df