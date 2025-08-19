"""
PHMSA PDF Report Scraper

Extracts narratives and structured data from PHMSA incident report PDFs.
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import json
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import pickle

logger = logging.getLogger(__name__)


class PHMSAPDFScraper:
    """Extract data from PHMSA incident report PDFs."""
    
    def __init__(self, 
                 output_dir: Union[str, Path],
                 checkpoint_file: str = "phmsa_scraping_checkpoint.pkl"):
        """
        Initialize the PDF scraper.
        
        Args:
            output_dir: Directory to save extracted data
            checkpoint_file: File to save progress for resuming
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / checkpoint_file
        self.failed_reports = []
        
        # Load checkpoint if exists
        self.checkpoint = self._load_checkpoint()
        
    def _load_checkpoint(self) -> Dict:
        """Load checkpoint data if it exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return {'processed': [], 'data': {}}
    
    def _save_checkpoint(self):
        """Save current progress."""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(self.checkpoint, f)
    
    def extract_from_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, any]:
        """
        Extract all relevant data from a single PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted data
        """
        try:
            # Extract text with layout preservation for better structure parsing
            text = extract_text(pdf_path, laparams=LAParams())
            
            # Extract different sections
            data = {
                'narrative1': self._extract_narrative1(text),
                'narrative2': self._extract_narrative2(text),
                **self._extract_consequences(text),
                'extraction_date': datetime.now().isoformat(),
                'pdf_path': str(pdf_path)
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting from {pdf_path}: {str(e)}")
            return None

    def _extract_narrative1(self, text: str) -> str:
        """Extract the first narrative section (Part VI)."""
        # More flexible pattern that handles variations in text
        patterns = [
            # Original pattern with dash variations
            re.compile(
                r"PART VI[^A-Z]*DESCRIPTION OF EVENTS.*?Describe:\s*(.*?)PART VII",
                re.DOTALL | re.IGNORECASE
            ),
            # Alternative pattern looking for the section more broadly
            re.compile(
                r"Describe:\s*(.*?)PART VII",
                re.DOTALL | re.IGNORECASE
            ),
            # Even simpler pattern
            re.compile(
                r"DESCRIPTION OF EVENTS.*?Describe:\s*(.*?)(?:PART VII|RECOMMENDATIONS)",
                re.DOTALL | re.IGNORECASE
            )
        ]
        
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                narrative = matches[0]
                # Clean up the narrative
                narrative = re.sub(r'PART VII.*', '', narrative, flags=re.IGNORECASE)
                narrative = re.sub(r'RECOMMENDATIONS.*', '', narrative, flags=re.IGNORECASE)
                return narrative.strip()
        
        # If no pattern matched, try to find any "Describe:" section
        simple_pattern = re.compile(r"Describe:\s*([^P]+)", re.DOTALL | re.IGNORECASE)
        matches = simple_pattern.findall(text)
        if matches:
            return matches[0].strip()
        
        return ""

    def _extract_narrative2(self, text: str) -> str:
        """Extract the second narrative section (Part VII)."""
        patterns = [
            # Look for PART VII with flexible formatting
            re.compile(
                r"PART VII[^A-Z]*(?:RECOMMENDATIONS|ACTIONS TAKEN).*?Describe:\s*(.*?)(?:PART VIII|CONTACT INFORMATION)",
                re.DOTALL | re.IGNORECASE
            ),
            # Simpler pattern
            re.compile(
                r"PREVENT RECURRENCE.*?Describe:\s*(.*?)(?:PART VIII|CONTACT)",
                re.DOTALL | re.IGNORECASE
            )
        ]
        
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                narrative = matches[0]
                # Clean up
                narrative = re.sub(r'PART VIII.*', '', narrative, flags=re.IGNORECASE)
                narrative = re.sub(r'CONTACT INFORMATION.*', '', narrative, flags=re.IGNORECASE)
                return narrative.strip()
        
        return ""

    def _extract_consequences(self, text: str) -> Dict[str, any]:
        """
        Extract structured data from PART IV - CONSEQUENCES section.
        More robust extraction that handles text variations.
        """
        consequences = {}
        
        # More flexible pattern for finding PART IV section
        part_iv_patterns = [
            re.compile(r"PART IV[^A-Z]*CONSEQUENCES(.*?)PART V", re.DOTALL | re.IGNORECASE),
            re.compile(r"CONSEQUENCES(.*?)(?:PART V|AIR INCIDENT)", re.DOTALL | re.IGNORECASE)
        ]
        
        part_iv_text = None
        for pattern in part_iv_patterns:
            match = pattern.search(text)
            if match:
                part_iv_text = match.group(1)
                break
        
        if not part_iv_text:
            logger.warning("Could not find PART IV - CONSEQUENCES section")
            return consequences
        
        # Extract incident results with more flexible patterns
        incident_results = {}
        
        # Look for checkbox-style entries (True/False)
        spillage_match = re.search(r"Spillage:?\s*(True|False)", part_iv_text, re.IGNORECASE)
        if spillage_match:
            incident_results['spillage'] = spillage_match.group(1).lower() == 'true'
        
        fire_match = re.search(r"Fire:?\s*(True|False)", part_iv_text, re.IGNORECASE)
        if fire_match:
            incident_results['fire'] = fire_match.group(1).lower() == 'true'
        
        explosion_match = re.search(r"Explosion:?\s*(True|False)", part_iv_text, re.IGNORECASE)
        if explosion_match:
            incident_results['explosion'] = explosion_match.group(1).lower() == 'true'
        
        # More flexible patterns for other fields
        waterway_patterns = [
            r"Material Entered Waterway[^:]*:?\s*(True|False)",
            r"Waterway/Storm Sewer:?\s*(True|False)"
        ]
        for pattern in waterway_patterns:
            match = re.search(pattern, part_iv_text, re.IGNORECASE)
            if match:
                incident_results['material_entered_waterway'] = match.group(1).lower() == 'true'
                break
        
        vapor_match = re.search(r"Vapor[^:]*:?\s*(True|False)", part_iv_text, re.IGNORECASE)
        if vapor_match:
            incident_results['vapor_dispersion'] = vapor_match.group(1).lower() == 'true'
        
        env_match = re.search(r"Environmental Damage:?\s*(True|False)", part_iv_text, re.IGNORECASE)
        if env_match:
            incident_results['environmental_damage'] = env_match.group(1).lower() == 'true'
        
        no_release_match = re.search(r"No Release:?\s*(True|False)", part_iv_text, re.IGNORECASE)
        if no_release_match:
            incident_results['no_release'] = no_release_match.group(1).lower() == 'true'
        
        consequences['incident_results'] = incident_results
        
        # Extract emergency response
        emergency_response = {}
        
        # Look for emergency response patterns
        if re.search(r"Fire/EMS Report", part_iv_text, re.IGNORECASE):
            # Check if it says False or has a report number
            fire_ems_match = re.search(r"Fire/EMS Report[^:]*:?\s*(False|True|\d+)", part_iv_text, re.IGNORECASE)
            if fire_ems_match:
                val = fire_ems_match.group(1).lower()
                emergency_response['fire_ems'] = val == 'true' or val.isdigit()
            else:
                emergency_response['fire_ems'] = False
        
        police_match = re.search(r"Police Report[^:]*:?\s*(False|True|\d+)", part_iv_text, re.IGNORECASE)
        if police_match:
            val = police_match.group(1).lower()
            emergency_response['police'] = val == 'true' or val.isdigit()
        
        inhouse_match = re.search(r"In-house cleanup:?\s*(True|False)", part_iv_text, re.IGNORECASE)
        if inhouse_match:
            emergency_response['in_house_cleanup'] = inhouse_match.group(1).lower() == 'true'
        
        other_cleanup_match = re.search(r"Other Cleanup:?\s*(True|False)", part_iv_text, re.IGNORECASE)
        if other_cleanup_match:
            emergency_response['other_cleanup'] = other_cleanup_match.group(1).lower() == 'true'
        
        consequences['emergency_response'] = emergency_response
        
        # Extract damage information
        damage_patterns = [
            r"Was the total damage cost more than \$500\??\s*(True|False)",
            r"total damage cost[^$]*\$500\??\s*(True|False)"
        ]
        for pattern in damage_patterns:
            match = re.search(pattern, part_iv_text, re.IGNORECASE)
            if match:
                consequences['damages_over_500'] = match.group(1).lower() == 'true'
                break
        
        # Extract damage amounts if damages_over_500 is True
        if consequences.get('damages_over_500'):
            damages = {}
            
            # Dollar amount patterns
            damage_fields = [
                ('material_loss', r"Material Loss:?\s*\$\s*([\d,]+\.?\d*)"),
                ('carrier_damage', r"Carrier Damage:?\s*\$\s*([\d,]+\.?\d*)"),
                ('property_damage', r"Property Damage:?\s*\$\s*([\d,]+\.?\d*)"),
                ('response_cost', r"Response Cost:?\s*\$\s*([\d,]+\.?\d*)"),
                ('remediation_cleanup_cost', r"Remediation/Cleanup Cost:?\s*\$\s*([\d,]+\.?\d*)")
            ]
            
            for field_name, pattern in damage_fields:
                match = re.search(pattern, part_iv_text, re.IGNORECASE)
                if match:
                    try:
                        damages[field_name] = float(match.group(1).replace(',', ''))
                    except ValueError:
                        damages[field_name] = 0.0
            
            consequences['damages'] = damages
        
        # Extract fatality information
        fatality_patterns = [
            r"Did the hazardous material cause[^?]*fatality\??\s*(True|False)",
            r"contribute to a human fatality\??\s*(True|False)"
        ]
        for pattern in fatality_patterns:
            match = re.search(pattern, part_iv_text, re.IGNORECASE)
            if match:
                consequences['fatalities_caused_by_hazmat'] = match.group(1).lower() == 'true'
                break
        
        # Extract injury information
        injury_patterns = [
            r"Did the hazardous material cause[^?]*Injury\??\s*(True|False)",
            r"contribute to personal Injury\??\s*(True|False)"
        ]
        for pattern in injury_patterns:
            match = re.search(pattern, part_iv_text, re.IGNORECASE)
            if match:
                consequences['injuries_caused_by_hazmat'] = match.group(1).lower() == 'true'
                break
        
        # Extract evacuation information
        evac_patterns = [
            r"Did the hazardous material cause[^?]*evacuation\??\s*(True|False)",
            r"contribute to an evacuation\??\s*(True|False)"
        ]
        for pattern in evac_patterns:
            match = re.search(pattern, part_iv_text, re.IGNORECASE)
            if match:
                consequences['evacuation_occurred'] = match.group(1).lower() == 'true'
                break
        
        # Transportation closure
        transport_patterns = [
            r"Was a major transportation[^?]*closed\??\s*(True|False)",
            r"major transportation artery[^?]*closed\??\s*(True|False)"
        ]
        for pattern in transport_patterns:
            match = re.search(pattern, part_iv_text, re.IGNORECASE)
            if match:
                consequences['major_transportation_closed'] = match.group(1).lower() == 'true'
                break
        
        # Crash/derailment
        crash_patterns = [
            r"Was the material involved in a crash[^?]*\??\s*(True|False)",
            r"crash or derailment\??\s*(True|False)"
        ]
        for pattern in crash_patterns:
            match = re.search(pattern, part_iv_text, re.IGNORECASE)
            if match:
                consequences['crash_or_derailment'] = match.group(1).lower() == 'true'
                break
        
        return consequences

    def _extract_boolean(self, text: str, pattern: str) -> Optional[bool]:
        """Extract boolean value from text with more flexibility."""
        # Try the provided pattern first
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            return value == 'true' or value == 'yes'
        
        # Try without colons
        pattern_no_colon = pattern.replace(':', '')
        match = re.search(pattern_no_colon, text, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            return value == 'true' or value == 'yes'
        
        return None
    
    def _extract_number(self, text: str, pattern: str, section: str = "") -> Optional[int]:
        """Extract numeric value from text."""
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None
    
    def _extract_dollar_amount(self, text: str, pattern: str) -> Optional[float]:
        """Extract dollar amount from text."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                # Remove commas and convert to float
                amount_str = match.group(1).replace(',', '')
                return float(amount_str)
            except ValueError:
                return None
        return None
    
    def _extract_text(self, text: str, pattern: str) -> Optional[str]:
        """Extract text value from pattern match."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def process_pdf_list(self, 
                        pdf_files: List[Union[str, Path]], 
                        report_ids: Optional[List[str]] = None,
                        save_interval: int = 100) -> pd.DataFrame:
        """
        Process a list of PDF files.
        
        Args:
            pdf_files: List of paths to PDF files
            report_ids: Optional list of report IDs corresponding to PDFs
            save_interval: Save checkpoint every N files
            
        Returns:
            DataFrame with all extracted data
        """
        results = []
        
        for i, pdf_path in enumerate(pdf_files):
            pdf_path = Path(pdf_path)
            
            # Get report ID
            if report_ids and i < len(report_ids):
                report_id = report_ids[i]
            else:
                # Try to extract from filename
                report_id = pdf_path.stem
            
            # Skip if already processed
            if report_id in self.checkpoint['processed']:
                logger.info(f"Skipping already processed: {report_id}")
                if report_id in self.checkpoint['data']:
                    results.append(self.checkpoint['data'][report_id])
                continue
            
            logger.info(f"Processing {report_id} ({i+1}/{len(pdf_files)})")
            
            # Extract data
            data = self.extract_from_pdf(pdf_path)
            
            if data:
                data['report_no'] = report_id
                results.append(data)
                
                # Update checkpoint
                self.checkpoint['processed'].append(report_id)
                self.checkpoint['data'][report_id] = data
            else:
                self.failed_reports.append((report_id, str(pdf_path)))
                logger.error(f"Failed to process: {report_id}")
            
            # Save checkpoint periodically
            if (i + 1) % save_interval == 0:
                self._save_checkpoint()
                logger.info(f"Checkpoint saved at {i+1} files")
        
        # Final save
        self._save_checkpoint()
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save failed reports list
        if self.failed_reports:
            failed_df = pd.DataFrame(self.failed_reports, columns=['report_no', 'pdf_path'])
            failed_df.to_csv(self.output_dir / 'failed_reports.csv', index=False)
            logger.warning(f"Failed to process {len(self.failed_reports)} reports. See failed_reports.csv")
        
        return df
    
    def export_results(self, df: pd.DataFrame, format: str = 'parquet'):
        """
        Export results to file.
        
        Args:
            df: DataFrame with extracted data
            format: Output format ('parquet', 'csv', 'json')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'parquet':
            output_file = self.output_dir / f'phmsa_extracted_{timestamp}.parquet'
            df.to_parquet(output_file, index=False)
        elif format == 'csv':
            output_file = self.output_dir / f'phmsa_extracted_{timestamp}.csv'
            df.to_csv(output_file, index=False)
        elif format == 'json':
            output_file = self.output_dir / f'phmsa_extracted_{timestamp}.json'
            df.to_json(output_file, orient='records', indent=2)
        
        logger.info(f"Results exported to {output_file}")
        
        # Also save a flattened version for easier analysis
        flattened_df = self._flatten_nested_data(df)
        flattened_file = self.output_dir / f'phmsa_extracted_flat_{timestamp}.{format}'
        
        if format == 'parquet':
            flattened_df.to_parquet(flattened_file, index=False)
        elif format == 'csv':
            flattened_df.to_csv(flattened_file, index=False)
            
        logger.info(f"Flattened results exported to {flattened_file}")
    
    def _flatten_nested_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested dictionary columns for easier analysis."""
        flat_df = df.copy()
        
        # Columns that contain nested data
        nested_cols = ['incident_results', 'emergency_response', 'damages', 
                      'fatalities', 'injuries', 'evacuation_info', 'crash_info']
        
        for col in nested_cols:
            if col in flat_df.columns:
                # Extract nested data
                if flat_df[col].notna().any():
                    nested_data = pd.json_normalize(flat_df[col].dropna())
                    # Prefix column names
                    nested_data.columns = [f"{col}_{c}" for c in nested_data.columns]
                    # Merge back
                    flat_df = pd.concat([flat_df.drop(col, axis=1), nested_data], axis=1)
        
        return flat_df


def main():
    """Example usage of the PHMSA PDF scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract data from PHMSA PDF reports")
    parser.add_argument('pdf_dir', help='Directory containing PDF files')
    parser.add_argument('--output-dir', default='data/processed/phmsa', 
                       help='Output directory for extracted data')
    parser.add_argument('--report-ids', help='CSV file with report IDs')
    parser.add_argument('--format', choices=['parquet', 'csv', 'json'], 
                       default='parquet', help='Output format')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save checkpoint every N files')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get PDF files
    pdf_dir = Path(args.pdf_dir)
    pdf_files = list(pdf_dir.glob('*.pdf'))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Load report IDs if provided
    report_ids = None
    if args.report_ids:
        report_ids_df = pd.read_csv(args.report_ids)
        report_ids = report_ids_df['report_no'].tolist()
    
    # Initialize scraper
    scraper = PHMSAPDFScraper(args.output_dir)
    
    # Process PDFs
    results_df = scraper.process_pdf_list(
        pdf_files, 
        report_ids=report_ids,
        save_interval=args.save_interval
    )
    
    # Export results
    scraper.export_results(results_df, format=args.format)
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()