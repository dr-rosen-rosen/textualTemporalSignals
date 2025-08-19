"""
Complete fixed version of PHMSA PDF Scraper with improved extraction and data handling
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
            
            # Debug: Save extracted text for inspection
            debug_file = self.output_dir / f"debug_{Path(pdf_path).stem}.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Extract different sections
            consequences_data = self._extract_consequences(text)
            
            data = {
                'narrative1': self._extract_narrative1(text),
                'narrative2': self._extract_narrative2(text),
                **consequences_data,  # Unpack all consequences data
                'extraction_date': datetime.now().isoformat(),
                'pdf_path': str(pdf_path)
            }
            
            # Add summary fields for easier analysis
            if 'incident_results' in consequences_data:
                results = consequences_data['incident_results']
                data['had_spillage'] = results.get('spillage', None)
                data['had_fire'] = results.get('fire', None)
                data['had_explosion'] = results.get('explosion', None)
                data['had_environmental_damage'] = results.get('environmental_damage', None)
                data['had_release'] = not results.get('no_release', True)
            
            # Add other summary fields
            data['had_fatalities'] = consequences_data.get('fatalities_caused_by_hazmat', None)
            data['had_injuries'] = consequences_data.get('injuries_caused_by_hazmat', None)
            data['had_evacuation'] = consequences_data.get('evacuation_occurred', None)
            data['had_transportation_closure'] = consequences_data.get('major_transportation_closed', None)
            
            # Calculate total damage if available
            if 'damages' in consequences_data and consequences_data['damages']:
                total_damage = sum(v for v in consequences_data['damages'].values() if v is not None)
                data['total_damage_cost'] = total_damage
            else:
                data['total_damage_cost'] = 0 if consequences_data.get('damages_over_500') == False else None
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting from {pdf_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _extract_narrative1(self, text: str) -> str:
        """Extract the first narrative section (Part VI)."""
        # Multiple patterns to try
        patterns = [
            # Look for Part VI section
            re.compile(
                r"PART VI[^\n]*\n[^\n]*DESCRIPTION OF EVENTS.*?Describe:\s*(.*?)(?:PART VII|$)",
                re.DOTALL | re.IGNORECASE
            ),
            # Simpler pattern
            re.compile(
                r"DESCRIPTION OF EVENTS.*?Describe:\s*(.*?)(?:PART VII|RECOMMENDATIONS|$)",
                re.DOTALL | re.IGNORECASE
            ),
            # Even simpler - just find Describe: in the latter part of the document
            re.compile(
                r"Describe:\s*\n([^P]*?)(?:PART VII|RECOMMENDATIONS|$)",
                re.DOTALL | re.IGNORECASE | re.MULTILINE
            )
        ]
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                narrative = match.group(1)
                # Clean up
                narrative = re.sub(r'PART VII.*', '', narrative, flags=re.IGNORECASE | re.DOTALL)
                narrative = re.sub(r'RECOMMENDATIONS.*', '', narrative, flags=re.IGNORECASE | re.DOTALL)
                return narrative.strip()
        
        return ""
    
    def _extract_narrative2(self, text: str) -> str:
        """Extract the second narrative section (Part VII)."""
        patterns = [
            # Look for PART VII
            re.compile(
                r"PART VII[^\n]*\n[^\n]*(?:RECOMMENDATIONS|ACTIONS).*?Describe:\s*(.*?)(?:PART VIII|CONTACT|$)",
                re.DOTALL | re.IGNORECASE
            ),
            # Alternative pattern
            re.compile(
                r"PREVENT RECURRENCE.*?Describe:\s*(.*?)(?:PART VIII|CONTACT|$)",
                re.DOTALL | re.IGNORECASE
            )
        ]
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                narrative = match.group(1)
                # Clean up
                narrative = re.sub(r'PART VIII.*', '', narrative, flags=re.IGNORECASE | re.DOTALL)
                narrative = re.sub(r'Contact.*', '', narrative, flags=re.IGNORECASE | re.DOTALL)
                return narrative.strip()
        
        return ""
    
    def _extract_consequences(self, text: str) -> Dict[str, any]:
        """
        Extract structured data from PART IV - CONSEQUENCES section.
        """
        consequences = {}
        
        # Find PART IV section - be very flexible
        part_iv_patterns = [
            re.compile(r"PART IV[^\n]*CONSEQUENCES(.*?)(?:PART V|$)", re.DOTALL | re.IGNORECASE),
            re.compile(r"30\.\s*Result of Incident(.*?)(?:PART V|38\.|$)", re.DOTALL | re.IGNORECASE),
            re.compile(r"CONSEQUENCES(.*?)(?:AIR INCIDENT|PART V|$)", re.DOTALL | re.IGNORECASE)
        ]
        
        part_iv_text = None
        for pattern in part_iv_patterns:
            match = pattern.search(text)
            if match:
                part_iv_text = match.group(0)  # Include the header for better context
                logger.info(f"Found PART IV section with pattern {pattern.pattern[:50]}...")
                break
        
        if not part_iv_text:
            logger.warning("Could not find PART IV - CONSEQUENCES section")
            # Try to extract from full text as fallback
            part_iv_text = text
        
        # Extract incident results (Question 30)
        incident_results = {}
        
        # These patterns look for the checkbox-style format in the PDFs
        result_patterns = [
            ('spillage', [r"-\s*Spillage:\s*(True|False)", r"Spillage:\s*(True|False)"]),
            ('fire', [r"-\s*Fire:\s*(True|False)", r"Fire:\s*(True|False)"]),
            ('explosion', [r"-\s*Explosion:\s*(True|False)", r"Explosion:\s*(True|False)"]),
            ('material_entered_waterway', [
                r"-\s*Material Entered Waterway[^:]*:\s*(True|False)",
                r"Material Entered Waterway/Storm Sewer:\s*(True|False)",
                r"Waterway[^:]*:\s*(True|False)"
            ]),
            ('vapor_dispersion', [
                r"-\s*Vapor[^:]*Dispersion:\s*(True|False)",
                r"Vapor \(Gas\) Dispersion:\s*(True|False)"
            ]),
            ('environmental_damage', [
                r"-\s*Environmental Damage:\s*(True|False)",
                r"Environmental Damage:\s*(True|False)"
            ]),
            ('no_release', [
                r"-\s*No Release:\s*(True|False)",
                r"No Release:\s*(True|False)"
            ])
        ]
        
        for field_name, patterns in result_patterns:
            for pattern in patterns:
                match = re.search(pattern, part_iv_text, re.IGNORECASE)
                if match:
                    incident_results[field_name] = match.group(1).lower() == 'true'
                    logger.debug(f"Found {field_name}: {incident_results[field_name]}")
                    break
        
        consequences['incident_results'] = incident_results
        
        # Extract emergency response (Question 31)
        emergency_response = {}
        
        response_patterns = [
            ('fire_ems', [r"Fire/EMS Report[^:]*:\s*(False|True)", r"Fire/EMS Report #:\s*(False|True)"]),
            ('police', [r"Police Report[^:]*:\s*(False|True)", r"Police Report #:\s*(False|True)"]),
            ('in_house_cleanup', [r"In-house cleanup:\s*(False|True)"]),
            ('other_cleanup', [r"Other Cleanup:\s*(False|True)"])
        ]
        
        for field_name, patterns in response_patterns:
            for pattern in patterns:
                match = re.search(pattern, part_iv_text, re.IGNORECASE)
                if match:
                    emergency_response[field_name] = match.group(1).lower() == 'true'
                    break
        
        consequences['emergency_response'] = emergency_response
        
        # Extract damages (Question 32)
        damage_match = re.search(
            r"Was the total damage cost more than \$500\?\s*(True|False)",
            part_iv_text, re.IGNORECASE
        )
        if damage_match:
            consequences['damages_over_500'] = damage_match.group(1).lower() == 'true'
            
            if consequences['damages_over_500']:
                damages = {}
                damage_fields = [
                    ('material_loss', r"Material Loss:\s*\$\s*([\d,]+\.?\d*)"),
                    ('carrier_damage', r"Carrier Damage:\s*\$\s*([\d,]+\.?\d*)"),
                    ('property_damage', r"Property Damage:\s*\$\s*([\d,]+\.?\d*)"),
                    ('response_cost', r"Response Cost:\s*\$\s*([\d,]+\.?\d*)"),
                    ('remediation_cleanup_cost', r"Remediation/Cleanup Cost:\s*\$\s*([\d,]+\.?\d*)")
                ]
                
                for field_name, pattern in damage_fields:
                    match = re.search(pattern, part_iv_text, re.IGNORECASE)
                    if match:
                        try:
                            damages[field_name] = float(match.group(1).replace(',', ''))
                        except ValueError:
                            pass
                
                if damages:
                    consequences['damages'] = damages
        
        # Extract fatalities (Question 33)
        fatality_match = re.search(
            r"Did the hazardous material cause[^?]*fatality\?\s*(True|False)",
            part_iv_text, re.IGNORECASE
        )
        if fatality_match:
            consequences['fatalities_caused_by_hazmat'] = fatality_match.group(1).lower() == 'true'
        
        # Extract injuries (Question 34)
        injury_match = re.search(
            r"Did the hazardous material cause[^?]*Injury\?\s*(True|False)",
            part_iv_text, re.IGNORECASE
        )
        if injury_match:
            consequences['injuries_caused_by_hazmat'] = injury_match.group(1).lower() == 'true'
        
        # Extract evacuation (Question 35)
        evac_match = re.search(
            r"Did the hazardous material cause[^?]*evacuation\?\s*(True|False)",
            part_iv_text, re.IGNORECASE
        )
        if evac_match:
            consequences['evacuation_occurred'] = evac_match.group(1).lower() == 'true'
        
        # Transportation closure (Question 36)
        transport_match = re.search(
            r"Was a major transportation[^?]*closed\?\s*(True|False)",
            part_iv_text, re.IGNORECASE
        )
        if transport_match:
            consequences['major_transportation_closed'] = transport_match.group(1).lower() == 'true'
        
        # Crash/derailment (Question 37)
        crash_match = re.search(
            r"Was the material involved in a crash[^?]*\?\s*(True|False)",
            part_iv_text, re.IGNORECASE
        )
        if crash_match:
            consequences['crash_or_derailment'] = crash_match.group(1).lower() == 'true'
        
        return consequences
    
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
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results exported to {output_file}")
        
        # Also save a flattened version for easier analysis
        flattened_df = self._flatten_nested_data(df)
        flattened_file = self.output_dir / f'phmsa_extracted_flat_{timestamp}.{format}'
        
        if format == 'parquet':
            flattened_df.to_parquet(flattened_file, index=False)
        elif format == 'csv':
            flattened_df.to_csv(flattened_file, index=False)
        elif format == 'json':
            flattened_df.to_json(flattened_file, orient='records', indent=2)
            
        logger.info(f"Flattened results exported to {flattened_file}")
    
    def _flatten_nested_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested dictionary columns for easier analysis."""
        flat_df = df.copy()
        
        # Columns that contain nested data
        nested_cols = ['incident_results', 'emergency_response', 'damages', 
                      'fatalities', 'injuries', 'evacuation_info', 'crash_info']
        
        for col in nested_cols:
            if col in flat_df.columns:
                # Check if column has any non-null values
                if flat_df[col].notna().any():
                    try:
                        # Handle dictionary columns
                        if isinstance(flat_df[col].dropna().iloc[0], dict):
                            # Extract nested data
                            nested_data = pd.json_normalize(flat_df[col].dropna())
                            # Prefix column names
                            nested_data.columns = [f"{col}_{c}" for c in nested_data.columns]
                            # Align indices
                            nested_data.index = flat_df[col].dropna().index
                            # Merge back
                            for new_col in nested_data.columns:
                                flat_df[new_col] = nested_data[new_col]
                            # Drop original nested column
                            flat_df = flat_df.drop(col, axis=1)
                    except Exception as e:
                        logger.warning(f"Could not flatten column {col}: {e}")
        
        return flat_df
    
    def process_pdf_list(self, 
                        pdf_files: List[Union[str, Path]], 
                        report_ids: Optional[List[str]] = None,
                        save_interval: int = 100) -> pd.DataFrame:
        """
        Process a list of PDF files.
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
    
    def create_consequences_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a summary of consequences similar to your CSV.
        """
        summary_data = []
        
        for _, row in df.iterrows():
            # Calculate severity score
            severity_score = 0
            if row.get('had_spillage'):
                severity_score += 1
            if row.get('had_fire'):
                severity_score += 3
            if row.get('had_explosion'):
                severity_score += 5
            if row.get('had_fatalities'):
                severity_score += 10
            if row.get('had_injuries'):
                severity_score += 5
            if row.get('had_evacuation'):
                severity_score += 3
            if row.get('had_environmental_damage'):
                severity_score += 2
            
            # Determine severity category
            if severity_score == 0:
                severity_category = 'None'
            elif severity_score <= 3:
                severity_category = 'Low'
            elif severity_score <= 10:
                severity_category = 'Medium'
            else:
                severity_category = 'High'
            
            summary_data.append({
                'report_no': row.get('report_no'),
                'date': None,  # Would need to extract from PDF
                'had_release': row.get('had_release'),
                'had_spillage': row.get('had_spillage'),
                'had_fire': row.get('had_fire'),
                'had_explosion': row.get('had_explosion'),
                'had_environmental_damage': row.get('had_environmental_damage'),
                'total_damage_cost': row.get('total_damage_cost'),
                'had_fatalities': row.get('had_fatalities'),
                'had_injuries': row.get('had_injuries'),
                'had_evacuation': row.get('had_evacuation'),
                'had_transportation_closure': row.get('had_transportation_closure'),
                'severity_score': severity_score,
                'severity_category': severity_category
            })
        
        return pd.DataFrame(summary_data)


# Test function
def test_extraction(pdf_path: str):
    """Test extraction on a single PDF."""
    scraper = PHMSAPDFScraper(output_dir='test_output')
    data = scraper.extract_from_pdf(pdf_path)
    
    if data:
        print("\n=== Extraction Results ===")
        print(f"Report: {pdf_path}")
        print(f"Had spillage: {data.get('had_spillage')}")
        print(f"Had fire: {data.get('had_fire')}")
        print(f"Had explosion: {data.get('had_explosion')}")
        print(f"Had release: {data.get('had_release')}")
        print(f"Had environmental damage: {data.get('had_environmental_damage')}")
        print(f"Had fatalities: {data.get('had_fatalities')}")
        print(f"Had injuries: {data.get('had_injuries')}")
        print(f"Had evacuation: {data.get('had_evacuation')}")
        print(f"Narrative 1 length: {len(data.get('narrative1', ''))}")
        print(f"Narrative 2 length: {len(data.get('narrative2', ''))}")
        
        # Show incident results details
        if 'incident_results' in data:
            print("\n=== Incident Results Details ===")
            for key, value in data['incident_results'].items():
                print(f"  {key}: {value}")
    else:
        print(f"Failed to extract data from {pdf_path}")
    
    return data


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        test_extraction(sys.argv[1])
    else:
        print("Usage: python script.py <path_to_pdf>")
