from logging import getLogger
from pathlib import Path
from typing import Any, Optional, TypeVar, Dict, List
from oceanprotocol_job_details.ocean import JobDetails
import json
import pandas as pd
import re
from datetime import datetime

# =============================== IMPORT LIBRARY ====================
# Data cleaning and privacy protection libraries
# =============================== END ===============================

T = TypeVar("T")

logger = getLogger(__name__)


class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: Optional[Any] = None
        self.cleaned_data = None
        self.removed_columns = []
        self.sensitive_columns_removed = []
        
        # Sensitive data patterns - focused on actual sensitive information
        self.sensitive_patterns = {
            # pw, password 등 민감정보가 대소문자 구분 없이, 공백/콜론/등호(:, =) 여러 개 뒤에 값이 오면 모두 감지
            'password': r'(?i)(password|pw|passwd|pwd)\s*[:= ]+\s*\S+',
            'login_credential': r'(?i)(login|username|user)\s*[:= ]+\s*\S+',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'file_path': r'[A-Z]:\\.*\.(?:pst|nsf|log|tmp)',
            'internal_folder': r'\\[A-Za-z_]+\\[A-Za-z\s]+\\',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        }
        
        # Business vs Personal classification patterns
        self.business_patterns = {
            'business_email_domains': [
                'enron.com', 'company.com', 'corp.com', 'business.com',
                'enterprise.com', 'office.com', 'work.com', 'firm.com'
            ],
            'business_phone_patterns': [
                r'\b\d{3}-\d{3}-\d{4}\b',  # Standard business format
                r'\b\d{3}\.\d{3}\.\d{4}\b',  # Dotted format
                r'\b\d{10}\b'  # Plain 10 digits
            ],
            'business_address_indicators': [
                'street', 'avenue', 'boulevard', 'drive', 'road',
                'plaza', 'center', 'tower', 'building', 'suite',
                'floor', 'office', 'corporate', 'headquarters'
            ],
            'personal_email_domains': [
                'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
                'aol.com', 'icloud.com', 'live.com', 'msn.com',
                'protonmail.com', 'mail.com'
            ],
            'personal_phone_indicators': [
                'cell', 'mobile', 'home', 'personal'
            ]
        }

    def _validate_input(self) -> None:
        if not self._job_details.files:
            logger.warning("No files found")
            raise ValueError("No files found")

    def _detect_file_type(self, filename: str) -> str:
        """Detect file type based on extension"""
        ext = Path(filename).suffix.lower()
        if ext in ['.csv', '.tsv']:
            return 'csv'
        elif ext in ['.json']:
            return 'json'
        elif ext in ['.xlsx', '.xls']:
            return 'excel'
        else:
            return 'text'

    def _load_data(self, filename: str) -> pd.DataFrame:
        """Load data based on file type"""
        file_type = self._detect_file_type(filename)
        
        try:
            if file_type == 'csv':
                return pd.read_csv(filename)
            elif file_type == 'json':
                return pd.read_json(filename)
            elif file_type == 'excel':
                return pd.read_excel(filename)
            else:
                # For unsupported file types, try to read as CSV
                return pd.read_csv(filename)
        except Exception as e:
            logger.error(f"Error loading file {filename}: {e}")
            raise

    def _identify_sensitive_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that might contain sensitive information"""
        sensitive_columns = []
        
        for col in df.columns:
            col_str = str(col).lower()
            # Check column name patterns - focus on actual sensitive data
            if any(pattern in col_str for pattern in ['password', 'pw', 'passwd', 'pwd', 'credential', 'secret', 'key']):
                sensitive_columns.append(col)
                continue
                
            # Check data patterns in first few rows
            sample_data = df[col].dropna().astype(str).head(100)
            for pattern_name, pattern in self.sensitive_patterns.items():
                if sample_data.str.contains(pattern, regex=True).any():
                    sensitive_columns.append(col)
                    break
                    
        return sensitive_columns

    def _remove_sensitive_data(self, df: pd.DataFrame, sensitive_columns: List[str]) -> pd.DataFrame:
        """Remove sensitive values in cells instead of dropping columns"""
        df_cleaned = df.copy()
        for col in sensitive_columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].astype(str).apply(
                    lambda x: '' if any(re.search(pattern, x, re.IGNORECASE) for pattern in self.sensitive_patterns.values()) else x
                )
                self.sensitive_columns_removed.append(col)
        return df_cleaned

    def _remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that are likely unnecessary"""
        unnecessary_patterns = [
            'index', 'id', 'row', 'unnamed', 'temp', 'tmp', 'old', 'backup',
            'duplicate', 'copy', 'test', 'debug', 'log', 'timestamp'
        ]
        
        columns_to_remove = []
        for col in df.columns:
            col_str = str(col).lower()
            if any(pattern in col_str for pattern in unnecessary_patterns):
                # Check if column has unique values (like index)
                if df[col].nunique() == len(df):
                    columns_to_remove.append(col)
                # Check if column is mostly empty
                elif df[col].isna().sum() > len(df) * 0.8:
                    columns_to_remove.append(col)
                    
        df_cleaned = df.drop(columns=columns_to_remove)
        self.removed_columns = columns_to_remove
        
        return df_cleaned

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform general data cleaning"""
        df_cleaned = df.copy()
        
        # Remove duplicate rows
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)
        
        # Handle missing values
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype in ['object', 'string']:
                df_cleaned[col] = df_cleaned[col].fillna('Unknown')
            else:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        
        # Clean string columns
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype in ['object', 'string']:
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
                df_cleaned[col] = df_cleaned[col].replace(['nan', 'None', 'null'], 'Unknown')
        
        return df_cleaned, duplicates_removed

    def _is_business_email(self, email: str) -> bool:
        """Determine if email is business or personal"""
        if not email or '@' not in email:
            return False
        
        domain = email.split('@')[-1].lower()
        
        # Check if it's a known business domain
        if domain in self.business_patterns['business_email_domains']:
            return True
        
        # Check if it's a known personal domain
        if domain in self.business_patterns['personal_email_domains']:
            return False
        
        # Heuristics for unknown domains
        business_indicators = ['corp', 'inc', 'llc', 'ltd', 'company', 'business', 'enterprise']
        if any(indicator in domain for indicator in business_indicators):
            return True
            
        return False  # Default to personal for unknown domains

    def _is_business_phone(self, phone: str, context: str = "") -> bool:
        """Determine if phone number is business or personal"""
        if not phone:
            return False
        
        # Check context for indicators
        context_lower = context.lower()
        if any(indicator in context_lower for indicator in self.business_patterns['personal_phone_indicators']):
            return False
        
        # Check for business patterns
        for pattern in self.business_patterns['business_phone_patterns']:
            if re.match(pattern, phone):
                return True
        
        # Check for business context keywords
        business_keywords = ['office', 'desk', 'work', 'business', 'company', 'corporate']
        if any(keyword in context_lower for keyword in business_keywords):
            return True
            
        return False  # Default to personal

    def _is_business_address(self, address: str) -> bool:
        """Determine if address is business or personal"""
        if not address:
            return False
        
        address_lower = address.lower()
        
        # Check for business indicators
        if any(indicator in address_lower for indicator in self.business_patterns['business_address_indicators']):
            return True
        
        # Check for business keywords
        business_keywords = ['corporate', 'headquarters', 'office', 'building', 'tower', 'center', 'plaza']
        if any(keyword in address_lower for keyword in business_keywords):
            return True
        
        # Check for suite/floor numbers
        if re.search(r'suite|floor|office|room', address_lower):
            return True
            
        return False  # Default to personal

    def _classify_contact_info(self, text: str) -> Dict[str, List[str]]:
        """Classify contact information as business or personal"""
        business_info = []
        personal_info = []
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        for email in emails:
            if self._is_business_email(email):
                business_info.append(f"email: {email}")
            else:
                personal_info.append(f"email: {email}")
        
        # Extract phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, text)
        
        for phone in phones:
            # Get context around the phone number
            phone_index = text.find(phone)
            start = max(0, phone_index - 50)
            end = min(len(text), phone_index + len(phone) + 50)
            context = text[start:end]
            
            if self._is_business_phone(phone, context):
                business_info.append(f"phone: {phone}")
            else:
                personal_info.append(f"phone: {phone}")
        
        return {
            'business': business_info,
            'personal': personal_info
        }

    def run(self) -> "Algorithm":
        # 1. Initialize results type
        self.results = {}

        # 2. validate input here
        self._validate_input()

        # 3. get input files here
        input_files = self._job_details.files.files[0].input_files
        filename = str(input_files[0])

        # 4. run algorithm here: data cleaning and privacy protection
        try:
            # Load data
            df = self._load_data(filename)
            
            # Identify sensitive columns
            sensitive_columns = self._identify_sensitive_columns(df)
            
            # Remove unnecessary columns
            df_cleaned = self._remove_unnecessary_columns(df)
            
            # Remove sensitive data
            df_cleaned = self._remove_sensitive_data(df_cleaned, sensitive_columns)
            
            # General data cleaning
            df_cleaned, duplicates_removed = self._clean_data(df_cleaned)
            
            # Classify contact information in text columns
            contact_classification = {'business': [], 'personal': []}
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype in ['object', 'string']:
                    # Sample some rows for contact classification
                    sample_text = ' '.join(df_cleaned[col].dropna().astype(str).head(100))
                    if sample_text:
                        col_classification = self._classify_contact_info(sample_text)
                        contact_classification['business'].extend(col_classification['business'])
                        contact_classification['personal'].extend(col_classification['personal'])
            
            self.cleaned_data = df_cleaned
            
            # Prepare results
            self.results = {
                "original_rows": len(df),
                "cleaned_rows": len(df_cleaned),
                "duplicates_removed": duplicates_removed,
                "columns_removed": len(self.removed_columns),
                "sensitive_columns_removed": len(self.sensitive_columns_removed),
                "removed_columns": self.removed_columns,
                "sensitive_columns_removed": self.sensitive_columns_removed,
                "contact_classification": contact_classification,
                "business_contacts": len(contact_classification['business']),
                "personal_contacts": len(contact_classification['personal']),
                "processing_timestamp": datetime.now().isoformat(),
                "file_type": "dataframe"
            }
                
        except Exception as e:
            logger.exception(f"Error processing data: {e}")
            self.results = {"error": str(e)}

        # 5. save results here (handled in save_result)

        # 6. return self
        return self

    def save_result(self, path: Path) -> None:
        # 7. define/add result path here
        result_path = path / "result.json"
        cleaned_data_path = path / "cleaned_data"

        with open(result_path, "w", encoding="utf-8") as f:
            try:
                # 8. save results here
                json.dump(self.results, f, indent=2)
            except Exception as e:
                logger.exception(f"Error saving data: {e}")
        
        # Save cleaned data
        try:
            if self.cleaned_data is not None:
                # Save as CSV
                self.cleaned_data.to_csv(cleaned_data_path.with_suffix('.csv'), index=False)
                # Also save as JSON for compatibility
                self.cleaned_data.to_json(cleaned_data_path.with_suffix('.json'), orient='records', indent=2)
        except Exception as e:
            logger.exception(f"Error saving cleaned data: {e}")
       