import os
import shutil
from pathlib import Path
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JSONLFilenameStandardizer:
    def __init__(self, data_root: str = "data", backup: bool = True):
        self.data_root = Path(data_root)
        self.backup = backup
        self.companies = ["aapl", "amgn", "dis", "lmt", "ma", "mcd", "msft", "nflx", "nvda", "sbux"]
        
        # Target format: relevance_results_{doc_type}_filter_q{num}_annotateds.jsonl
        self.target_pattern = "relevance_results_{doc_type}_filter_q{question_num}_annotateds.jsonl"
        
        # Statistics tracking
        self.stats = defaultdict(int)
        self.renamed_files = []
        self.problematic_files = []
        
    def identify_file_patterns(self):
        """Analyze all existing file patterns across companies"""
        patterns = defaultdict(list)
        
        for company in self.companies:
            qa_path = self.data_root / company / "qa"
            if not qa_path.exists():
                logger.warning(f"QA directory not found for {company}: {qa_path}")
                continue
                
            # Recursively find all .jsonl files
            for jsonl_file in qa_path.rglob("*.jsonl"):
                filename = jsonl_file.name
                
                # Classify the pattern
                if filename.endswith("_annotateds.jsonl"):
                    patterns["_annotateds.jsonl"].append(str(jsonl_file))
                elif filename.endswith("_annotated_annotated.jsonl"):
                    patterns["_annotated_annotated.jsonl"].append(str(jsonl_file))
                elif filename.endswith("_annotated.jsonl"):
                    patterns["_annotated.jsonl"].append(str(jsonl_file))
                elif filename.endswith("_annotated_annotateds.jsonl"):
                    patterns["_annotated_annotateds.jsonl"].append(str(jsonl_file))
                elif "annotatedjsonl" in filename:
                    patterns["anomaly_jsonl_inserted"].append(str(jsonl_file))
                else:
                    patterns["unknown_pattern"].append(str(jsonl_file))
                    
        return patterns
    
    def create_backup(self):
        """Create backup of original data"""
        if not self.backup:
            return
            
        backup_path = Path("data_backup")
        if backup_path.exists():
            logger.info(f"Backup already exists at {backup_path}")
            return
            
        logger.info(f"Creating backup at {backup_path}")
        shutil.copytree(self.data_root, backup_path)
        logger.info("✅ Backup created successfully")
    
    def parse_filename_components(self, filename: str):
        """Extract doc_type and question_num from filename"""
        try:
            # Remove the .jsonl extension
            base_name = filename.replace(".jsonl", "")
            
            # Handle the anomaly case first
            if "annotatedjsonl" in base_name:
                base_name = base_name.replace("annotatedjsonl", "")
                
            # Remove various annotated suffixes
            suffixes_to_remove = [
                "_annotated_annotated",
                "_annotated_annotateds", 
                "_annotateds",
                "_annotated"
            ]
            
            for suffix in suffixes_to_remove:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            
            # Expected format now: relevance_results_{doc_type}_filter_q{num}
            if not base_name.startswith("relevance_results_"):
                raise ValueError(f"Unexpected prefix: {base_name}")
                
            # Remove prefix
            remaining = base_name[len("relevance_results_"):]
            
            # Split by _filter_q
            if "_filter_q" not in remaining:
                raise ValueError(f"Missing _filter_q: {remaining}")
                
            doc_type, question_part = remaining.split("_filter_q", 1)
            question_num = int(question_part)
            
            return doc_type, question_num
            
        except Exception as e:
            logger.error(f"Failed to parse filename {filename}: {e}")
            return None, None
    
    def generate_target_filename(self, doc_type: str, question_num: int):
        """Generate standardized filename"""
        return f"relevance_results_{doc_type}_filter_q{question_num}_annotateds.jsonl"
    
    def standardize_file(self, file_path: Path, dry_run: bool = False):
        """Standardize a single file"""
        original_filename = file_path.name
        doc_type, question_num = self.parse_filename_components(original_filename)
        
        if doc_type is None or question_num is None:
            self.problematic_files.append(str(file_path))
            self.stats["parse_failures"] += 1
            return False
            
        target_filename = self.generate_target_filename(doc_type, question_num)
        
        # Check if already in correct format
        if original_filename == target_filename:
            self.stats["already_correct"] += 1
            return True
            
        target_path = file_path.parent / target_filename
        
        # Check for conflicts
        if target_path.exists() and target_path != file_path:
            logger.warning(f"Conflict: Target file already exists: {target_path}")
            self.stats["conflicts"] += 1
            return False
            
        # Perform the rename
        if not dry_run:
            try:
                file_path.rename(target_path)
                self.renamed_files.append((str(file_path), str(target_path)))
                logger.info(f"Renamed: {file_path.name} → {target_filename}")
                self.stats["renamed"] += 1
                return True
            except Exception as e:
                logger.error(f"Failed to rename {file_path}: {e}")
                self.stats["rename_failures"] += 1
                return False
        else:
            logger.info(f"[DRY RUN] Would rename: {file_path.name} → {target_filename}")
            self.stats["would_rename"] += 1
            return True
    
    def standardize_company(self, company: str, dry_run: bool = False):
        """Standardize all files for a single company"""
        logger.info(f"Processing company: {company.upper()}")
        
        qa_path = self.data_root / company / "qa"
        if not qa_path.exists():
            logger.warning(f"QA directory not found for {company}")
            return
            
        # Find all JSONL files
        jsonl_files = list(qa_path.rglob("*.jsonl"))
        logger.info(f"Found {len(jsonl_files)} JSONL files for {company}")
        
        success_count = 0
        for jsonl_file in jsonl_files:
            if self.standardize_file(jsonl_file, dry_run):
                success_count += 1
                
        logger.info(f"Successfully processed {success_count}/{len(jsonl_files)} files for {company}")
    
    def standardize_all(self, dry_run: bool = False):
        """Standardize all files across all companies"""
        logger.info("=== JSONL Filename Standardization ===")
        
        if not dry_run:
            self.create_backup()
        
        # Analyze current patterns
        logger.info("Analyzing current filename patterns...")
        patterns = self.identify_file_patterns()
        
        for pattern, files in patterns.items():
            logger.info(f"{pattern}: {len(files)} files")
            
        # Process each company
        for company in self.companies:
            self.standardize_company(company, dry_run)
            
        # Print final statistics
        self.print_summary()
    
    def print_summary(self):
        """Print summary of standardization process"""
        logger.info("=== Standardization Summary ===")
        for key, value in sorted(self.stats.items()):
            logger.info(f"{key}: {value}")
            
        if self.problematic_files:
            logger.warning(f"Problematic files ({len(self.problematic_files)}):")
            for file_path in self.problematic_files:
                logger.warning(f"  - {file_path}")
                
        if self.renamed_files:
            logger.info(f"Successfully renamed {len(self.renamed_files)} files")
            
    def verify_standardization(self):
        """Verify that all files now follow the standard pattern"""
        logger.info("=== Verification ===")
        
        non_standard_files = []
        total_files = 0
        
        for company in self.companies:
            qa_path = self.data_root / company / "qa"
            if not qa_path.exists():
                continue
                
            for jsonl_file in qa_path.rglob("*.jsonl"):
                total_files += 1
                filename = jsonl_file.name
                
                if not filename.endswith("_annotateds.jsonl"):
                    non_standard_files.append(str(jsonl_file))
                    
        logger.info(f"Total JSONL files: {total_files}")
        logger.info(f"Non-standard files: {len(non_standard_files)}")
        
        if non_standard_files:
            logger.warning("Non-standard files found:")
            for file_path in non_standard_files:
                logger.warning(f"  - {file_path}")
        else:
            logger.info("✅ All files are now standardized!")

def main():
    """Main function with configuration options"""
    
    # Configuration
    DRY_RUN = False  # Set to True to see what would be renamed without actually doing it
    CREATE_BACKUP = True  # Set to False to skip backup creation
    
    # Initialize standardizer
    standardizer = JSONLFilenameStandardizer(backup=CREATE_BACKUP)
    
    if DRY_RUN:
        logger.info("🔍 Running in DRY RUN mode - no files will be renamed")
    else:
        logger.info("⚠️  Running in LIVE mode - files will be renamed")
        
    # Run standardization
    standardizer.standardize_all(dry_run=DRY_RUN)
    
    # Verify results (only if not dry run)
    if not DRY_RUN:
        standardizer.verify_standardization()
        
    logger.info("✅ Standardization process completed")

if __name__ == "__main__":
    main()