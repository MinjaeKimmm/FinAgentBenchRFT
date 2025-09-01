import json
import re
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import random

class QRELSRFTGenerator:
    """Generate RFT data using QRELS format with quality filtering"""
    
    def __init__(self, base_dir: str = "part_1"):
        self.base_dir = Path(base_dir)
        self.raw_data_dir = self.base_dir / "output" / "raw_data"
        self.output_dir = self.base_dir / "output" / "rft_data"
        self.companies = ["aapl", "amgn", "dis", "lmt", "ma", "mcd", "msft", "nflx", "nvda", "sbux"]
        
        # Statistics tracking
        self.stats = {
            "total_samples": 0,
            "all_zero_filtered": 0,
            "kept_samples": 0,
            "document_samples": 0,
            "chunk_samples": 0
        }
        
    def generate_all(self):
        """Generate complete RFT structure with QRELS format"""
        print("🚀 Generating Complete RFT Data Structure with QRELS...")
        
        # Create directory structure
        self._create_directory_structure()
        
        # Generate datasets with QRELS format
        self._generate_datasets()
        
        # Generate combined datasets
        self._generate_combined_datasets()
        
        print("✅ Complete QRELS RFT structure generated successfully!")
        self._print_summary()
    
    def _create_directory_structure(self):
        """Create the complete directory structure"""
        print("📁 Creating directory structure...")
        
        directories = [
            # Dataset directories
            self.output_dir / "dataset" / "document_ranking" / "train",
            self.output_dir / "dataset" / "document_ranking" / "eval",
            self.output_dir / "dataset" / "chunk_ranking" / "train", 
            self.output_dir / "dataset" / "chunk_ranking" / "eval",
            
            # Combined datasets
            self.output_dir / "combined_datasets",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _generate_datasets(self):
        """Generate individual company dataset files with QRELS format"""
        print("📊 Generating datasets with QRELS format...")
        
        for company in self.companies:
            print(f"  Processing {company.upper()}...")
            
            # Document ranking
            doc_train_count = self._process_document_ranking(company, "train")
            doc_eval_count = self._process_document_ranking(company, "eval")
            
            # Chunk ranking  
            chunk_train_count = self._process_chunk_ranking(company, "train")
            chunk_eval_count = self._process_chunk_ranking(company, "eval")
            
            self.stats["document_samples"] += doc_train_count + doc_eval_count
            self.stats["chunk_samples"] += chunk_train_count + chunk_eval_count
            
            print(f"    ✅ {company}: Doc({doc_train_count}+{doc_eval_count}), Chunk({chunk_train_count}+{chunk_eval_count})")
        
        print(f"📊 Total kept: {self.stats['kept_samples']}, Filtered: {self.stats['all_zero_filtered']}")
    
    def _process_document_ranking(self, company: str, split: str) -> int:
        """Process document ranking data for a company"""
        
        input_file = self.raw_data_dir / company / "document-ranking" / f"{split}.jsonl"
        output_file = self.output_dir / "dataset" / "document_ranking" / split / f"{company}.jsonl"
        
        if not input_file.exists():
            return 0
        
        samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                try:
                    sample = json.loads(line.strip())
                    qrels_sample = self._convert_document_sample_to_qrels(sample)
                    if qrels_sample and self._should_keep_sample(qrels_sample):
                        samples.append(qrels_sample)
                        self.stats["kept_samples"] += 1
                    else:
                        self.stats["all_zero_filtered"] += 1
                    
                    self.stats["total_samples"] += 1
                except json.JSONDecodeError:
                    continue
        
        if samples:
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        return len(samples)
    
    def _process_chunk_ranking(self, company: str, split: str) -> int:
        """Process chunk ranking data for a company - FIXED VERSION"""
        
        input_file = self.raw_data_dir / company / "chunk-ranking" / f"{split}.jsonl"
        output_file = self.output_dir / "dataset" / "chunk_ranking" / split / f"{company}.jsonl"
        
        if not input_file.exists():
            return 0
        
        samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                try:
                    sample = json.loads(line.strip())
                    
                    # SIMPLE CONVERSION - NO SAMPLING OR CONTENT MANIPULATION
                    qrels_sample = self._convert_chunk_sample_to_qrels(sample)
                    
                    if qrels_sample and self._should_keep_sample(qrels_sample):
                        samples.append(qrels_sample)
                        self.stats["kept_samples"] += 1
                    else:
                        self.stats["all_zero_filtered"] += 1
                    
                    self.stats["total_samples"] += 1
                    
                except json.JSONDecodeError:
                    continue
        
        if samples:
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        return len(samples)
    
    def _convert_document_sample_to_qrels(self, sample: Dict) -> Optional[Dict]:
        """Convert document ranking sample to QRELS format - FIXED VERSION"""
        
        ranking = sample.get("document_ranking", [])
        score_counts = sample.get("document_score_counts", {})
        
        if not ranking or not score_counts:
            return None
        
        # Fix 2: Filter out all-zero questions
        # Check if any document has relevant chunks
        has_relevant_chunks = False
        for doc_type, counts in score_counts.items():
            if counts.get("2", 0) > 0 or counts.get("1", 0) > 0:
                has_relevant_chunks = True
                break
        
        if not has_relevant_chunks:
            return None  # Filter out - no learning signal
        
        # Fix 1: Handle tied relevance scores properly
        # Group documents by their (count_2, count_1, count_0) signature
        doc_signatures = {}
        for i, doc_type in enumerate(["def14a", "10k", "10q", "8k", "earnings"]):
            if doc_type in score_counts:
                counts = score_counts[doc_type]
                signature = (counts.get("2", 0), counts.get("1", 0), counts.get("0", 0))
                
                if signature not in doc_signatures:
                    doc_signatures[signature] = []
                doc_signatures[signature].append(i)
        
        # Sort signatures by relevance (same logic as original ranking)
        sorted_signatures = sorted(doc_signatures.keys(), 
                                key=lambda x: (-x[0], -x[1], x[2]))
        
        # Assign QRELS scores - tied documents get same score
        qrel = {}
        current_relevance = len(sorted_signatures) - 1  # Start from highest
        
        for signature in sorted_signatures:
            doc_indices = doc_signatures[signature]
            
            # All documents with this signature get the same relevance score
            for doc_idx in doc_indices:
                qrel[str(doc_idx)] = current_relevance
            
            current_relevance -= 1  # Next group gets lower score
        
        return {
            "messages": sample["messages"],
            "qrel": qrel,
            "metadata": {
                **sample.get("metadata", {}),
                "task_type": "document_ranking",
                "num_documents": len(ranking),
                "max_score": max(qrel.values()) if qrel else 0,
                "min_score": min(qrel.values()) if qrel else 0,
                "relevant_docs": sum(1 for s in qrel.values() if s > 0),
                "tied_groups": len(sorted_signatures),  # New: track number of tied groups
                "signature_groups": [{"signature": sig, "doc_indices": doc_signatures[sig]} for sig in sorted_signatures]  # New: track the signatures and their docs
            }
        }
    
    def _convert_chunk_sample_to_qrels(self, sample: Dict) -> Optional[Dict]:
        """Convert chunk ranking sample to QRELS format - FIXED VERSION"""
        
        scores = sample.get("chunk_scores", [])
        if not scores:
            return None
        
        # NO SAMPLING - USE ALL CHUNKS AS REQUESTED
        # The original preprocessor already created perfect formatted content
        
        # Create QRELS format directly from scores
        qrel = {}
        for idx, score in enumerate(scores):
            qrel[str(idx)] = score
        
        return {
            "messages": sample["messages"],  # ✅ Use original perfect content - DON'T TOUCH IT!
            "qrel": qrel,
            "metadata": {
                **sample.get("metadata", {}),
                "task_type": "chunk_ranking",
                "num_chunks": len(scores),
                "positive_chunks": sum(1 for s in scores if s > 0),
                "zero_chunks": sum(1 for s in scores if s == 0),
                "max_score": max(scores) if scores else 0
            }
        }
    
    def _should_keep_sample(self, sample: Dict) -> bool:
        """Filter out all-zero samples and other poor quality samples"""
        
        qrel = sample.get("qrel", {})
        if not qrel:
            return False
        
        # Get all relevance scores
        scores = list(qrel.values())
        
        # Filter 1: All zeros (no learning signal)
        if all(score == 0 for score in scores):
            return False
        
        # Filter 2: All same non-zero values (no ranking preference)
        unique_scores = set(scores)
        if len(unique_scores) == 1 and list(unique_scores)[0] != 0:
            return False
        
        return True

    def _generate_combined_datasets(self):
        """Generate combined dataset files with 80 train, 20 eval (removed 'balanced' from filename)"""
        print("🔗 Generating combined datasets (80 train, 20 eval)...")
        
        tasks = ["document_ranking", "chunk_ranking"]
        
        for task_type in tasks:
            # Collect all samples from all companies
            all_train_samples = []
            all_eval_samples = []
            
            for company in self.companies:
                # Load train samples
                train_file = self.output_dir / "dataset" / task_type / "train" / f"{company}.jsonl"
                if train_file.exists():
                    with open(train_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                sample = json.loads(line.strip())
                                sample["metadata"]["source_company"] = company
                                all_train_samples.append(sample)
                
                # Load eval samples
                eval_file = self.output_dir / "dataset" / task_type / "eval" / f"{company}.jsonl"
                if eval_file.exists():
                    with open(eval_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                sample = json.loads(line.strip())
                                sample["metadata"]["source_company"] = company
                                all_eval_samples.append(sample)
            
            # Balanced sampling across companies and categories
            if all_train_samples:
                balanced_train = self._balanced_sample(all_train_samples, 80)
                train_output = self.output_dir / "combined_datasets" / f"{task_type}_train.jsonl"  # Removed "_balanced"
                with open(train_output, 'w', encoding='utf-8') as f:
                    for sample in balanced_train:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                print(f"    ✅ {task_type}_train.jsonl: {len(balanced_train)} samples")
            
            if all_eval_samples:
                balanced_eval = self._balanced_sample(all_eval_samples, 20)
                eval_output = self.output_dir / "combined_datasets" / f"{task_type}_eval.jsonl"  # Removed "_balanced"
                with open(eval_output, 'w', encoding='utf-8') as f:
                    for sample in balanced_eval:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                print(f"    ✅ {task_type}_eval.jsonl: {len(balanced_eval)} samples")
    
    def _balanced_sample(self, samples: List[Dict], target_count: int) -> List[Dict]:
        """Sample balanced across companies and categories"""
        
        if len(samples) <= target_count:
            return samples
        
        # Group by company and category for balanced sampling
        groups = {}
        for sample in samples:
            company = sample["metadata"].get("source_company", "unknown")
            category = sample["metadata"].get("category", "unknown")
            key = f"{company}_{category}"
            
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)
        
        # Calculate samples per group
        num_groups = len(groups)
        base_per_group = target_count // num_groups
        remainder = target_count % num_groups
        
        balanced_samples = []
        group_keys = sorted(groups.keys())  # Deterministic ordering
        
        for i, key in enumerate(group_keys):
            group_samples = groups[key]
            
            # Some groups get one extra sample for remainder
            samples_for_this_group = base_per_group + (1 if i < remainder else 0)
            samples_for_this_group = min(samples_for_this_group, len(group_samples))
            
            # Random sample from this group
            if samples_for_this_group > 0:
                sampled = random.sample(group_samples, samples_for_this_group)
                balanced_samples.extend(sampled)
        
        # If we still need more samples, randomly add from all
        if len(balanced_samples) < target_count:
            remaining_samples = [s for s in samples if s not in balanced_samples]
            needed = target_count - len(balanced_samples)
            if remaining_samples and needed > 0:
                additional = random.sample(remaining_samples, min(needed, len(remaining_samples)))
                balanced_samples.extend(additional)
        
        # Shuffle final result
        random.shuffle(balanced_samples)
        return balanced_samples[:target_count]
    
    def _print_summary(self):
        print("\n" + "="*70)
        print("🎉 QRELS RFT DATA GENERATION COMPLETE")
        print("="*70)

        dataset_files = len(list((self.output_dir / "dataset").rglob("*.jsonl")))
        combined_files = len(list((self.output_dir / "combined_datasets").rglob("*.jsonl")))
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"📊 Dataset Files: {dataset_files}")
        print(f"🔗 Combined Files: {combined_files}")

        print("\n📈 Data Quality Statistics:")
        total = self.stats['total_samples']
        filtered = self.stats['all_zero_filtered']
        kept = self.stats['kept_samples']
        print(f"   Total samples processed: {total}")
        print(f"   All-zero samples filtered: {filtered}")
        print(f"   High-quality samples kept: {kept}")
        filter_rate = (filtered / total * 100) if total > 0 else 0.0
        print(f"   Filter rate: {filter_rate:.1f}%")

def main():
    """Main execution"""
    generator = QRELSRFTGenerator()
    generator.generate_all()

if __name__ == "__main__":
    main()