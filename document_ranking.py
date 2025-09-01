import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional
import random
from bs4 import BeautifulSoup
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentLevelRFTPreprocessor:
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.companies = ["aapl", "amgn", "dis", "lmt", "ma", "mcd", "msft", "nflx", "nvda", "sbux"]
        
        # Document types mapping
        self.doc_types = ["def14a", "10k", "10q", "8k", "earnings"]
        self.doc_type_display = {
            "def14a": "DEF14A", 
            "10k": "10-K", 
            "10q": "10-Q", 
            "8k": "8-K", 
            "earnings": "Earnings"
        }
        
        # Document file mapping
        self.doc_files = {
            "def14a": "def14a.json",
            "10k": "10-k.html", 
            "10q": "10-q.html",
            "8k": "8-k.json",
            "earnings": "earnings.html"
        }
        
        # CSV filename mapping (handle special cases)
        self.csv_files = {
            "aapl": "aapl.csv",
            "amgn": "amgn.csv", 
            "dis": "dis.csv",
            "lmt": "lmt.csv",
            "ma": "mastercard.csv",  # Special case
            "mcd": "mcd.csv",
            "msft": "msft.csv",
            "nflx": "nflx.csv", 
            "nvda": "nvda.csv",
            "sbux": "sbux.csv"
        }
        
        # Statistics tracking
        self.stats = defaultdict(int)
        
    def csv_category_to_directory_name(self, category: str) -> str:
        """Convert CSV category name to directory name"""
        mapping = {
            "Management Commentary": "Management_Commentary",
            "Analyst Q&A": "Analyst_Q&A", 
            "Earnings result/Financials": "Earnings_result/Financials",
            "Industry & Market": "Industry_&_Market",  
            "Macro & Economics": "Macro_&_Economics",
            "Risks & Challenges": "Risks_&_Challenges",
            "Investor Sentiment": "Investor_Sentiment",
            "Operating metric": "Operating_metric",
            "Compensation": "Compensation",
            "Guidance": "Guidance"
        }
        
        if category in mapping:
            return mapping[category]
        else:
            # Fallback: replace spaces with underscores
            return category.replace(" ", "_").replace("&", "&")
    
    def load_company_questions(self, company: str) -> List[Tuple[str, str, int]]:
        """Load questions from company CSV file"""
        csv_file = self.data_root / company / self.csv_files[company]
        
        if not csv_file.exists():
            logger.warning(f"CSV file not found: {csv_file}")
            return []

        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                logger.info(f"Loaded {len(df)} questions from {csv_file} using {encoding} encoding")
                
                # Group by category to get question indices within each category
                questions_with_indices = []
                category_counters = defaultdict(int)
                
                for _, row in df.iterrows():
                    category = row['category']
                    question = row['question']
                    
                    category_counters[category] += 1
                    question_index = category_counters[category]
                    
                    questions_with_indices.append((category, question, question_index))
                    
                return questions_with_indices
                
            except UnicodeDecodeError:
                logger.debug(f"Failed to read {csv_file} with {encoding} encoding, trying next...")
                continue
            except Exception as e:
                logger.error(f"Error loading CSV {csv_file} with {encoding}: {e}")
                continue
        
        logger.error(f"Failed to read {csv_file} with any supported encoding")
        return []
    
    def extract_full_document_content(self, company: str, doc_type: str) -> str:
        """Extract full content from each document without truncation"""
        doc_file = self.data_root / company / self.doc_files[doc_type]
        
        if not doc_file.exists():
            return "Document not available"
            
        try:
            if doc_file.suffix == '.json':
                # Handle JSON files (8-k, def14a)
                with open(doc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, dict):
                    # Try to get meaningful content from various possible fields
                    content = ""
                    for key in ['text', 'content', 'description', 'summary', 'body']:
                        if key in data and data[key]:
                            content = str(data[key])
                            break
                    
                    if not content and data:
                        # Fallback: get first non-empty string value
                        for value in data.values():
                            if isinstance(value, str) and len(value.strip()) > 50:
                                content = value
                                break
                                
                    # If still no content, serialize the entire dict
                    if not content:
                        content = json.dumps(data, indent=2)
                                
                elif isinstance(data, list) and data:
                    # If it's a list, get first meaningful item or serialize all
                    if len(data) == 1:
                        content = str(data[0])
                    else:
                        content = json.dumps(data, indent=2)
                else:
                    content = str(data)
                    
            else:
                # Handle HTML files (10-k, 10-q, earnings)
                with open(doc_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Parse HTML and extract text
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace but preserve structure
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Clean content but do NOT truncate
            if content:
                # Remove excessive whitespace but keep full content
                content = re.sub(r'\s+', ' ', content.strip())
                return content
            else:
                return "No readable content found"
                
        except Exception as e:
            logger.warning(f"Error extracting content from {doc_file}: {e}")
            return "Error reading document"
    
    def load_chunk_scores_from_jsonl(self, jsonl_file: Path) -> List[int]:
        """Load chunk scores from a JSONL file"""
        if not jsonl_file.exists():
            return []
            
        scores = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data = json.loads(line)
                            # Priority: relevance_score first, then score as fallback
                            if 'relevance_score' in data and data['relevance_score'] is not None:
                                scores.append(data['relevance_score'])
                            elif 'score' in data and data['score'] is not None:
                                scores.append(data['score'])
                            else:
                                logger.warning(f"No valid score field found in line: {line[:100]}...")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line in {jsonl_file}: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error reading JSONL file {jsonl_file}: {e}")
            
        return scores
    
    def rank_documents_by_relevance(self, document_chunk_scores: Dict[str, List[int]]) -> Tuple[List[int], Dict[str, Dict[str, int]]]:
        """Rank documents by chunk score distribution and return ranking + score counts"""
        
        # Calculate score counts for each document
        doc_rankings = []
        score_counts = {}
        
        for doc_type in self.doc_types:
            scores = document_chunk_scores.get(doc_type, [])
            count_2 = scores.count(2)
            count_1 = scores.count(1) 
            count_0 = scores.count(0)
            
            doc_rankings.append({
                'doc_type': doc_type,
                'count_2': count_2,
                'count_1': count_1, 
                'count_0': count_0,
                'original_index': self.doc_types.index(doc_type)
            })
            
            score_counts[doc_type] = {
                "2": count_2,
                "1": count_1, 
                "0": count_0
            }
        
        # Sort by: count_2 DESC, count_1 DESC, count_0 ASC
        doc_rankings.sort(key=lambda x: (-x['count_2'], -x['count_1'], x['count_0']))
        
        # Return ranked indices (most relevant first)
        ranking = [doc['original_index'] for doc in doc_rankings]
        
        return ranking, score_counts
    
    def get_document_scores_for_question(self, company: str, category: str, question_index: int) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        """Get document chunk scores and full content for a specific question"""
        
        # Convert category to directory name
        category_dir = self.csv_category_to_directory_name(category)
        qa_path = self.data_root / company / "qa" / category_dir
        
        if not qa_path.exists():
            logger.warning(f"Category directory not found: {qa_path}")
            return {}, {}
            
        document_chunk_scores = {}
        document_content = {}
        
        for doc_type in self.doc_types:
            # Try double annotated pattern first (AAPL format)
            jsonl_filename = f"relevance_results_{doc_type}_filter_q{question_index}_annotated_annotated.jsonl"
            jsonl_path = qa_path / jsonl_filename
            
            # If double annotated doesn't exist, try single annotated pattern
            if not jsonl_path.exists():
                jsonl_filename = f"relevance_results_{doc_type}_filter_q{question_index}_annotateds.jsonl"
                jsonl_path = qa_path / jsonl_filename
            
            # Load chunk scores from JSONL
            chunk_scores = self.load_chunk_scores_from_jsonl(jsonl_path)
            document_chunk_scores[doc_type] = chunk_scores
            
            # Get full document content (no truncation)
            full_content = self.extract_full_document_content(company, doc_type)
            document_content[doc_type] = full_content
            
            # Track statistics
            if chunk_scores:
                self.stats[f"{doc_type}_files_with_data"] += 1
                self.stats[f"{doc_type}_total_chunks"] += len(chunk_scores)
            else:
                self.stats[f"{doc_type}_empty_files"] += 1
                
        return document_chunk_scores, document_content
    
    def create_ranking_prompt(self, question: str, document_content: Dict[str, str]) -> str:
        """Create a clear ranking task prompt with full document content"""
        
        prompt = f"""Rank the following financial document types by relevance to answer the question. Provide your ranking as a list of indices from most relevant to least relevant.

Question: {question}

Document Types to rank:
"""
        
        for i, doc_type in enumerate(self.doc_types):
            display_name = self.doc_type_display[doc_type]
            prompt += f"[Document Index {i}] {display_name}\n\n"
        
        prompt += f"Your response must be a list of indices in exact list format (e.g., [4, 2, 1, 0, 3]), ranking every index from 0 to {len(self.doc_types)-1} by most relevant document type index to least relevant document type index."
        
        return prompt
    
    def create_rft_training_sample(self, company: str, category: str, question: str, 
                                 document_ranking: List[int], score_counts: Dict[str, Dict[str, int]],
                                 document_content: Dict[str, str]) -> Dict:
        """Create a single RFT training sample with ranking task"""
        
        # Create the ranking prompt with full content
        prompt_content = self.create_ranking_prompt(question, document_content)
        
        # Create RFT training sample
        rft_sample = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            "document_ranking": document_ranking,  # Ground truth ranking
            "document_score_counts": score_counts,  # Detailed score breakdown
            "metadata": {
                "company": company,
                "category": category,
                "question": question,
                "document_types": self.doc_types,
                "task_type": "document_ranking"
            }
        }
        
        return rft_sample
    
    def process_company(self, company: str) -> List[Dict]:
        """Process all questions for a single company"""
        logger.info(f"Processing company: {company.upper()}")
        
        # Load questions from CSV
        questions = self.load_company_questions(company)
        
        if not questions:
            logger.warning(f"No questions found for company {company}")
            return []
            
        rft_samples = []
        
        for category, question, question_index in questions:
            logger.debug(f"Processing {category} - Q{question_index}: {question}")
            
            # Get document chunk scores and full content for this question
            document_chunk_scores, document_content = self.get_document_scores_for_question(
                company, category, question_index
            )
            
            if not document_chunk_scores:
                logger.warning(f"No document data found for {company} - {category} - Q{question_index}")
                continue
            
            # Rank documents by relevance
            document_ranking, score_counts = self.rank_documents_by_relevance(document_chunk_scores)
            
            # Create RFT training sample
            rft_sample = self.create_rft_training_sample(
                company, category, question, document_ranking, score_counts, document_content
            )
            
            rft_samples.append(rft_sample)
            self.stats["total_samples"] += 1
            
        logger.info(f"Generated {len(rft_samples)} RFT samples for {company.upper()}")
        return rft_samples
    
    def split_train_eval(self, samples: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Split samples into train and eval sets"""
        if not samples:
            return [], []
            
        # Shuffle samples for random split
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # Calculate split point
        split_point = int(len(shuffled) * train_ratio)
        
        train_samples = shuffled[:split_point]
        eval_samples = shuffled[split_point:]
        
        return train_samples, eval_samples
    
    def save_rft_samples(self, rft_samples: List[Dict], output_file: str):
        """Save RFT samples to JSONL file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in rft_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
        logger.info(f"Saved {len(rft_samples)} RFT samples to {output_path}")
    
    def save_stats(self, stats_dict: Dict, output_file: str):
        """Save statistics to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)
                
        logger.info(f"Saved statistics to {output_path}")
    
    def print_statistics(self):
        """Print processing statistics"""
        logger.info("=== Processing Statistics ===")
        for key, value in sorted(self.stats.items()):
            logger.info(f"{key}: {value}")
            
    def create_sample_analysis(self, rft_samples: List[Dict], num_samples: int = 3):
        """Print analysis of sample RFT data"""
        logger.info(f"=== Sample Analysis (showing {num_samples} samples) ===")
        
        for i, sample in enumerate(rft_samples[:num_samples]):
            logger.info(f"\n--- Sample {i+1} ---")
            logger.info(f"Company: {sample['metadata']['company']}")
            logger.info(f"Category: {sample['metadata']['category']}")
            logger.info(f"Question: {sample['metadata']['question']}")
            logger.info(f"Document Ranking: {sample['document_ranking']}")
            
            # Show ranking explanation
            doc_names = [self.doc_type_display[doc_type] for doc_type in self.doc_types]
            ranked_docs = [doc_names[i] for i in sample['document_ranking']]
            logger.info(f"Ranking Order: {' > '.join(ranked_docs)}")
            
            # Show score distribution
            score_counts = sample['document_score_counts']
            non_empty_docs = [doc for doc, counts in score_counts.items() 
                            if sum(counts.values()) > 0]
            logger.info(f"Documents with chunks: {len(non_empty_docs)}/{len(self.doc_types)}")
            
            # Show content lengths
            content_lengths = {}
            for j, doc_type in enumerate(self.doc_types):
                # Extract content length from the prompt
                prompt = sample['messages'][0]['content']
                doc_sections = prompt.split(f"{j}. {self.doc_type_display[doc_type]}: ")
                if len(doc_sections) > 1:
                    doc_content = doc_sections[1].split(f"\n\n{j+1}.")[0] if j < len(self.doc_types)-1 else doc_sections[1].split("\n\nYour response")[0]
                    content_lengths[doc_type] = len(doc_content)
                    
            logger.info(f"Document content lengths: {content_lengths}")

def main():
    """Main function - process all companies and create train/eval splits"""
    
    # Configuration
    OUTPUT_BASE_DIR = 'output'
    TRAIN_RATIO = 0.8  # 80% train, 20% eval
    
    # Initialize preprocessor
    preprocessor = DocumentLevelRFTPreprocessor()
    
    # Process each company
    for company in preprocessor.companies:
        logger.info(f"Starting RFT preprocessing for {company.upper()}")
        
        # Process company samples
        rft_samples = preprocessor.process_company(company)
        
        if not rft_samples:
            logger.error(f"❌ No samples generated for {company.upper()}")
            continue
        
        # Split into train/eval
        train_samples, eval_samples = preprocessor.split_train_eval(rft_samples, TRAIN_RATIO)
        
        # Create output directory structure
        output_dir = Path(OUTPUT_BASE_DIR) / "raw_data" / company / "document-ranking"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train/eval files
        train_file = output_dir / "train.jsonl"
        eval_file = output_dir / "eval.jsonl"
        stats_file = output_dir / "stats.json"
        
        preprocessor.save_rft_samples(train_samples, str(train_file))
        preprocessor.save_rft_samples(eval_samples, str(eval_file))
        
        # Save statistics
        company_stats = {
            "company": company,
            "total_samples": len(rft_samples),
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples),
            "train_ratio": TRAIN_RATIO,
            "task_type": "document_ranking",
            "document_types": preprocessor.doc_types,
            "processing_stats": dict(preprocessor.stats)
        }
        
        preprocessor.save_stats(company_stats, str(stats_file))
        
        # Print analysis for this company
        logger.info(f"\n🔍 Analysis for {company.upper()}")
        logger.info(f"Total samples: {len(rft_samples)}")
        logger.info(f"Train samples: {len(train_samples)}")
        logger.info(f"Eval samples: {len(eval_samples)}")
        
        preprocessor.create_sample_analysis(train_samples, 2)
        
        logger.info(f"✅ Successfully processed {company.upper()}")
        logger.info(f"Files saved to: {output_dir}")
        
        # Reset stats for next company
        preprocessor.stats = defaultdict(int)
    
    logger.info("🎉 All companies processed successfully!")

if __name__ == "__main__":
    main()