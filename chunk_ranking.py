import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional
import random
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkLevelRFTPreprocessor:
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
            
        # Try multiple encodings to handle special characters
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
    
    # Chunking functions - NO FILTERING, KEEP ALL CHUNKS
    def chunk_10k_10q_html(self, text: str) -> Dict[int, str]:
        """10-K, 10-Q HTML 파일을 p 태그와 table 태그 단위로 청킹 - KEEP ALL CHUNKS"""
        soup = BeautifulSoup(text, 'html.parser')
        chunks = {}
        
        elements = soup.find_all(['p', 'table'])
        
        for idx, element in enumerate(elements):
            # Keep ALL chunks, even if empty or short
            if element.name == 'p':
                chunks[idx] = element.text.strip() if element.text else ""
            elif element.name == 'table':
                chunks[idx] = element.text.strip() if element.text else ""
        
        return chunks
    
    def chunk_earnings_html(self, text: str) -> Dict[int, str]:
        """Earnings HTML 파일을 발화자와 발화 내용으로 청킹 - KEEP ALL CHUNKS"""
        soup = BeautifulSoup(text, 'html.parser')
        chunks = {}
        
        elements = soup.find_all(['strong', 'p'])
        
        for idx, element in enumerate(elements):
            if element.name == 'strong':
                speaker_text = element.text.strip() if element.text else ""
                
                if ' - ' in speaker_text:
                    speaker_name = speaker_text.split(' - ')[0].strip()
                    speaker_role = speaker_text.split(' - ')[1].strip()
                    formatted_speaker = f"{speaker_name} - {speaker_role}"
                else:
                    formatted_speaker = speaker_text
                    
                chunks[idx] = formatted_speaker
            
            elif element.name == 'p':
                content = element.text.strip() if element.text else ""
                chunks[idx] = content
        
        return chunks
    
    def chunk_8k_json(self, text: str) -> Dict[int, str]:
        """8-K JSON 파일을 content 기준으로 청킹 - KEEP ALL CHUNKS"""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse 8-K JSON")
            return {}
            
        chunks = {}
        idx = 0
        
        for item in data:
            if "content" in item:
                chunks[idx] = item["content"].strip() if item["content"] else ""
                idx += 1
            else:
                # Keep empty chunks to maintain index alignment
                chunks[idx] = ""
                idx += 1
                
        return chunks
    
    def chunk_def14a_json(self, text: str) -> Dict[int, str]:
        """DEF14A JSON 파일을 각 항목 기준으로 청킹 - KEEP ALL CHUNKS"""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse DEF14A JSON")
            return {}
            
        chunks = {}
        idx = 0
        
        for item in data:
            if isinstance(item, dict):
                if "content" in item:
                    chunks[idx] = item["content"].strip() if item["content"] else ""
                else:
                    chunks[idx] = ""
                idx += 1
            else:
                # Handle non-dict items
                chunks[idx] = str(item) if item else ""
                idx += 1
        
        return chunks
    
    def get_chunk(self, text: str, file_type: str) -> Dict[int, str]:
        """파일 타입에 따라 적절한 chunking 메서드를 호출 - KEEP ALL CHUNKS"""
        if file_type == "10-K" or file_type == "10-Q":
            return self.chunk_10k_10q_html(text)
        elif file_type == "8-K":
            return self.chunk_8k_json(text)
        elif file_type == "DEF14A":
            return self.chunk_def14a_json(text)
        elif file_type == "Earnings":
            return self.chunk_earnings_html(text)
        else:
            # 기본적으로는 줄바꿈 기준으로 chunking - KEEP ALL LINES
            chunks = {}
            for idx, line in enumerate(text.split("\n")):
                chunks[idx] = line.strip()  # Keep even empty lines
            return chunks
    
    def load_existing_annotations(self, company: str, category: str, question_index: int, doc_type: str) -> Dict[int, int]:
        """Load existing chunk annotations for a specific question and document"""
        category_dir = self.csv_category_to_directory_name(category)
        qa_path = self.data_root / company / "qa" / category_dir
        
        # Try double annotated pattern first (AAPL format)
        jsonl_filename = f"relevance_results_{doc_type}_filter_q{question_index}_annotated_annotated.jsonl"
        jsonl_path = qa_path / jsonl_filename
        
        # If double annotated doesn't exist, try single annotated pattern
        if not jsonl_path.exists():
            jsonl_filename = f"relevance_results_{doc_type}_filter_q{question_index}_annotateds.jsonl"
            jsonl_path = qa_path / jsonl_filename
        
        chunk_scores = {}
        
        if not jsonl_path.exists():
            return chunk_scores
            
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            chunk_idx = data.get('index')
                            
                            # Priority: relevance_score first, then score as fallback
                            if 'score' in data and data['score'] is not None:
                                chunk_scores[chunk_idx] = data['score']
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line in {jsonl_path}: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error reading JSONL file {jsonl_path}: {e}")
            
        return chunk_scores
    
    def create_complete_chunk_ranking_data(self, company: str, category: str, question: str, 
                                         question_index: int, doc_type: str) -> Optional[Dict]:
        """Create complete chunk ranking data including ALL chunks"""
        
        # Load the original document
        if doc_type == "def14a":
            file_path = self.data_root / company / "def14a.json"
        elif doc_type == "10k":
            file_path = self.data_root / company / "10-k.html"
        elif doc_type == "10q":
            file_path = self.data_root / company / "10-q.html"
        elif doc_type == "8k":
            file_path = self.data_root / company / "8-k.json"
        elif doc_type == "earnings":
            file_path = self.data_root / company / "earnings.html"
        else:
            logger.warning(f"Unknown document type: {doc_type}")
            return None
            
        if not file_path.exists():
            logger.warning(f"Document file not found: {file_path}")
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            logger.error(f"Error reading document file {file_path}: {e}")
            return None
        
        # Generate complete chunks using the same logic as run_annotation.py
        if doc_type == "10k":
            all_chunks = self.get_chunk(file_content, "10-K")
        elif doc_type == "10q":
            all_chunks = self.get_chunk(file_content, "10-Q")
        elif doc_type == "8k":
            all_chunks = self.get_chunk(file_content, "8-K")
        elif doc_type == "def14a":
            all_chunks = self.get_chunk(file_content, "DEF14A")
        elif doc_type == "earnings":
            all_chunks = self.get_chunk(file_content, "Earnings")
        else:
            all_chunks = self.get_chunk(file_content, doc_type)
        
        if not all_chunks:
            logger.warning(f"No chunks generated for {company} {doc_type}")
            return None
            
        # Load existing annotations
        existing_scores = self.load_existing_annotations(company, category, question_index, doc_type)
        
        # Create complete chunk scores - PRESERVE ALL CHUNKS AND INDICES
        chunk_texts = []
        chunk_scores = []
        
        for chunk_idx in sorted(all_chunks.keys()):
            chunk_text = all_chunks[chunk_idx]
            chunk_score = existing_scores.get(chunk_idx, 0)  # Default to 0 if not annotated
            
            # NO FILTERING - KEEP ALL CHUNKS
            chunk_texts.append(chunk_text)
            chunk_scores.append(chunk_score)
        
        # Track statistics
        self.stats[f"{doc_type}_total_chunks"] += len(chunk_texts)
        self.stats[f"{doc_type}_annotated_chunks"] += len([s for s in chunk_scores if s > 0])
        self.stats[f"{doc_type}_zero_score_chunks"] += len([s for s in chunk_scores if s == 0])
        
        return {
            "company": company,
            "category": category,
            "question": question,
            "doc_type": doc_type,
            "chunks": chunk_texts,
            "chunk_scores": chunk_scores
        }
    
    def create_chunk_prompt_top_k(self, question: str, chunks: List[str], k: int = 10) -> str:
        """Ask model to select and rank only top-k most relevant chunks"""
        # Use k if chunks length > 10, else use chunks length
        actual_k = k if len(chunks) > 10 else len(chunks)
        
        prompt = f"""Identify the {actual_k} most relevant text chunks for answering this question, then rank them in order of relevance (best first).
Question: {question}
Text chunks:
"""
        for i, chunk in enumerate(chunks):
            prompt += f"[Chunk Index {i}] {chunk}\n"
        prompt += f"""
Task: Select and rank the {actual_k} most relevant chunks among the given text chunks(from index 0 to {len(chunks)-1}).
- Put the BEST chunk first
- Put the 2nd best chunk second  
- Continue until you have ranked your top {actual_k} chunks
Response Format: [1st_most_relevant_index, 2nd_most_relevant_index, ..., {actual_k}th_most_relevant_index]"""
   
        return prompt
    
    def create_chunk_ranking_rft_sample(self, chunk_data: Dict) -> Dict:
        """Create a single chunk ranking RFT training sample"""
        
        # Create instructional prompt with ALL chunks
        prompt_content = self.create_chunk_prompt_top_k(
            chunk_data["question"], 
            chunk_data["chunks"],
            k=10
        )
        
        # Create RFT training sample
        rft_sample = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            "chunk_scores": chunk_data["chunk_scores"],
            "metadata": {
                "company": chunk_data["company"],
                "category": chunk_data["category"],
                "question": chunk_data["question"],
                "doc_type": chunk_data["doc_type"],
                "total_chunks": len(chunk_data["chunks"]),
                "annotated_chunks": len([s for s in chunk_data["chunk_scores"] if s > 0]),
                "zero_score_chunks": len([s for s in chunk_data["chunk_scores"] if s == 0]),
                "task_type": "chunk_ranking"
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
            logger.debug(f"Processing {category} - Q{question_index}: {question[:50]}...")
            
            # Process each document type for this question
            for doc_type in self.doc_types:
                chunk_data = self.create_complete_chunk_ranking_data(
                    company, category, question, question_index, doc_type
                )
                
                if not chunk_data:
                    logger.debug(f"No chunk data for {company} - {category} - Q{question_index} - {doc_type}")
                    continue
                
                # Skip if no chunks at all
                if len(chunk_data["chunks"]) == 0:
                    continue
                
                # Create RFT training sample
                rft_sample = self.create_chunk_ranking_rft_sample(chunk_data)
                rft_samples.append(rft_sample)
                self.stats["total_samples"] += 1
                
        logger.info(f"Generated {len(rft_samples)} chunk ranking RFT samples for {company.upper()}")
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
        logger.info("=== Chunk Ranking Processing Statistics ===")
        for key, value in sorted(self.stats.items()):
            logger.info(f"{key}: {value}")
            
    def create_sample_analysis(self, rft_samples: List[Dict], num_samples: int = 2):
        """Print analysis of sample RFT data"""
        logger.info(f"=== Chunk Ranking Sample Analysis (showing {num_samples} samples) ===")
        
        for i, sample in enumerate(rft_samples[:num_samples]):
            logger.info(f"\n--- Sample {i+1} ---")
            logger.info(f"Company: {sample['metadata']['company']}")
            logger.info(f"Category: {sample['metadata']['category']}")
            logger.info(f"Document Type: {sample['metadata']['doc_type']}")
            logger.info(f"Question: {sample['metadata']['question'][:100]}...")
            logger.info(f"Total Chunks: {sample['metadata']['total_chunks']}")
            logger.info(f"Annotated Chunks: {sample['metadata']['annotated_chunks']}")
            logger.info(f"Zero Score Chunks: {sample['metadata']['zero_score_chunks']}")
            
            # Show score distribution
            scores = sample['chunk_scores']
            score_dist = {0: 0, 1: 0, 2: 0}
            for score in scores:
                if score in score_dist:
                    score_dist[score] += 1
            
            logger.info(f"Score Distribution: Score 0: {score_dist[0]}, Score 1: {score_dist[1]}, Score 2: {score_dist[2]}")
            
            # Show first few chunks to verify no gaps
            logger.info("First 3 chunks indices and content preview:")
            chunks_preview = sample['messages'][0]['content'].split('\n')[3:6]  # Skip question part
            for j, chunk_line in enumerate(chunks_preview):
                if chunk_line.strip():
                    logger.info(f"  {chunk_line[:100]}...")

def main():
    """Main function - process all companies and create chunk ranking train/eval splits"""
    
    # Configuration
    OUTPUT_BASE_DIR = "output/raw_data"
    TRAIN_RATIO = 0.8  # 80% train, 20% eval
    
    # Initialize preprocessor
    preprocessor = ChunkLevelRFTPreprocessor()
    
    # Process each company
    for company in preprocessor.companies:
        logger.info(f"Starting Chunk Ranking RFT preprocessing for {company.upper()}")
        
        # Process company samples
        rft_samples = preprocessor.process_company(company)
        
        if not rft_samples:
            logger.error(f"❌ No chunk ranking samples generated for {company.upper()}")
            continue
        
        # Split into train/eval
        train_samples, eval_samples = preprocessor.split_train_eval(rft_samples, TRAIN_RATIO)
        
        # Create output directory structure
        output_dir = Path(OUTPUT_BASE_DIR) / company / "chunk-ranking"
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
            "document_types": preprocessor.doc_types,
            "processing_stats": dict(preprocessor.stats)
        }
        
        preprocessor.save_stats(company_stats, str(stats_file))
        
        # Print analysis for this company
        logger.info(f"\n🔍 Chunk Ranking Analysis for {company.upper()}")
        logger.info(f"Total samples: {len(rft_samples)}")
        logger.info(f"Train samples: {len(train_samples)}")
        logger.info(f"Eval samples: {len(eval_samples)}")
        
        preprocessor.create_sample_analysis(train_samples, 2)
        
        logger.info(f"✅ Successfully processed chunk ranking for {company.upper()}")
        logger.info(f"Files saved to: {output_dir}")
        
        # Reset stats for next company
        preprocessor.stats = defaultdict(int)
    
    logger.info("🎉 All companies processed for chunk ranking successfully!")

if __name__ == "__main__":
    main()