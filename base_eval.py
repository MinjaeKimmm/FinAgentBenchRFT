import json
import math
import re
import asyncio
import os
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file")
except ImportError:
    print("Using system environment variables")
# API clients
try:
    import openai
    from anthropic import Anthropic
except ImportError:
    print("Install: pip install openai anthropic python-dotenv pandas")
    exit(1)

@dataclass
class EvaluationResult:
    model_name: str
    task_type: str
    total_samples: int
    successful_parses: int
    ndcg_scores: List[float]
    map_scores: List[float]
    mrr_scores: List[float]
    avg_ndcg: float
    avg_map: float
    avg_mrr: float
    parse_success_rate: float

class BaselineEvaluator:
    def __init__(self, rft_data_dir: str = "output/rft_data", max_retries: int = 1):
        self.rft_data_dir = Path(rft_data_dir)
        self.results_dir = self.rft_data_dir / "baseline_results"
        self.logs_dir = self.results_dir / "logs"
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.max_retries = max_retries
        
        # Initialize API clients
        self.openai_client = None
        self.anthropic_client = None
        self._setup_api_clients()
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup detailed logging for debugging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create main logger
        self.logger = logging.getLogger('baseline_evaluator')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler for detailed logs
        log_file = self.logs_dir / f"evaluation_log_{timestamp}.txt"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Detailed logs: {log_file}")
        
    def _setup_api_clients(self):
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.openai_client = openai.OpenAI(
                    api_key=openai_key,
                    timeout=10000.0 
                )
                print("OpenAI client initialized")
            else:
                print("OPENAI_API_KEY not found")
        except Exception as e:
            print(f"OpenAI setup failed: {e}")
            
        try:
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                self.anthropic_client = Anthropic(
                    api_key=anthropic_key,
                    timeout=10000.0
                )
                print("Anthropic client initialized")
            else:
                print("ANTHROPIC_API_KEY not found")
        except Exception as e:
            print(f"Anthropic setup failed: {e}")
    
    async def evaluate_all_baselines(self, num_samples: int = 5, tasks: List[str] = None):
        """
        Evaluate baseline models on specified tasks
        
        Args:
            num_samples: Number of samples per task
            tasks: List of tasks to evaluate. Options: ["document_ranking", "chunk_ranking"]
                  If None, evaluates both tasks
        """
        if tasks is None:
            tasks = ["document_ranking", "chunk_ranking"]
        
        # Validate tasks
        valid_tasks = ["document_ranking", "chunk_ranking"]
        invalid_tasks = [task for task in tasks if task not in valid_tasks]
        if invalid_tasks:
            print(f"Invalid tasks: {invalid_tasks}. Valid tasks: {valid_tasks}")
            return
        
        print(f"Starting baseline evaluation ({num_samples} samples per task)")
        print(f"Tasks to evaluate: {tasks}")
        
        # Your specific models with reasoning capability
        models = [
            {"name": "o4-mini", "client": "openai", "reasoning": True},
        ]
        
        all_results = []
        
        for task in tasks:
            print(f"\nEvaluating {task}")
            self.logger.info(f"Starting task: {task}")
            
            eval_data = self._load_evaluation_data(task, num_samples)
            if not eval_data:
                print(f"No evaluation data found for {task}")
                self.logger.warning(f"No evaluation data found for {task}")
                continue
                
            print(f"Loaded {len(eval_data)} samples")
            self.logger.info(f"Loaded {len(eval_data)} samples for {task}")
            
            for model_config in models:
                model_name = model_config["name"]
                client_type = model_config["client"]
                
                # Check client availability
                if client_type == "openai" and not self.openai_client:
                    print(f"Skipping {model_name} - OpenAI client not available")
                    self.logger.warning(f"Skipping {model_name} - OpenAI client not available")
                    continue
                elif client_type == "anthropic" and not self.anthropic_client:
                    print(f"Skipping {model_name} - Anthropic client not available")
                    self.logger.warning(f"Skipping {model_name} - Anthropic client not available")
                    continue
                
                print(f"Testing {model_name}")
                self.logger.info(f"Starting evaluation: {model_name} on {task}")
                
                try:
                    result = await self._evaluate_model(model_config, task, eval_data)
                    all_results.append(result)
                    
                    print(f"{model_name} results: nDCG={result.avg_ndcg:.3f}, "
                          f"MAP={result.avg_map:.3f}, MRR={result.avg_mrr:.3f}, "
                          f"Parse={result.parse_success_rate:.1%}")
                    
                except Exception as e:
                    print(f"{model_name} failed: {e}")
                    self.logger.error(f"{model_name} failed: {e}")
                    continue
        
        if all_results:
            self._save_results(all_results)
            self._create_comparison_report(all_results)
            print("Baseline evaluation complete")
            self.logger.info("Baseline evaluation complete")
        else:
            print("No results to save")
            self.logger.warning("No results to save")
    
    def _load_evaluation_data(self, task_type: str, num_samples: int) -> List[Dict]:
        eval_file = self.rft_data_dir / "combined_datasets" / f"{task_type}_eval.jsonl"
    
        if eval_file.exists():
            print(f"Loading from combined dataset: {eval_file}")
            with open(eval_file, 'r', encoding='utf-8') as f:
                samples = []
                for line in f:
                    if line.strip():
                        try:
                            sample = json.loads(line.strip())
                            samples.append(sample)
                        except json.JSONDecodeError:
                            continue
                print(f"Loaded {len(samples)} samples from combined dataset")
                return samples[:num_samples]
        
        raise FileNotFoundError(f"Evaluation data not found for {task_type}")
    
    async def _evaluate_model(self, model_config: Dict, task_type: str, 
                            eval_data: List[Dict]) -> EvaluationResult:
        
        model_name = model_config["name"]
        is_reasoning = model_config.get("reasoning", False)
        
        self.logger.info(f"Starting evaluation for {model_name} on {task_type}")
        self.logger.info(f"Model config: {model_config}")
        self.logger.info(f"Total samples: {len(eval_data)}")
        
        # Create detailed log file for this model/task combo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_log_file = self.logs_dir / f"{model_name}_{task_type}_{timestamp}.jsonl"
        
        ndcg_scores = []
        map_scores = []
        mrr_scores = []
        successful_parses = 0
        
        for i, sample in enumerate(eval_data):
            if i % 5 == 0:
                print(f"Progress: {i}/{len(eval_data)}")
            
            sample_log = {
                "sample_index": i,
                "model_name": model_name,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat(),
                "sample_id": sample.get("id", f"sample_{i}"),
            }
            
            try:
                # Log original prompt details
                original_prompt = sample["messages"][0]["content"]
                sample_log["original_prompt_length"] = len(original_prompt)
                sample_log["original_prompt_preview"] = original_prompt[:500] + "..." if len(original_prompt) > 500 else original_prompt
                sample_log["num_items_to_rank"] = len(sample["qrel"])
                sample_log["qrel"] = sample["qrel"]
                
                self.logger.debug(f"Sample {i}: Prompt length={len(original_prompt)}, Items to rank={len(sample['qrel'])}")
                
                # Get ranking with retry logic
                predicted_ranking = await self._get_ranking_with_retry(
                    model_config, original_prompt, len(sample["qrel"]), sample_log, is_reasoning
                )
                
                sample_log["final_predicted_ranking"] = predicted_ranking
                
                if not predicted_ranking:
                    sample_log["parsing_success"] = False
                    sample_log["ndcg"] = 0.0
                    sample_log["map"] = 0.0
                    sample_log["mrr"] = 0.0
                    
                    ndcg_scores.append(0.0)
                    map_scores.append(0.0)
                    mrr_scores.append(0.0)
                    
                    self.logger.warning(f"Sample {i}: Failed to parse ranking after all retries")
                    continue
                
                successful_parses += 1
                sample_log["parsing_success"] = True
                
                # Calculate metrics
                qrel = sample["qrel"]
                k = 5 if task_type == "document_ranking" else 10
                
                ndcg = self._calculate_ndcg_at_k(qrel, predicted_ranking, k)
                map_score = self._calculate_map_at_k(qrel, predicted_ranking, k)
                mrr = self._calculate_mrr_at_k(qrel, predicted_ranking, k)
                
                sample_log["ndcg"] = ndcg
                sample_log["map"] = map_score
                sample_log["mrr"] = mrr
                sample_log["k_value"] = k
                
                ndcg_scores.append(ndcg)
                map_scores.append(map_score)
                mrr_scores.append(mrr)
                
                self.logger.debug(f"Sample {i}: nDCG={ndcg:.3f}, MAP={map_score:.3f}, MRR={mrr:.3f}")
                
                # Rate limiting
                await asyncio.sleep(1.0)  # Increased wait time
                
            except Exception as e:
                sample_log["error"] = str(e)
                sample_log["parsing_success"] = False
                sample_log["ndcg"] = 0.0
                sample_log["map"] = 0.0
                sample_log["mrr"] = 0.0
                
                self.logger.error(f"Sample {i}: Error - {e}")
                
                ndcg_scores.append(0.0)
                map_scores.append(0.0)
                mrr_scores.append(0.0)
                continue
            
            finally:
                # Write sample log to detailed file
                with open(detailed_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(sample_log, ensure_ascii=False) + '\n')
        
        # Calculate averages
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
        parse_success_rate = successful_parses / len(eval_data)
        
        self.logger.info(f"{model_name} completed: nDCG={avg_ndcg:.3f}, MAP={avg_map:.3f}, MRR={avg_mrr:.3f}, Parse={parse_success_rate:.1%}")
        self.logger.info(f"Detailed logs saved to: {detailed_log_file}")
        
        return EvaluationResult(
            model_name=model_name,
            task_type=task_type,
            total_samples=len(eval_data),
            successful_parses=successful_parses,
            ndcg_scores=ndcg_scores,
            map_scores=map_scores,
            mrr_scores=mrr_scores,
            avg_ndcg=avg_ndcg,
            avg_map=avg_map,
            avg_mrr=avg_mrr,
            parse_success_rate=parse_success_rate
        )
    
    async def _get_ranking_with_retry(self, model_config: Dict, original_prompt: str, 
                                    num_items: int, sample_log: Dict, is_reasoning: bool, 
                                    max_retries: int = None) -> List[int]:
        """
        Get ranking with retry logic - tries up to max_retries times until successful parsing
        """
        if max_retries is None:
            max_retries = self.max_retries
        
        sample_log["ranking_retry_attempts"] = []
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Ranking attempt {attempt + 1}/{max_retries}")
                
                attempt_log = {
                    "attempt": attempt + 1,
                    "timestamp": datetime.now().isoformat()
                }
                
                if is_reasoning:
                    # Two-step approach for reasoning models
                    predicted_ranking = await self._get_reasoning_model_ranking(
                        model_config, original_prompt, num_items, attempt_log
                    )
                else:
                    # Direct approach for regular models
                    response = await self._get_model_response(model_config, original_prompt)
                    attempt_log["direct_response"] = response
                    attempt_log["direct_response_length"] = len(response)
                    
                    predicted_ranking = self._parse_ranking_from_response(response, num_items)
                    attempt_log["direct_parsed_ranking"] = predicted_ranking
                    attempt_log["final_ranking_source"] = "direct_parsing"
                
                attempt_log["predicted_ranking"] = predicted_ranking
                attempt_log["parsing_success"] = bool(predicted_ranking)
                
                sample_log["ranking_retry_attempts"].append(attempt_log)
                
                if predicted_ranking:
                    self.logger.debug(f"Successfully parsed ranking on attempt {attempt + 1}")
                    sample_log["successful_attempt"] = attempt + 1
                    sample_log["total_attempts"] = attempt + 1
                    return predicted_ranking
                else:
                    self.logger.debug(f"Failed to parse ranking on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2.0)  # Wait before retry
                    
            except Exception as e:
                attempt_log = {
                    "attempt": attempt + 1,
                    "error": str(e),
                    "parsing_success": False,
                    "timestamp": datetime.now().isoformat()
                }
                sample_log["ranking_retry_attempts"].append(attempt_log)
                self.logger.error(f"Ranking attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2.0)  # Wait before retry
        
        # All attempts failed
        sample_log["successful_attempt"] = None
        sample_log["total_attempts"] = max_retries
        self.logger.error(f"All {max_retries} ranking attempts failed")
        return []
    
    async def _get_reasoning_model_ranking(self, model_config: Dict, 
                                     original_prompt: str, num_items: int, sample_log: Dict) -> List[int]:
        """Two-step approach for reasoning models with retry logic"""
        
        try:
            # Step 1: Get reasoning model response with retry logic
            step1_response = await self._get_model_response_with_retry(model_config, original_prompt, sample_log)
            
            if not step1_response or len(step1_response.strip()) == 0:
                sample_log["step1_empty_response"] = True
                sample_log["step1_final_response"] = step1_response
                self.logger.warning("Step 1 returned empty response after retries")
                return []
            
            sample_log["step1_raw_response"] = step1_response
            sample_log["step1_response_length"] = len(step1_response)
            sample_log["step1_empty_response"] = False
            
            self.logger.debug(f"Step 1 response length: {len(step1_response)}")
            
            # Step 2: Extract ranking using GPT-4.1-mini with IMPROVED prompt
            extraction_prompt = f"""Extract the exact ranking from this response and return ONLY the JSON object.
    Do not use markdown formatting or code blocks.

    Original response:
    {step1_response}

    IMPORTANT: 
    - Use the EXACT same numbers from the original response
    - Do NOT change or limit the range of numbers
    - The indices can be any valid chunk numbers (0 to {num_items-1})

    Return ONLY this JSON format (no markdown, no code blocks):
    {{"ranking": [actual_indices_from_response]}}

    Example: If the response contains [157, 64, 23, 8, 145], return {{"ranking": [157, 64, 23, 8, 145]}}"""
            
            sample_log["extraction_prompt"] = extraction_prompt
            
            try:
                extraction_response = await self._get_openai_extraction(extraction_prompt)
                sample_log["step2_extraction_response"] = extraction_response
                
                # Clean the response - remove markdown if present
                cleaned_response = self._clean_json_response(extraction_response)
                sample_log["step2_cleaned_response"] = cleaned_response
                
                # Try to parse JSON
                try:
                    extracted_data = json.loads(cleaned_response)
                    ranking = extracted_data.get("ranking", [])
                    sample_log["step2_extracted_ranking"] = ranking
                    sample_log["step2_extraction_success"] = True
                    
                    # Validate ranking
                    if (isinstance(ranking, list) and 
                        len(ranking) <= num_items and
                        all(isinstance(x, int) and 0 <= x < num_items for x in ranking) and
                        len(set(ranking)) == len(ranking)):
                        sample_log["final_ranking_source"] = "step2_extraction"
                        return ranking
                    else:
                        sample_log["step2_validation_failed"] = True
                        sample_log["step2_validation_error"] = "Invalid ranking format or values"
                        raise ValueError("Invalid ranking from extraction")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    sample_log["step2_json_error"] = str(e)
                    sample_log["step2_extraction_success"] = False
                    raise e
                    
            except Exception as e:
                sample_log["step2_extraction_error"] = str(e)
                sample_log["step2_extraction_success"] = False
                
                # Fallback to regex parsing
                self.logger.debug("Step 2 failed, trying regex fallback")
                predicted_ranking = self._parse_ranking_from_response(step1_response, num_items)
                sample_log["fallback_regex_ranking"] = predicted_ranking
                sample_log["final_ranking_source"] = "regex_fallback"
                return predicted_ranking
            
        except Exception as e:
            sample_log["reasoning_model_error"] = str(e)
            self.logger.error(f"Reasoning model parsing failed: {e}")
            return []
    
    async def _get_model_response_with_retry(self, model_config: Dict, prompt: str, 
                                           sample_log: Dict, max_retries: int = None) -> str:
        """Get model response with retry logic for empty responses"""
        
        if max_retries is None:
            max_retries = self.max_retries
        
        client_type = model_config["client"]
        model_name = model_config["name"]
        
        sample_log["retry_attempts"] = []
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Attempt {attempt + 1}/{max_retries} for {model_name}")
                
                response = await self._get_model_response(model_config, prompt)
                
                attempt_log = {
                    "attempt": attempt + 1,
                    "response_length": len(response) if response else 0,
                    "response_preview": response[:200] if response else "EMPTY",
                    "success": len(response.strip()) > 0 if response else False
                }
                sample_log["retry_attempts"].append(attempt_log)
                
                if response and len(response.strip()) > 0:
                    self.logger.debug(f"Successful response on attempt {attempt + 1}")
                    return response
                else:
                    self.logger.warning(f"Empty response on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2.0)  # Wait before retry
                    
            except Exception as e:
                attempt_log = {
                    "attempt": attempt + 1,
                    "error": str(e),
                    "success": False
                }
                sample_log["retry_attempts"].append(attempt_log)
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2.0)  # Wait before retry
        
        # All attempts failed
        self.logger.error(f"All {max_retries} attempts failed for {model_name}")
        return ""
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response by removing markdown formatting"""
        
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Find JSON object in response
        json_match = re.search(r'\{[^}]*"ranking"[^}]*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return response.strip()
    
    async def _get_openai_extraction(self, extraction_prompt: str) -> str:
        """Use GPT-4.1-mini for extraction step with higher token limit"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=500,  
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Extraction step failed: {e}")
    
    async def _get_model_response(self, model_config: Dict, prompt: str) -> str:
        client_type = model_config["client"]
        model_name = model_config["name"]
        
        if client_type == "openai":
            return await self._get_openai_response(model_name, prompt)
        elif client_type == "anthropic":
            return await self._get_anthropic_response(model_name, prompt)
        else:
            raise ValueError(f"Unknown client type: {client_type}")
    
    async def _get_openai_response(self, model_name: str, prompt: str) -> str:
        try:
            if model_name == "o3" or "o4-mini":
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=32000,  # Increased token limit
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=32000,  # Increased token limit
                )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    async def _get_anthropic_response(self, model_name: str, prompt: str) -> str:
        try:
            response = self.anthropic_client.messages.create(
                model=model_name,
                max_tokens=32000,
                thinking={"type": "enabled", "budget_tokens": 16000},
                messages=[{"role": "user", "content": prompt}],
            )
            visible_answer = "".join(
                block.text
                for block in response.content
                if getattr(block, "type", "") == "text"         # skip ThinkingBlock / RedactedThinkingBlock
            )
            # (optional) capture Claude's reasoning for your logs
            thinking_blocks = [b for b in response.content if getattr(b, "type", "") == "thinking"]
            if thinking_blocks:
                sample_log = {}                      # or pass one in
                sample_log["claude_thinking"] = thinking_blocks[0].thinking
            return visible_answer
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")
    
    def _parse_ranking_from_response(self, response: str, num_items: int) -> List[int]:
        """Parse ranking from model response using multiple patterns"""
        
        patterns = [
            r'\[([0-9,\s]+)\]',                      # [0, 1, 2, 3, 4]
            r'(\d+(?:,\s*\d+)*)',                    # 0, 1, 2, 3, 4
            r'(?:ranking|order):\s*\[([0-9,\s]+)\]', # ranking: [0, 1, 2, 3, 4]
            r'(?:ranking|order):\s*(\d+(?:,\s*\d+)*)', # ranking: 0, 1, 2, 3, 4
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                try:
                    numbers = [int(x.strip()) for x in match.split(',') if x.strip()]
                    
                    # Validate ranking
                    if (len(numbers) <= num_items and 
                        all(0 <= n < num_items for n in numbers) and 
                        len(set(numbers)) == len(numbers)):
                        return numbers
                        
                except ValueError:
                    continue
        
        return []
    
    def _calculate_ndcg_at_k(self, qrel: Dict, predicted_ranking: List[int], k: int) -> float:
        if not predicted_ranking:
            return 0.0
            
        top_k_ranking = predicted_ranking[:k]
        
        # DCG calculation
        dcg = 0.0
        for position, doc_idx in enumerate(top_k_ranking):
            relevance = qrel.get(str(doc_idx), 0)
            dcg += (2**relevance - 1) / math.log2(position + 2)
        
        # IDCG calculation
        relevances = list(qrel.values())
        sorted_relevances = sorted(relevances, reverse=True)
        
        idcg = 0.0
        for position in range(min(k, len(sorted_relevances))):
            relevance = sorted_relevances[position]
            idcg += (2**relevance - 1) / math.log2(position + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_map_at_k(self, qrel: Dict, predicted_ranking: List[int], k: int) -> float:
        if not predicted_ranking:
            return 0.0
            
        top_k_ranking = predicted_ranking[:k]
        relevant_items = set(doc_id for doc_id, score in qrel.items() if score > 0)
        
        if not relevant_items:
            return 0.0
        
        precision_sum = 0.0
        relevant_found = 0
        
        for position, doc_idx in enumerate(top_k_ranking):
            if str(doc_idx) in relevant_items:
                relevant_found += 1
                precision_at_position = relevant_found / (position + 1)
                precision_sum += precision_at_position
        
        return precision_sum / len(relevant_items)
    
    def _calculate_mrr_at_k(self, qrel: Dict, predicted_ranking: List[int], k: int) -> float:
        if not predicted_ranking:
            return 0.0
            
        top_k_ranking = predicted_ranking[:k]
        
        for position, doc_idx in enumerate(top_k_ranking):
            if qrel.get(str(doc_idx), 0) > 0:
                return 1.0 / (position + 1)
        
        return 0.0
    
    def _save_results(self, results: List[EvaluationResult]):
        results_data = []
        for result in results:
            results_data.append({
                "model_name": result.model_name,
                "task_type": result.task_type,
                "total_samples": result.total_samples,
                "successful_parses": result.successful_parses,
                "parse_success_rate": result.parse_success_rate,
                "avg_ndcg": result.avg_ndcg,
                "avg_map": result.avg_map,
                "avg_mrr": result.avg_mrr,
                "ndcg_scores": result.ndcg_scores,
                "map_scores": result.map_scores,
                "mrr_scores": result.mrr_scores
            })
        
        results_file = self.results_dir / "baseline_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        self.logger.info(f"Results saved to: {results_file}")
    
    def _create_comparison_report(self, results: List[EvaluationResult]):
        summary_data = []
        for result in results:
            summary_data.append({
                "Model": result.model_name,
                "Task": result.task_type,
                "Samples": result.total_samples,
                "Parse Success": f"{result.parse_success_rate:.1%}",
                "nDCG": f"{result.avg_ndcg:.3f}",
                "MAP": f"{result.avg_map:.3f}",
                "MRR": f"{result.avg_mrr:.3f}"
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save CSV
        csv_file = self.results_dir / "baseline_comparison.csv"
        df.to_csv(csv_file, index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("BASELINE MODEL PERFORMANCE SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        print(f"Summary saved to: {csv_file}")
        self.logger.info(f"Summary saved to: {csv_file}")

async def main():
    evaluator = BaselineEvaluator(max_retries=1)
    
    # Example usage:
    # Run both tasks (default behavior)
    await evaluator.evaluate_all_baselines(num_samples=20)
    
    # Run only document_ranking
    #await evaluator.evaluate_all_baselines(num_samples=20, tasks=["document_ranking"])
    
    # Run only chunk_ranking
    # await evaluator.evaluate_all_baselines(num_samples=20, tasks=["chunk_ranking"])
    
    # Run both tasks explicitly
    # await evaluator.evaluate_all_baselines(num_samples=20, tasks=["document_ranking", "chunk_ranking"])

if __name__ == "__main__":
    asyncio.run(main())