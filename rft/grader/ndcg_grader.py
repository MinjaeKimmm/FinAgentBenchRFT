import math, re
from typing import List, Dict

# Helpers
_LIST_RE = re.compile(r'\[([0-9,\s]+)\]')          # matches "[0, 2, 1]" etc.
_NUMS_RE = re.compile(r'(\d+(?:\s*,\s*\d+)*)')     # fallback "0,2,1"

def parse_ranking(text: str, max_index: int) -> List[int]:
    """
    Extract list of ints from model text. Returns [] on failure.
    """
    for pattern in (_LIST_RE, _NUMS_RE):
        m = pattern.search(text)
        if not m:
            continue
        try:
            nums = [int(x.strip()) for x in m.group(1).split(",")]
        except ValueError:
            continue
        # basic validity checks
        if (1 <= len(nums) <= max_index + 1
                and all(0 <= n <= max_index for n in nums)
                and len(set(nums)) == len(nums)):
            return nums
    return []

def ndcg_at_k(qrel: Dict[str, int], ranking: List[int], k: int) -> float:
    """Standard nDCG@k (graded relevance, log₂ discount)."""
    if not ranking:
        return 0.0
    
    dcg = 0.0
    for pos, idx in enumerate(ranking[:k]):
        rel = qrel.get(str(idx), 0)
        dcg += (2**rel - 1) / math.log2(pos + 2)
    
    # ideal DCG
    ideal = sorted(qrel.values(), reverse=True)
    idcg = 0.0
    for pos, rel in enumerate(ideal[:k]):
        idcg += (2**rel - 1) / math.log2(pos + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def grade(sample: dict, item: dict) -> float:
    """
    Grade function following OpenAI cookbook patterns.
    
    Args:
        sample: {"output_text": "...", "choices": [...], "output_json": {}, "output_tools": []}
        item: {"qrel": {...}, "metadata": {...}, ...}
    
    Returns:
        float: nDCG score between 0.0 and 1.0
    """
    # Direct access to output_text as per cookbook
    output_text = sample["output_text"]
    
    # Extract required data
    qrel = item["qrel"]
    task_type = item.get("metadata", {}).get("task_type", "document_ranking")
    k = 5 if task_type == "document_ranking" else 10
    
    # Validate qrel is not empty
    if not qrel:
        return 0.0
    
    # Get max index from qrel keys
    max_idx = max(int(i) for i in qrel.keys())
    
    # Parse ranking from model output
    ranking = parse_ranking(output_text, max_idx)
    
    # Calculate and return nDCG score
    return ndcg_at_k(qrel, ranking, k)