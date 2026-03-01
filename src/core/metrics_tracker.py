"""
Metrics Tracker for Dinesh AI
Tracks model improvements across versions using:
- Perplexity (model confusion)
- BLEU Score (n-gram overlap)
- Vocabulary Match Ratio (% real English words)
"""

import torch
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import math
import re

logger = logging.getLogger(__name__)

class MetricsTracker:
    def __init__(self, config: dict):
        self.config = config
        self.metrics_dir = Path("metrics")
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Load English dictionary for vocab matching
        self.english_words = self._load_english_dict()
        
    def _load_english_dict(self) -> set:
        """Load basic English dictionary"""
        try:
            import nltk
            nltk.download('words', quiet=True)
            from nltk.corpus import words
            return set(w.lower() for w in words.words())
        except:
            # Fallback: common English words
            common = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 
                     'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
                     'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
                     'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
                     'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
                     'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
                     'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
                     'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
                     'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
                     'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
                     'is', 'was', 'are', 'been', 'has', 'had', 'were', 'said', 'did', 'having',
                     'may', 'should', 'am', 'being', 'does', 'done', 'hi', 'hello', 'yes', 'no'}
            return common
    
    def calculate_perplexity(self, model, dataloader, device) -> float:
        """Calculate perplexity (lower is better)"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    target_ids.view(-1),
                    ignore_index=0,
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += (target_ids != 0).sum().item()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return perplexity
    
    def calculate_vocab_match_ratio(self, text: str) -> Dict[str, float]:
        """Calculate % of real English words in generated text"""
        # Tokenize by whitespace and clean
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        if not words:
            return {"vocab_match_ratio": 0.0, "total_words": 0, "valid_words": 0}
        
        # Count valid English words
        min_len = self.config.get('metrics', {}).get('min_word_length', 2)
        valid_words = [w for w in words if len(w) >= min_len and w in self.english_words]
        
        ratio = len(valid_words) / len(words) if words else 0.0
        
        return {
            "vocab_match_ratio": ratio,
            "total_words": len(words),
            "valid_words": len(valid_words)
        }
    
    def calculate_bleu_score(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score (0-1, higher is better)"""
        # Simple BLEU-1 implementation
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if not hyp_words:
            return 0.0
        
        # Count matching words
        matches = sum(1 for w in hyp_words if w in ref_words)
        precision = matches / len(hyp_words) if hyp_words else 0.0
        
        # Brevity penalty
        bp = 1.0 if len(hyp_words) >= len(ref_words) else math.exp(1 - len(ref_words) / len(hyp_words))
        
        return bp * precision
    
    def evaluate_model(self, model, tokenizer, device, version: str) -> Dict:
        """Comprehensive model evaluation"""
        model.eval()
        
        test_prompts = self.config.get('metrics', {}).get('test_prompts', ['hi'])
        results = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "samples": []
        }
        
        for prompt in test_prompts:
            try:
                # Generate response
                encoded = tokenizer.encode(prompt)
                input_ids = torch.tensor([encoded.ids[:50]], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_length=100,
                        temperature=0.8,
                        top_p=0.92,
                        top_k=50,
                        eos_token_id=2
                    )
                
                response = tokenizer.decode(output_ids[0].cpu().tolist())
                
                # Calculate metrics
                vocab_metrics = self.calculate_vocab_match_ratio(response)
                
                results["samples"].append({
                    "prompt": prompt,
                    "response": response,
                    "vocab_match_ratio": vocab_metrics["vocab_match_ratio"],
                    "total_words": vocab_metrics["total_words"],
                    "valid_words": vocab_metrics["valid_words"]
                })
                
            except Exception as e:
                logger.error(f"Error evaluating prompt '{prompt}': {e}")
        
        # Calculate average vocab match ratio
        if results["samples"]:
            avg_ratio = sum(s["vocab_match_ratio"] for s in results["samples"]) / len(results["samples"])
            results["avg_vocab_match_ratio"] = avg_ratio
        else:
            results["avg_vocab_match_ratio"] = 0.0
        
        # Save results
        self._save_metrics(results, version)
        
        return results
    
    def _save_metrics(self, results: Dict, version: str):
        """Save metrics to JSON file"""
        filename = self.metrics_dir / f"metrics_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Metrics saved to {filename}")
    
    def compare_versions(self) -> Dict:
        """Compare all versions and show improvement"""
        metric_files = sorted(self.metrics_dir.glob("metrics_*.json"))
        
        if not metric_files:
            return {"error": "No metrics found"}
        
        versions = []
        for file in metric_files:
            with open(file) as f:
                data = json.load(f)
                versions.append({
                    "version": data.get("version", "unknown"),
                    "timestamp": data.get("timestamp", ""),
                    "avg_vocab_match_ratio": data.get("avg_vocab_match_ratio", 0.0)
                })
        
        # Sort by version
        versions.sort(key=lambda x: x["version"])
        
        # Calculate improvement
        if len(versions) > 1:
            first = versions[0]["avg_vocab_match_ratio"]
            latest = versions[-1]["avg_vocab_match_ratio"]
            improvement = ((latest - first) / first * 100) if first > 0 else 0
        else:
            improvement = 0
        
        return {
            "versions": versions,
            "total_versions": len(versions),
            "improvement_percentage": improvement,
            "first_version": versions[0] if versions else None,
            "latest_version": versions[-1] if versions else None
        }
    
    def generate_report(self) -> str:
        """Generate human-readable improvement report"""
        comparison = self.compare_versions()
        
        if "error" in comparison:
            return "No metrics available yet. Train the model first."
        
        report = []
        report.append("=" * 60)
        report.append("DINESH AI - MODEL IMPROVEMENT REPORT")
        report.append("=" * 60)
        report.append(f"Total Versions Tracked: {comparison['total_versions']}")
        report.append(f"Overall Improvement: {comparison['improvement_percentage']:.1f}%")
        report.append("")
        
        if comparison['first_version']:
            report.append(f"First Version: {comparison['first_version']['version']}")
            report.append(f"  Vocab Match: {comparison['first_version']['avg_vocab_match_ratio']:.1%}")
        
        if comparison['latest_version']:
            report.append(f"Latest Version: {comparison['latest_version']['version']}")
            report.append(f"  Vocab Match: {comparison['latest_version']['avg_vocab_match_ratio']:.1%}")
        
        report.append("")
        report.append("Version History:")
        for v in comparison['versions']:
            report.append(f"  {v['version']}: {v['avg_vocab_match_ratio']:.1%}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
