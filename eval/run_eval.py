import os
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import json
import sys
import argparse
import asyncio
from pathlib import Path
from eval.schema import EvalItem
from backend.app.retrieval.vector_store import search
from backend.app.generation.answer import answer_question
from backend.app.core.config import settings

# --- RAGAS Imports ---
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# --- Providers ---
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.outputs import ChatResult

class SafeChatGroq(ChatGroq):
    """Bypasses Groq's n=1 limitation by splitting multiple generations into parallel requests."""
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        n = kwargs.pop("n", 1)
        results = [super(SafeChatGroq, self)._generate(messages, stop, run_manager, **kwargs) for _ in range(n)]
        combined = [gen for res in results for gen in res.generations]
        return ChatResult(generations=combined, llm_output=results[0].llm_output)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        n = kwargs.pop("n", 1)
        tasks = [super(SafeChatGroq, self)._agenerate(messages, stop, run_manager, **kwargs) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        combined = [gen for res in results for gen in res.generations]
        return ChatResult(generations=combined, llm_output=results[0].llm_output)

def load_dataset(filepath: str) -> list[EvalItem]:
    items = []
    with open(filepath, 'r') as f:
        for line_idx, line in enumerate(f):
            try:
                items.append(EvalItem(**json.loads(line)))
            except Exception as e:
                print(f"Schema Error on line {line_idx + 1}: {e}")
                sys.exit(1)
    return items

def run_eval(smoke: bool = False):
    dataset = load_dataset('eval/dataset.jsonl')
    eval_items = [item for item in dataset if item.split == 'dev']
    
    if smoke:
        with open('eval/smoke_ids.txt', 'r') as f:
            smoke_ids = set(line.strip() for line in f if line.strip())
        eval_items = [item for item in eval_items if item.id in smoke_ids]
        print(f"Running SMOKE test on {len(eval_items)} questions (Retrieval Only)...")
        
    reports = []
    ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    hits, mrr_sum = 0, 0.0
    
    for item in eval_items:
        print(f"Evaluating: {item.id} - {item.question[:50]}...")
        
        # 1. Retrieval
        results = search(collection_name="documents", query=item.question, top_k=5)
        retrieved_pages = [res["payload"]["page"] for res in results]
        retrieved_texts = [res["payload"]["text"] for res in results]
        
        # 2. Retrieval Metrics
        hit, mrr = False, 0.0
        if item.answerable and item.pages:
            gold_page = item.pages[0]
            if gold_page in retrieved_pages:
                hit = True
                hits += 1
                mrr = 1.0 / (retrieved_pages.index(gold_page) + 1)
                mrr_sum += mrr
        
        report = {
            "id": item.id,
            "question": item.question,
            "hit": hit,
            "mrr": mrr
        }
        
        # 3. Generation & RAGAS Prep (Skip if smoke)
        if not smoke:
            ans = answer_question(item.question)
            report["generated_answer"] = ans["answer"]
            
            ragas_data["question"].append(item.question)
            ragas_data["answer"].append(ans["answer"])
            ragas_data["contexts"].append(retrieved_texts)
            ragas_data["ground_truth"].append(item.reference_answer or "")
            
        reports.append(report)
        
    # 4. Summary & RAGAS Execution
    total = len(eval_items)
    summary = {
        "total_questions": total,
        "hit_rate_at_5": (hits / total) if total > 0 else 0.0,
        "mrr": (mrr_sum / total) if total > 0 else 0.0
    }
    
    if not smoke and total > 0:
        print("\nPreparing RAGAS LLM-as-a-Judge evaluation...")
        
        # --- SMART ROUTER FOR EVALUATION ---
        provider_choice = settings.LLM_PROVIDER.lower().strip()
        
        if provider_choice == "groq" and settings.GROQ_API_KEY:
            print("Using Groq Cloud (openai/gpt-oss-20b) for lightning-fast grading...")
            base_llm = SafeChatGroq(
                model="openai/gpt-oss-20b",
                api_key=settings.GROQ_API_KEY,
                temperature=0.0
            )
        else:
            print("Using Local Ollama (qwen2.5) for offline grading... (This takes a moment)")
            base_llm = ChatOllama(model="qwen2.5:latest")
        
        # Embeddings remain local because they are highly optimized and nearly instantaneous
        base_embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        
        eval_llm = LangchainLLMWrapper(base_llm)
        eval_embeddings = LangchainEmbeddingsWrapper(base_embeddings)
        
        ragas_ds = Dataset.from_dict(ragas_data)
        ragas_config = RunConfig(timeout=300, max_retries=2, max_workers=1)
        
        ragas_result = evaluate(
            ragas_ds,
            metrics=[Faithfulness()],
            llm=eval_llm,
            embeddings=eval_embeddings,
            run_config=ragas_config
        )
        
        try:
            summary["ragas_faithfulness"] = ragas_result["faithfulness"]
        except KeyError:
            summary["ragas_faithfulness"] = 0.0

    print("\n EVALUATION SUMMARY ")
    print(json.dumps(summary, indent=2))
    
    # 5. Save Reports
    Path('eval/reports').mkdir(parents=True, exist_ok=True)
    with open(f'eval/reports/summary_{"smoke" if smoke else "full"}.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    with open(f'eval/reports/per_question_{"smoke" if smoke else "full"}.jsonl', 'w') as f:
        for r in reports:
            f.write(json.dumps(r) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run_eval(args.smoke)