import re
import time
import json
import subprocess
import signal
import sys
import requests
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# -------------------------------
# LOGGING CONFIGURATION
# -------------------------------
# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# USAGE
# 1. Using OPENROUTER:
# Set your OpenRouter API key in the environment variable before running the script:
# export OPENROUTER_API_KEY="YOUR Openrouter API KEY"
#
# 1.1 List all available subjects
# python evaluate_mmlu_vllm_online_openai.py --list-subjects
# 1.2 Evaluate high_school_mathematics and high_school_physics subjects with max 10 questions each and batch size of 16
# python evaluate_mmlu_vllm_online_openai.py --model openai/gpt-oss-20b:free  --subjects high_school_mathematics  high_school_physics  --max-questions 10 --batch-size 16
# 1.3 Evaluate ALL subjects with max 10 questions each and batch size of 32
# python evaluate_mmlu_vllm_online_openai.py --model openai/gpt-oss-20b:free  --max-questions 10 --batch-size 32
#
# 2. Using local vLLM server:
# 2.1 Start vLLM server (adjust model name and parameters as needed)
# 2.2 Evaluate high_school_mathematics and high_school_physics subjects with max 10 questions each
# unset OPENROUTER_API_KEY
# python evaluate_mmlu_vllm_online_openai.py --model openai/gpt-oss-20b  --subjects high_school_mathematics  high_school_physics  --max-questions 10 
# 2.3 Evaluate ALL subjects with max 32 questions each and batch size of 32 with log-level DEBUG
# python evaluate_mmlu_vllm_online_openai.py --model openai/gpt-oss-20b  --max-questions 32 --batch-size 32 --log-level DEBUG
# -------------------------------
# PERFORMANCE METRICS TRACKED:
# - TTFT (Time To First Token): Time from request to first token generation
# - TPOT (Time Per Output Token): Average time to generate each output token
# - Output Throughput: Output tokens generated per second
# - Total Token Throughput: Total tokens (input + output) processed per second
# - End-to-End Latency: Total time from request start to completion
# -------------------------------
MAX_BATCH_SIZE = 128
# -------------------------------
# CONFIGURATION DATACLASS
# -------------------------------
@dataclass
class EvaluationConfig:
    """Configuration for MMLU evaluation"""
    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-405B-Instruct"
    
    # API configuration
    api_base_url: str = "http://localhost:8080/v1"
    api_server_port: int = 8080
    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(default_factory=lambda: os.getenv('OPENROUTER_API_BASE_URL', 'https://openrouter.ai/api/v1').rstrip('/'))
    
    # Request configuration
    batch_size: int = 128
    max_output_tokens: int = 4096
    temperature: float = 0.0
    request_timeout: int = 200  # seconds
    max_num_questions: Optional[int] = 10
    
    # Subject filtering
    subjects: Optional[List[str]] = None
    
    # Logging
    log_level: str = "INFO"
    
    # vLLM server configuration
    vllm_server_config: Dict = field(default_factory=lambda: {
        "max_num_seqs": "128",
        "dtype": "bfloat16",
        "gpu_memory_util": "0.95",
        "tensor_parallel_size": "8",
        "max_model_len": "8196",
        "block_size": "256",
        "num_scheduler_steps": "1",
        "max_num_batched_tokens": "32768",
        "max_num_prefill_seqs": "4",
        "use_padding_aware_scheduling": True,
        "disable_log_stats": True,
        "disable_log_requests": True,
        "quantization": "inc",
        "kv_cache_dtype": "fp8_inc",
        "weights_load_device": "cpu"
    })
    
    # Environment variables
    pt_hpu_lazy_mode: str = "1"
    
    def __post_init__(self):
        """Post-initialization to set API URL based on API key"""
        if self.openrouter_api_key:
            self.api_base_url = self.openrouter_base_url
            logger.info(f"Using OpenRouter API endpoint: {self.api_base_url}")
        else:
            logger.warning("OPENROUTER_API_KEY environment variable not set")
            logger.info(f"Using default API endpoint: {self.api_base_url}")
    
    def get_vllm_server_command(self) -> List[str]:
        """Generate vLLM server command from configuration"""
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--port", str(self.api_server_port),
        ]
        
        # Add configuration parameters
        for key, value in self.vllm_server_config.items():
            param_name = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(param_name)
            else:
                cmd.extend([param_name, str(value)])
        
        return cmd
    
    def get_vllm_env(self) -> Dict[str, str]:
        """Get environment variables for vLLM server"""
        env = os.environ.copy()
        env["PT_HPU_LAZY_MODE"] = self.pt_hpu_lazy_mode
        return env

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    ttft_times: List[float] = field(default_factory=list)
    tpot_times: List[float] = field(default_factory=list)
    output_throughput: List[float] = field(default_factory=list)
    total_throughput: List[float] = field(default_factory=list)
    end_to_end_latency: List[float] = field(default_factory=list)
    input_tokens: List[int] = field(default_factory=list)
    output_tokens: List[int] = field(default_factory=list)
    
    def add_metrics(self, end_to_end_latency: float, prompt_tokens: int, 
                   completion_tokens: int, total_tokens: int):
        """Add metrics for a single request"""
        self.end_to_end_latency.append(end_to_end_latency)
        self.input_tokens.append(prompt_tokens)
        self.output_tokens.append(completion_tokens)
        
        # Calculate throughputs
        if end_to_end_latency > 0:
            output_throughput = completion_tokens / end_to_end_latency
            total_throughput = total_tokens / end_to_end_latency
            self.output_throughput.append(output_throughput)
            self.total_throughput.append(total_throughput)
        
        # Estimate TTFT and TPOT: Assumes first token time equals TPOT; actual TTFT is typically higher due to prompt processing overhead
        if completion_tokens > 0:
            estimated_tpot = end_to_end_latency / completion_tokens
            self.tpot_times.append(estimated_tpot)
            estimated_ttft = estimated_tpot
            self.ttft_times.append(estimated_ttft)
    
    def print_summary(self):
        """Print performance metrics summary"""
        logger.info("\n========== PERFORMANCE METRICS ==========")
        logger.info("Note: TTFT and TPOT are estimated from total latency when using non-streaming API")
        logger.info("For accurate TTFT/TPOT measurements, consider using streaming API endpoints")

        if self.ttft_times:
            avg_ttft = np.mean(self.ttft_times)
            p50_ttft = np.median(self.ttft_times)
            p95_ttft = np.percentile(self.ttft_times, 95)
            logger.info(f"\nEstimated Time To First Token (TTFT):")
            logger.info(f"  Average: {avg_ttft:.4f} seconds")
            logger.info(f"  Median (P50): {p50_ttft:.4f} seconds")
            logger.info(f"  P95: {p95_ttft:.4f} seconds")

        if self.tpot_times:
            avg_tpot = np.mean(self.tpot_times)
            p50_tpot = np.median(self.tpot_times)
            p95_tpot = np.percentile(self.tpot_times, 95)
            logger.info(f"\nEstimated Time Per Output Token (TPOT):")
            logger.info(f"  Average: {avg_tpot:.4f} seconds")
            logger.info(f"  Median (P50): {p50_tpot:.4f} seconds")
            logger.info(f"  P95: {p95_tpot:.4f} seconds")

        if self.output_throughput:
            avg_output_throughput = np.mean(self.output_throughput)
            p50_output_throughput = np.median(self.output_throughput)
            logger.info(f"\nOutput Throughput:")
            logger.info(f"  Average: {avg_output_throughput:.2f} tokens/second")
            logger.info(f"  Median (P50): {p50_output_throughput:.2f} tokens/second")

        if self.total_throughput:
            avg_total_throughput = np.mean(self.total_throughput)
            p50_total_throughput = np.median(self.total_throughput)
            logger.info(f"\nTotal Token Throughput:")
            logger.info(f"  Average: {avg_total_throughput:.2f} tokens/second")
            logger.info(f"  Median (P50): {p50_total_throughput:.2f} tokens/second")

        if self.end_to_end_latency:
            avg_e2e_latency = np.mean(self.end_to_end_latency)
            p50_e2e_latency = np.median(self.end_to_end_latency)
            p95_e2e_latency = np.percentile(self.end_to_end_latency, 95)
            logger.info(f"\nEnd-to-End Latency:")
            logger.info(f"  Average: {avg_e2e_latency:.4f} seconds")
            logger.info(f"  Median (P50): {p50_e2e_latency:.4f} seconds")
            logger.info(f"  P95: {p95_e2e_latency:.4f} seconds")

        # Token statistics
        total_input_tokens = sum(self.input_tokens) if self.input_tokens else 0
        total_output_tokens = sum(self.output_tokens) if self.output_tokens else 0
        total_tokens = total_input_tokens + total_output_tokens

        logger.info(f"\n========== TOKEN STATISTICS ==========")
        logger.info(f"Total Input Tokens: {total_input_tokens:,}")
        logger.info(f"Total Output Tokens: {total_output_tokens:,}")
        logger.info(f"Total Tokens: {total_tokens:,}")
        if self.input_tokens:
            logger.info(f"Average Input Tokens per Request: {np.mean(self.input_tokens):.1f}")
        if self.output_tokens:
            logger.info(f"Average Output Tokens per Request: {np.mean(self.output_tokens):.1f}")

# -------------------------------
# API CLIENT FUNCTIONS
# -------------------------------
def start_vllm_server(config: EvaluationConfig):
    """Start the vLLM OpenAI API server"""
    logger.info(f"Starting vLLM server with model: {config.model_name}")
    cmd = config.get_vllm_server_command()
    logger.debug(f"Server command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        env=config.get_vllm_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Wait for server to start
    logger.info("Waiting for server to start...")
    max_wait_time = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(f"{config.api_base_url}/models", timeout=5)
            if response.status_code == 200:
                logger.info("✓ vLLM server is ready!")
                return process
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(5)
        logger.debug("Waiting for server...")
    
    logger.error(f"Server failed to start within {max_wait_time} seconds")
    process.terminate()
    return None

def stop_vllm_server(process):
    """Stop the vLLM server process"""
    if process:
        logger.info("Stopping vLLM server...")
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        logger.info("✓ Server stopped")

def make_api_request(prompt: str, config: EvaluationConfig):
    """Make a completion request to the vLLM API server"""
    payload = {
        "model": "/mnt/huggingface/hub/upstage-solar-pro2",
        "messages": [
            {"role": "system", "content": "You are a scholar with knowledge on various subjects."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]
       ,
       "max_tokens": config.max_output_tokens,
       "temperature": config.temperature,
       "stream": False
    }
    
    start_time = time.time()
    
    # Prepare headers with API key
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add Authorization header if API key is provided
    if config.openrouter_api_key:
        headers["Authorization"] = f"Bearer {config.openrouter_api_key}"
    
    try:
        response = requests.post(
            f"{config.api_base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=config.request_timeout
        )
        response.raise_for_status()
        
        end_time = time.time()
        result = response.json()
        
        # Extract metrics
        choice = result["choices"][0]
        usage = result.get("usage", {})
        
        # Calculate metrics
        end_to_end_latency = end_time - start_time
        output_text = choice["message"]["content"]

        logger.debug(f"Prompt: {prompt[:100]}...")
        logger.debug(f"Output: {output_text[:100]}...")
        
        # Estimate tokens (since API might not always provide exact counts). 
        # 1.3 is a magic number from rule-of-thumb for English
        prompt_tokens = usage.get("prompt_tokens", len(prompt.split()) * 1.3)
        completion_tokens = usage.get("completion_tokens", len(output_text.split()) * 1.3)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        
        return {
            "text": output_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "end_to_end_latency": end_to_end_latency,
            "usage": usage
        }
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error during API request: {e}")
        return None
    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        return None
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return None

# -------------------------------
# PARSE COMMAND LINE ARGUMENTS
# -------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MMLU dataset with vLLM API")
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="List of subjects to evaluate (space-separated). If not provided, all subjects will be evaluated. "
             "Example: --subjects abstract_algebra anatomy business_ethics"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="give name of model to use. If not provided, default model will be used."
    )
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="List all available subjects and exit"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions per subject (default: all questions)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=200,
        help="Request timeout in seconds (default: 200)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )
    return parser.parse_args()


def format_prompt(example, prompt_template: str, subject: str) -> str:
    choices = example["choices"]
    return prompt_template.format(
        subject=subject.replace("_", " "),
        question=example["question"],
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
    )

def extract_choice(output_text: str) -> Optional[str]:
    # Look for pattern like (A), (B), (C), or (D)
    match = re.search(r'\(([A-D])\)', output_text)
    return match.group(1) if match else None

def cleanup_and_exit(server_process, signum=None, frame=None):
    """Cleanup function to stop server on exit"""
    if server_process:
        stop_vllm_server(server_process)
    sys.exit(0)

def create_config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    """Create EvaluationConfig from command line arguments"""
    config = EvaluationConfig()
    
    # Update config from arguments
    if args.model:
        config.model_name = args.model
    if args.max_questions is not None:
        config.max_num_questions = args.max_questions
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.request_timeout:
        config.request_timeout = args.request_timeout
    config.log_level = args.log_level
    config.subjects = args.subjects
    
    return config

def evaluate_subject(subject: str, data, config: EvaluationConfig, 
                    prompt_template: str, metrics: PerformanceMetrics) -> tuple[int, int]:
    """Evaluate a single subject
    
    Returns:
        tuple: (correct_count, total_count)
    """
    # Limit number of questions if MAX_NUM_QUESTIONS is set
    num_questions = min(len(data), config.max_num_questions) if config.max_num_questions else len(data)
    data = data.select(range(num_questions))
    
    prompts = [format_prompt(x, prompt_template, subject) for x in data]

    logger.info(f"Evaluating {subject} ({len(prompts)} samples)...")
    subject_correct = 0

    for i in tqdm(range(0, len(prompts), config.batch_size), desc=f"Processing {subject} batches"):
        batch_prompts = prompts[i:i+config.batch_size]
        
        # Process batch using API requests
        batch_start_time = time.time()
        batch_results = []
        
        # Use ThreadPoolExecutor for concurrent API calls
        max_batch_size = min(MAX_BATCH_SIZE, config.batch_size)
        with ThreadPoolExecutor(max_workers=min(len(batch_prompts), max_batch_size)) as executor:
            future_to_idx = {
                executor.submit(make_api_request, prompt, config): j 
                for j, prompt in enumerate(batch_prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result:
                        batch_results.append((idx, result))
                except Exception as e:
                    logger.error(f"Request failed for batch item {idx}: {e}")
                    batch_results.append((idx, None))
        
        batch_end_time = time.time()
        batch_end_to_end_latency = batch_end_time - batch_start_time
        
        # Sort results by original order
        batch_results.sort(key=lambda x: x[0])
        logger.debug(f"Subject: {subject}, Batch results: {len(batch_results)}")
        
        for j, (original_idx, api_result) in enumerate(batch_results):
            if api_result is None:
                logger.warning(f"Skipping question {i+original_idx+1} due to API failure")
                continue
                
            output_text = api_result["text"].strip()
            pred_choice = extract_choice(output_text)
            gold_idx = data[i+original_idx]["answer"]
            gold_choice = chr(ord("A") + gold_idx)

            # Debug output - only print when predictions don't match or are None
            if pred_choice != gold_choice:
                logger.debug(f"[{subject}] Question {i+original_idx+1}:")
                logger.debug(f"  Output: {output_text[:150]}...")
                logger.debug(f"  Predicted: {pred_choice if pred_choice else 'NO ANSWER FOUND'}")
                logger.debug(f"  Gold: {gold_choice}")
            
            # Only count as correct if we extracted a valid choice AND it matches
            if pred_choice is not None and pred_choice == gold_choice:
                subject_correct += 1
                logger.debug(f"✓ Correct answer for question {i+original_idx+1}")
            elif pred_choice is None:
                logger.warning(f"Could not extract answer from response for question {i+original_idx+1}")
            else:
                logger.debug(f"✗ Incorrect answer for question {i+original_idx+1}")
            
            # Extract and store performance metrics
            end_to_end_latency = api_result["end_to_end_latency"]
            prompt_tokens = api_result["prompt_tokens"]
            completion_tokens = api_result["completion_tokens"]
            total_tokens = api_result["total_tokens"]

            logger.debug(f"Metrics - Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Latency: {end_to_end_latency:.3f}s")
            
            metrics.add_metrics(end_to_end_latency, prompt_tokens, completion_tokens, total_tokens)

    return subject_correct, len(prompts)

def main():
    server_process = None
    
    # Parse command line arguments
    args = parse_args()
    
    # Set logging level based on argument
    logger.setLevel(getattr(logging, args.log_level))
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # -------------------------------
    # LOAD DATASET
    # -------------------------------
    logger.info("Loading MMLU dataset...")
    mmlu = load_dataset("cais/mmlu", "all")
    dataset_subset_name_to_use = "test"
    mmlu_test = mmlu[dataset_subset_name_to_use]
    all_subjects = sorted(set(mmlu_test["subject"]))

    # If --list-subjects is provided, print subjects and exit
    if args.list_subjects:
        logger.info(f"Available subjects ({len(all_subjects)}):")
        for i, subject in enumerate(all_subjects, 1):
            print(f"  {i:2d}. {subject}")
        sys.exit(0)

    # Create configuration from arguments
    config = create_config_from_args(args)
    
    # Filter subjects based on command line argument
    if config.subjects:
        # Validate that provided subjects exist
        invalid_subjects = [s for s in config.subjects if s not in all_subjects]
        if invalid_subjects:
            logger.error(f"Invalid subjects provided: {', '.join(invalid_subjects)}")
            logger.info("Use --list-subjects to see all available subjects")
            sys.exit(1)
        
        subjects = config.subjects
        logger.info(f"Selected {len(subjects)} subject(s): {', '.join(subjects)}")
    else:
        subjects = all_subjects
        logger.info(f"Evaluating all {len(subjects)} subjects")

    logger.info("Configuration:")
    logger.info(f"  - Model: {config.model_name}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Max questions per subject: {config.max_num_questions if config.max_num_questions else 'All'}")
    logger.info(f"  - API Base URL: {config.api_base_url}")
    logger.info(f"  - Request timeout: {config.request_timeout}s")
    logger.info(f"  - Log level: {config.log_level}")

    # Prompt template
    PROMPT_TEMPLATE = """The following are multiple choice questions (with answers) about {subject}.

Question: {question} Provide one correct answer: A, B, C, or D at the end of your reasoning within a pair of parentheses, e.g. "Answer is (A)", "The answer is (B)", "Answer is (C)", or "Answer is (D)". Keep your reasoning brief.
A. {A}
B. {B}
C. {C}
D. {D}
Answer:"""

    # -------------------------------
    # RUN EVALUATION
    # -------------------------------
    results = {}
    overall_correct, overall_total = 0, 0
    metrics = PerformanceMetrics()

    try:
        for i in tqdm(range(0, len(subjects)), desc="Evaluating subjects"):
            subject = subjects[i]
            data = mmlu_test.filter(lambda x: x["subject"] == subject)
            subject_correct, subject_total = evaluate_subject(
                subject, data, config, PROMPT_TEMPLATE, metrics
            )
            
            logger.info(f"Subject results - Correct: {subject_correct}, Total: {subject_total}")
            acc = subject_correct / subject_total if subject_total > 0 else 0
            results[subject] = acc
            overall_correct += subject_correct
            overall_total += subject_total
            logger.info(f"{subject:30s} Accuracy: {acc:.3f}")

    finally:
        pass

    # -------------------------------
    # FINAL SUMMARY
    # -------------------------------
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0
    logger.info("\n========== FINAL RESULTS ==========")
    for s, a in sorted(results.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{s:30s}: {a:.3f}")
    logger.info(f"\nOverall Accuracy: {overall_acc:.3f}")

    # Print performance metrics
    metrics.print_summary()

if __name__ == "__main__":
    main()
