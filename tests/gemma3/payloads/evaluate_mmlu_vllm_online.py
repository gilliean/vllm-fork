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

# -------------------------------
# PERFORMANCE METRICS TRACKED:
# - TTFT (Time To First Token): Time from request to first token generation
# - TPOT (Time Per Output Token): Average time to generate each output token
# - Output Throughput: Output tokens generated per second
# - Total Token Throughput: Total tokens (input + output) processed per second
# - End-to-End Latency: Total time from request start to completion
# -------------------------------
# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_NAME = "meta-llama/Llama-3.1-405B-Instruct"  # Updated to match your command
BATCH_SIZE = 32
MAX_TOKENS = 8
TEMPERATURE = 0.0
API_BASE_URL = "http://localhost:8080/v1"
API_SERVER_PORT = 8080

# vLLM API Server Configuration
VLLM_SERVER_CMD = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_NAME,
    "--port", str(API_SERVER_PORT),
    "--max-num-seqs", "128",
    "--dtype", "bfloat16",
    "--gpu-memory-util", "0.95",
    "--tensor-parallel-size", "8",
    "--max-model-len", "4300",
    "--block-size", "256",
    "--num_scheduler_steps", "1",
    "--max-num-batched-tokens", "32768",
    "--max-num-prefill-seqs", "4",
    "--use-padding-aware-scheduling",
    "--disable-log-stats",
    "--disable-log-requests",
    "--quantization", "inc",
    "--kv-cache-dtype", "fp8_inc",
    "--weights-load-device", "cpu"
]

# Environment variables for Intel Gaudi
VLLM_ENV = os.environ.copy()
VLLM_ENV["PT_HPU_LAZY_MODE"] = "1"

# extra_prompt = "Provide the only one best answer at the end with a single letter, A, B, C, or D, in a pair of parentheses."

# -------------------------------
# API CLIENT FUNCTIONS
# -------------------------------
def start_vllm_server():
    """Start the vLLM OpenAI API server"""
    print(f"Starting vLLM server with model: {MODEL_NAME}")
    print("Server command:", " ".join(VLLM_SERVER_CMD))
    
    process = subprocess.Popen(
        VLLM_SERVER_CMD,
        env=VLLM_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    max_wait_time = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(f"{API_BASE_URL}/models", timeout=5)
            if response.status_code == 200:
                print("✓ vLLM server is ready!")
                return process
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(5)
        print(".", end="", flush=True)
    
    print(f"\n✗ Server failed to start within {max_wait_time} seconds")
    process.terminate()
    return None

def stop_vllm_server(process):
    """Stop the vLLM server process"""
    if process:
        print("\nStopping vLLM server...")
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        print("✓ Server stopped")

def make_api_request(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    """Make a completion request to the vLLM API server"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        
        end_time = time.time()
        result = response.json()
        
        # Extract metrics
        choice = result["choices"][0]
        usage = result.get("usage", {})
        
        # Calculate metrics
        end_to_end_latency = end_time - start_time
        output_text = choice["text"]
        
        # Estimate tokens (since API might not always provide exact counts)
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
        
    except Exception as e:
        print(f"API request failed: {e}")
        return None

# -------------------------------
# LOAD MODEL (replaced with API server startup)
# -------------------------------
# server_process = None
# try:
#     server_process = start_vllm_server()
#     if not server_process:
#         print("Failed to start vLLM server. Exiting.")
#         sys.exit(1)
# except KeyboardInterrupt:
#     print("\nInterrupted during server startup")
#     sys.exit(1)

# -------------------------------
# LOAD DATASET
# -------------------------------
print("Loading MMLU dataset...")
mmlu = load_dataset("cais/mmlu", "all")
dataset_subset_name_to_use = "test"
mmlu_test = mmlu[dataset_subset_name_to_use]
subjects = set(mmlu_test["subject"])
# os.environ["VLLM_SKIP_WARMUP"] = "true"

print(f"Subjects loaded: {len(subjects)} total")

# -------------------------------
# PROMPT TEMPLATE
# -------------------------------
PROMPT_TEMPLATE = """The following are multiple choice questions (with answers) about {subject}.

Question: {question}
A. {A}
B. {B}
C. {C}
D. {D}
Answer:"""

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def format_prompt(example, subject):
    choices = example["choices"]
    return PROMPT_TEMPLATE.format(
        subject=subject.replace("_", " "),
        question=example["question"],
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
    )

def extract_choice(output_text):
    # match = re.findall(r'\([A-Z]\)', output_text)
    # if len(match) == 0:
    #     return 'NA'
    # else:
    #     return match[-1][1]  # match is likely to be e.g. ['(B)']    
    match = re.search(r"\b([A-D])\b", output_text.strip())
    return match.group(1) if match else None

def safe_get_metrics(request_output):
    """Safely extract metrics from API response"""
    return request_output.get("usage", {})

# -------------------------------
# RUN EVALUATION
# -------------------------------
results = {}
overall_correct, overall_total = 0, 0

# Performance metrics tracking
all_ttft_times = []
all_tpot_times = []
all_output_throughput = []
all_total_throughput = []
all_end_to_end_latency = []
all_input_tokens = []
all_output_tokens = []

# -------------------------------
# RUN EVALUATION
# -------------------------------
results = {}
overall_correct, overall_total = 0, 0

# Performance metrics tracking
all_ttft_times = []
all_tpot_times = []
all_output_throughput = []
all_total_throughput = []
all_end_to_end_latency = []
all_input_tokens = []
all_output_tokens = []

def cleanup_and_exit(signum=None, frame=None):
    """Cleanup function to stop server on exit"""
    global server_process
    if server_process:
        stop_vllm_server(server_process)
    sys.exit(0)

# Register cleanup function
# signal.signal(signal.SIGINT, cleanup_and_exit)
# signal.signal(signal.SIGTERM, cleanup_and_exit)

try:
    for subject in subjects:
        data = mmlu_test.filter(lambda x: x["subject"] == subject)
        prompts = [format_prompt(x, subject) for x in data]

        print(f"\nEvaluating {subject} ({len(prompts)} samples)...")
        subject_correct = 0

        for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
            batch_prompts = prompts[i:i+BATCH_SIZE]
            
            # Process batch using API requests
            batch_start_time = time.time()
            batch_results = []
            
            # Use ThreadPoolExecutor for concurrent API calls
            with ThreadPoolExecutor(max_workers=min(len(batch_prompts), 8)) as executor:
                future_to_idx = {
                    executor.submit(make_api_request, prompt): j 
                    for j, prompt in enumerate(batch_prompts)
                }
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        if result:
                            batch_results.append((idx, result))
                    except Exception as e:
                        print(f"Request failed for batch item {idx}: {e}")
                        batch_results.append((idx, None))
            
            batch_end_time = time.time()
            batch_end_to_end_latency = batch_end_time - batch_start_time
            
            # Sort results by original order
            batch_results.sort(key=lambda x: x[0])
            
            for j, (original_idx, api_result) in enumerate(batch_results):
                if api_result is None:
                    continue
                    
                output_text = api_result["text"].strip()
                pred_choice = extract_choice(output_text)
                gold_idx = data[i+original_idx]["answer"]
                gold_choice = chr(ord("A") + gold_idx)
                if pred_choice == gold_choice:
                    subject_correct += 1
                
                # Extract performance metrics from API response
                end_to_end_latency = api_result["end_to_end_latency"]
                prompt_tokens = api_result["prompt_tokens"]
                completion_tokens = api_result["completion_tokens"]
                total_tokens = api_result["total_tokens"]
                
                # Store metrics
                all_end_to_end_latency.append(end_to_end_latency)
                all_input_tokens.append(prompt_tokens)
                all_output_tokens.append(completion_tokens)
                
                # Calculate throughputs
                if end_to_end_latency > 0:
                    output_throughput = completion_tokens / end_to_end_latency
                    total_throughput = total_tokens / end_to_end_latency
                    all_output_throughput.append(output_throughput)
                    all_total_throughput.append(total_throughput)
                
                # Note: TTFT and TPOT are not directly available from the API
                # These would require streaming responses to measure accurately
                # For now, we'll estimate based on total latency
                if completion_tokens > 0:
                    estimated_tpot = end_to_end_latency / completion_tokens
                    all_tpot_times.append(estimated_tpot)
                    
                    # Rough TTFT estimation (assume first token takes similar time as others)
                    estimated_ttft = estimated_tpot
                    all_ttft_times.append(estimated_ttft)

        acc = subject_correct / len(prompts)
        results[subject] = acc
        overall_correct += subject_correct
        overall_total += len(prompts)
        print(f"{subject:30s} Accuracy: {acc:.3f}")

finally:
    # Ensure server is stopped
    #cleanup_and_exit()
    pass

# -------------------------------
# FINAL SUMMARY
# -------------------------------
overall_acc = overall_correct / overall_total
print("\n========== FINAL RESULTS ==========")
for s, a in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{s:30s}: {a:.3f}")
print(f"\nOverall Accuracy: {overall_acc:.3f}")

# -------------------------------
# PERFORMANCE METRICS SUMMARY
# -------------------------------
print("\n========== PERFORMANCE METRICS ==========")
print("Note: TTFT and TPOT are estimated from total latency when using non-streaming API")
print("For accurate TTFT/TPOT measurements, consider using streaming API endpoints")

if all_ttft_times:
    avg_ttft = np.mean(all_ttft_times)
    p50_ttft = np.median(all_ttft_times)
    p95_ttft = np.percentile(all_ttft_times, 95)
    print(f"\nEstimated Time To First Token (TTFT):")
    print(f"  Average: {avg_ttft:.4f} seconds")
    print(f"  Median (P50): {p50_ttft:.4f} seconds")
    print(f"  P95: {p95_ttft:.4f} seconds")

if all_tpot_times:
    avg_tpot = np.mean(all_tpot_times)
    p50_tpot = np.median(all_tpot_times)
    p95_tpot = np.percentile(all_tpot_times, 95)
    print(f"\nEstimated Time Per Output Token (TPOT):")
    print(f"  Average: {avg_tpot:.4f} seconds")
    print(f"  Median (P50): {p50_tpot:.4f} seconds")
    print(f"  P95: {p95_tpot:.4f} seconds")

if all_output_throughput:
    avg_output_throughput = np.mean(all_output_throughput)
    p50_output_throughput = np.median(all_output_throughput)
    print(f"\nOutput Throughput:")
    print(f"  Average: {avg_output_throughput:.2f} tokens/second")
    print(f"  Median (P50): {p50_output_throughput:.2f} tokens/second")

if all_total_throughput:
    avg_total_throughput = np.mean(all_total_throughput)
    p50_total_throughput = np.median(all_total_throughput)
    print(f"\nTotal Token Throughput:")
    print(f"  Average: {avg_total_throughput:.2f} tokens/second")
    print(f"  Median (P50): {p50_total_throughput:.2f} tokens/second")

if all_end_to_end_latency:
    avg_e2e_latency = np.mean(all_end_to_end_latency)
    p50_e2e_latency = np.median(all_end_to_end_latency)
    p95_e2e_latency = np.percentile(all_end_to_end_latency, 95)
    print(f"\nEnd-to-End Latency:")
    print(f"  Average: {avg_e2e_latency:.4f} seconds")
    print(f"  Median (P50): {p50_e2e_latency:.4f} seconds")
    print(f"  P95: {p95_e2e_latency:.4f} seconds")

# Additional summary statistics
total_input_tokens = sum(all_input_tokens) if all_input_tokens else 0
total_output_tokens = sum(all_output_tokens) if all_output_tokens else 0
total_tokens = total_input_tokens + total_output_tokens

print(f"\n========== TOKEN STATISTICS ==========")
print(f"Total Input Tokens: {total_input_tokens:,}")
print(f"Total Output Tokens: {total_output_tokens:,}")
print(f"Total Tokens: {total_tokens:,}")
print(f"Average Input Tokens per Request: {np.mean(all_input_tokens):.1f}" if all_input_tokens else "N/A")
print(f"Average Output Tokens per Request: {np.mean(all_output_tokens):.1f}" if all_output_tokens else "N/A")
