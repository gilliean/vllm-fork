import re
import time
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
import numpy as np
import os

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
MODEL_NAME = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"  # example model
BATCH_SIZE = 32
MAX_TOKENS = 8
TEMPERATURE = 0.0
# extra_prompt = "Provide the only one best answer at the end with a single letter, A, B, C, or D, in a pair of parentheses."

# -------------------------------
# LOAD MODEL (with tokenizer-mode)
# -------------------------------
print(f"Loading model: {MODEL_NAME} (tokenizer-mode='mistral')")
llm = LLM(model=MODEL_NAME, tokenizer_mode="mistral")
sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

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
    """Safely extract metrics from vLLM request output"""
    try:
        return request_output.metrics if hasattr(request_output, 'metrics') else None
    except:
        return None

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

for subject in subjects:
    data = mmlu_test.filter(lambda x: x["subject"] == subject)
    # data = mmlu[dataset_name_to_use]["subject"]
    # list(mmlu[dataset_name_to_use]["subject"].keys())
    prompts = [format_prompt(x, subject) for x in data]

    print(f"\nEvaluating {subject} ({len(prompts)} samples)...")
    subject_correct = 0

    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        
        # Record start time for end-to-end latency
        start_time = time.time()
        outputs = llm.generate(batch_prompts, sampling_params)
        end_time = time.time()
        
        # Calculate end-to-end latency for this batch
        batch_end_to_end_latency = end_time - start_time
        
        for j, out in enumerate(outputs):
            output_text = out.outputs[0].text.strip()
            pred_choice = extract_choice(output_text)
            gold_idx = data[i+j]["answer"]
            gold_choice = chr(ord("A") + gold_idx)
            if pred_choice == gold_choice:
                subject_correct += 1
            
            # Extract performance metrics from vLLM output
            request_output = out
            metrics = safe_get_metrics(request_output)
            
            if metrics:
                # Time to first token (TTFT) in seconds
                if hasattr(metrics, 'first_token_time') and hasattr(metrics, 'arrival_time'):
                    ttft = metrics.first_token_time - metrics.arrival_time
                    all_ttft_times.append(ttft)
                
                # Time from first token to last token (for TPOT calculation)
                if (hasattr(metrics, 'finished_time') and hasattr(metrics, 'first_token_time') 
                    and metrics.finished_time and metrics.first_token_time):
                    time_for_output_tokens = metrics.finished_time - metrics.first_token_time
                    num_output_tokens = len(request_output.outputs[0].token_ids)
                    
                    if num_output_tokens > 1:  # Avoid division by zero
                        tpot = time_for_output_tokens / (num_output_tokens - 1)
                        all_tpot_times.append(tpot)
                    
                    # Output throughput (output tokens per second)
                    if time_for_output_tokens > 0:
                        output_throughput = num_output_tokens / time_for_output_tokens
                        all_output_throughput.append(output_throughput)
                    
                    all_output_tokens.append(num_output_tokens)
            else:
                # Fallback: use basic token counting if detailed metrics unavailable
                num_output_tokens = len(request_output.outputs[0].token_ids) if hasattr(request_output.outputs[0], 'token_ids') else len(output_text.split())
                all_output_tokens.append(num_output_tokens)
            
            # End-to-end latency per request (approximate from batch)
            per_request_latency = batch_end_to_end_latency / len(batch_prompts)
            all_end_to_end_latency.append(per_request_latency)
            
            # Estimate input tokens
            try:
                input_tokens = len(llm.get_tokenizer().encode(batch_prompts[j]))
                all_input_tokens.append(input_tokens)
                
                # Total token throughput (input + output tokens per second)
                total_tokens = input_tokens + (all_output_tokens[-1] if all_output_tokens else 0)
                if per_request_latency > 0:
                    total_throughput = total_tokens / per_request_latency
                    all_total_throughput.append(total_throughput)
            except:
                # Fallback estimation if tokenizer access fails
                estimated_input_tokens = len(batch_prompts[j].split()) * 1.3  # Rough estimation
                all_input_tokens.append(estimated_input_tokens)
                
                # Still calculate total throughput with estimated input tokens
                total_tokens = estimated_input_tokens + (all_output_tokens[-1] if all_output_tokens else 0)
                if per_request_latency > 0:
                    total_throughput = total_tokens / per_request_latency
                    all_total_throughput.append(total_throughput)

    acc = subject_correct / len(prompts)
    results[subject] = acc
    overall_correct += subject_correct
    overall_total += len(prompts)
    print(f"{subject:30s} Accuracy: {acc:.3f}")

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

if all_ttft_times:
    avg_ttft = np.mean(all_ttft_times)
    p50_ttft = np.median(all_ttft_times)
    p95_ttft = np.percentile(all_ttft_times, 95)
    print(f"Time To First Token (TTFT):")
    print(f"  Average: {avg_ttft:.4f} seconds")
    print(f"  Median (P50): {p50_ttft:.4f} seconds")
    print(f"  P95: {p95_ttft:.4f} seconds")

if all_tpot_times:
    avg_tpot = np.mean(all_tpot_times)
    p50_tpot = np.median(all_tpot_times)
    p95_tpot = np.percentile(all_tpot_times, 95)
    print(f"\nTime Per Output Token (TPOT):")
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