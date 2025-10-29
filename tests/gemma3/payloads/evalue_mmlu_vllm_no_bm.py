import re
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
import numpy as np
import os
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

# -------------------------------
# RUN EVALUATION
# -------------------------------
results = {}
overall_correct, overall_total = 0, 0

for subject in subjects:
    data = mmlu_test.filter(lambda x: x["subject"] == subject)
    # data = mmlu[dataset_name_to_use]["subject"]
    # list(mmlu[dataset_name_to_use]["subject"].keys())
    prompts = [format_prompt(x, subject) for x in data]

    print(f"\nEvaluating {subject} ({len(prompts)} samples)...")
    subject_correct = 0

    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        outputs = llm.generate(batch_prompts, sampling_params)

        for j, out in enumerate(outputs):
            output_text = out.outputs[0].text.strip()
            pred_choice = extract_choice(output_text)
            gold_idx = data[i+j]["answer"]
            gold_choice = chr(ord("A") + gold_idx)
            if pred_choice == gold_choice:
                subject_correct += 1

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
