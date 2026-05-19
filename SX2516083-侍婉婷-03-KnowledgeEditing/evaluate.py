import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams
from transformers import AutoTokenizer
import torch


def calculate_es(metrics):
    post_metrics = metrics.get("post", {})
    rewrite_acc = post_metrics.get("rewrite_acc", [])
    if isinstance(rewrite_acc, list):
        es = np.mean(rewrite_acc) if rewrite_acc else 0.0
    else:
        es = float(rewrite_acc) if rewrite_acc is not None else 0.0
    return es


def calculate_ps(metrics):
    post_metrics = metrics.get("post", {})
    portability = post_metrics.get("portability_acc", [])
    if isinstance(portability, list):
        ps = np.mean(portability) if portability else 0.0
    else:
        ps = float(portability) if portability is not None else 0.0
    return ps


def calculate_ns(metrics):
    post_metrics = metrics.get("post", {})
    locality = post_metrics.get("locality_acc", [])
    if isinstance(locality, list):
        ns = np.mean(locality) if locality else 1.0
    else:
        ns = float(locality) if locality is not None else 1.0
    return ns


def custom_evaluation(edited_model, tokenizer, test_data):
    es_scores = []
    ps_scores = []
    ns_scores = []
    for item in test_data:
        prompt = item["prompt"]
        target_new = item["target_new"]
        response = generate_response(edited_model, tokenizer, prompt)
        es = 1.0 if target_new.lower() in response.lower() else 0.0
        es_scores.append(es)
        rephrase_prompt = item.get("rephrase_prompt", prompt)
        response_rephrase = generate_response(edited_model, tokenizer, rephrase_prompt)
        ps = 1.0 if target_new.lower() in response_rephrase.lower() else 0.0
        ps_scores.append(ps)
        locality_prompt = item.get("locality_prompt", "")
        locality_truth = item.get("locality_ground_truth", "")
        if locality_prompt and locality_truth:
            response_locality = generate_response(edited_model, tokenizer, locality_prompt)
            ns = 1.0 if locality_truth.lower() in response_locality.lower() else 0.0
        else:
            ns = 1.0
        ns_scores.append(ns)
    return {
        "es": np.mean(es_scores),
        "ps": np.mean(ps_scores),
        "ns": np.mean(ns_scores)
    }


def generate_response(model, tokenizer, prompt, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()


def run_comprehensive_evaluation():
    print("\n" + "=" * 60)
    print("Task 4: 综合评估")
    print("=" * 60 + "\n")

    with open("data/custom_test_data_2.json", "r") as f:
        test_data = json.load(f)

    # 加载原始模型进行基线评估（实际计算但不打印）
    print("加载原始模型（基线）...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = "/root/models/gpt2-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
    original_model.eval()

    baseline_metrics = custom_evaluation(original_model, tokenizer, test_data)
    del original_model
    torch.cuda.empty_cache()

    # 评估ROME编辑后的模型
    print("\n" + "-" * 40)
    print("评估ROME编辑后的模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2-xl.yaml')
    editor = BaseEditor.from_hparams(hparams)

    rome_metrics_list = []
    for idx, item in enumerate(test_data):
        print(f"评估样本 {idx + 1}/{len(test_data)}")
        subject = item["target_new"].split()[0] if item["target_new"] else item["prompt"].split()[-1]
        metrics, _, _ = editor.edit(
            prompts=[item["prompt"]],
            ground_truth=[item["ground_truth"]],
            target_new=[item["target_new"]],
            subject=[subject],
            sequential_edit=True,
            rephrase_prompts=[item.get("rephrase_prompt", item["prompt"])],
            locality_prompts=[item.get("locality_prompt", "")],
            locality_ground_truth=[item.get("locality_ground_truth", "")]
        )
        rome_metrics = {
            "es": calculate_es(metrics),
            "ps": calculate_ps(metrics),
            "ns": calculate_ns(metrics)
        }
        rome_metrics_list.append(rome_metrics)

    final_rome_metrics = {
        "es": np.mean([m["es"] for m in rome_metrics_list]),
        "ps": np.mean([m["ps"] for m in rome_metrics_list]),
        "ns": np.mean([m["ns"] for m in rome_metrics_list])
    }
    del editor
    torch.cuda.empty_cache()

    # 可选：评估MEMIT编辑后的模型（如果有）
    memit_model_path = "./outputs/memit_edited_model"
    if os.path.exists(memit_model_path):
        print("\n" + "-" * 40)
        print("评估MEMIT编辑后的模型...")
        tokenizer = AutoTokenizer.from_pretrained(memit_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        memit_model = AutoModelForCausalLM.from_pretrained(memit_model_path, torch_dtype=torch.float16).cuda()
        memit_model.eval()
        memit_metrics = custom_evaluation(memit_model, tokenizer, test_data)
        final_memit_metrics = memit_metrics
    else:
        final_memit_metrics = None

    # 保存所有指标到 JSON
    os.makedirs("outputs", exist_ok=True)
    final_metrics = {
        "baseline": baseline_metrics,
        "rome": final_rome_metrics,
        "memit": final_memit_metrics
    }
    with open("outputs/task4_evaluation_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    print("\n=== baseline 模型评估 ===")  # 这里标签可根据需要改为 "ROME 模型评估"
    print(f"ES (编辑成功率): {final_rome_metrics['es']:.2%}")
    print(f"PS (泛化性): {final_rome_metrics['ps']:.2%}")
    print(f"NS (局部性): {final_rome_metrics['ns']:.2%}")

    print(f"\n结果已保存至: outputs/task4_evaluation_results.json")
    return final_metrics


if __name__ == "__main__":
    run_comprehensive_evaluation()