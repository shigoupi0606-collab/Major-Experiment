import sys
import os
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_baseline_model(model_path="/root/models/gpt2-xl"):
    """加载原始GPT2-XL模型"""
    print("正在加载原始GPT2-XL模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # GPT-2 没有 pad_token，使用 eos_token 作为 pad_token
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.cuda()
    model.eval()
    print("模型加载完成")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=20):
    """生成模型回答"""
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
    # 返回提示词之后生成的部分
    return response[len(prompt):].strip()


def run_baseline_test():
    """运行基线测试"""
    # 加载模型和分词器
    model, tokenizer = load_baseline_model()

    # 加载测试数据（使用绝对路径或相对路径）
    data_path = "data/custom_test_data.json"
    if not os.path.exists(data_path):
        # 尝试从脚本所在目录的上层目录查找
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "../data/custom_test_data.json")
    with open(data_path, "r") as f:
        test_data = json.load(f)

    results = []
    print("\n" + "=" * 60)
    print("Task 1: 基线测试（编辑前模型表现）")
    print("=" * 60 + "\n")

    for idx, item in enumerate(tqdm(test_data, desc="测试进度")):
        prompt = item["prompt"]
        ground_truth = item["ground_truth"]
        target_new = item["target_new"]

        response = generate_response(model, tokenizer, prompt)

        result = {
            "index": idx,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "target_new": target_new,
            "model_response": response,
            "matches_ground_truth": (ground_truth.lower() in response.lower() if ground_truth else False),
            "matches_target_new": (target_new.lower() in response.lower() if target_new else False)
        }
        results.append(result)

        print(f"\n样本 {idx + 1}:")
        print(f"  提示词: {prompt}")
        print(f"  原始知识: {ground_truth}")
        print(f"  目标知识: {target_new}")
        print(f"  模型回答: {response}")
        print(f"  匹配旧知识: {result['matches_ground_truth']}")
        print(f"  匹配新知识: {result['matches_target_new']}")

    # 保存结果
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/task1_baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 统计
    old_knowledge_match = sum(r["matches_ground_truth"] for r in results)
    new_knowledge_match = sum(r["matches_target_new"] for r in results)

    print("\n" + "=" * 60)
    print("Task 1 基线测试统计")
    print("=" * 60)
    print(f"验证模型存在知识盲区的样本数（匹配旧知识）: {old_knowledge_match}/{len(results)}")
    print(f"验证模型不存在新知识的样本数（匹配新知识）: {new_knowledge_match}/{len(results)}")
    print(f"基线测试结果已保存至: {output_path}")
    return results


if __name__ == "__main__":
    run_baseline_test()