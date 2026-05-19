import sys
import os
import json
import time
import gc
import torch
import psutil
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams

def get_gpu_memory_info():
    """获取GPU显存使用信息（已用MB）"""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
        capture_output=True, text=True
    )
    used = int(result.stdout.strip())
    return used

def load_zsre_dataset(file_path, limit=500):
    """加载ZsRE数据集，返回编辑请求列表"""
    with open(file_path, "r") as f:
        data = json.load(f)
    edit_requests = []
    for item in data[:limit]:
        if "requested_rewrite" in item:
            req = item["requested_rewrite"]
            edit_requests.append({
                "prompt": req.get("prompt", ""),
                "target_new": req.get("target_new", {}).get("str", "") if isinstance(req.get("target_new"), dict) else req.get("target_new", ""),
                "ground_truth": req.get("target_true", {}).get("str", "") if isinstance(req.get("target_true"), dict) else req.get("target_true", ""),
                "subject": req.get("subject", "")
            })
        else:
            edit_requests.append({
                "prompt": item.get("prompt", ""),
                "target_new": item.get("target_new", ""),
                "ground_truth": item.get("ground_truth", ""),
                "subject": item.get("subject", "")
            })
    print(f"加载了 {len(edit_requests)} 条编辑请求")
    return edit_requests

def run_memit_batch_edit():
    # 加载数据集
    dataset_path = "./data/ZsRE-test-all.json"
    if not os.path.exists(dataset_path):
        print(f"警告：数据集文件不存在 - {dataset_path}")
        return
    edit_requests = load_zsre_dataset(dataset_path, limit=500)

    # 记录开始时间与资源
    start_time = time.time()
    gpu_mem_start = get_gpu_memory_info()
    cpu_mem_start = psutil.virtual_memory().used / (1024**2)  # MB

    print("加载原始模型...")
    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/gpt2-xl.yaml')
    editor = BaseEditor.from_hparams(hparams)

    prompts = [req["prompt"] for req in edit_requests]
    target_new_list = [req["target_new"] for req in edit_requests]
    ground_truth_list = [req["ground_truth"] for req in edit_requests]
    subjects = [req["subject"] if req["subject"] else req["prompt"].split()[-1] for req in edit_requests]

    print("开始 MEMIT 批量编辑...")
    edit_start = time.time()
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth_list,
        target_new=target_new_list,
        subject=subjects,
        sequential_edit=False
    )
    edit_end = time.time()

    # 资源统计
    gpu_mem_end = get_gpu_memory_info()
    gpu_peak = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    cpu_mem_end = psutil.virtual_memory().used / (1024**2)
    gpu_mem_increase = gpu_mem_end - gpu_mem_start
    cpu_mem_increase = (cpu_mem_end - cpu_mem_start) / 1024  # GB

    # 保存模型
    save_path = "./outputs/memit_edited_model"
    os.makedirs(save_path, exist_ok=True)
    edited_model.save_pretrained(save_path)
    # 同时保存tokenizer（editor中应包含tokenizer，但BaseEditor可能没有直接暴露，这里从模型获取）
    if hasattr(editor, 'tok'):
        editor.tok.save_pretrained(save_path)

    print("\n批量编辑完成！")
    print(f"耗时: {edit_end - edit_start:.2f} 秒")
    print(f"GPU 显存峰值: {gpu_peak:.2f} MB")
    print(f"GPU 显存增加: {gpu_mem_increase:.2f} MB")
    print(f"CPU 内存增加: {cpu_mem_increase:.2f} GB")
    print(f"编辑后的模型已保存至: {save_path}")

    # 验证前5条编辑结果
    tokenizer = editor.tok if hasattr(editor, 'tok') else None
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
        tokenizer.pad_token = tokenizer.eos_token

    print("\n验证前5条编辑结果:")
    edited_model.eval()
    for i in range(min(5, len(prompts))):
        prompt = prompts[i]
        target = target_new_list[i]
        inputs = tokenizer(prompt, return_tensors="pt").cuda()
        with torch.no_grad():
            outputs = edited_model.generate(**inputs, max_new_tokens=20, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        success = target.lower() in response.lower()
        # 截断过长的prompt显示
        prompt_short = prompt if len(prompt) <= 50 else prompt[:47] + "..."
        print(f"{i+1}. {prompt_short} -> {response[:30]} (预期:{target}) {'✓' if success else '✗'}")

    # 保存统计
    stats = {
        "num_edits": len(prompts),
        "time_seconds": edit_end - edit_start,
        "gpu_memory_peak_mb": gpu_peak,
        "gpu_memory_increase_mb": gpu_mem_increase,
        "cpu_memory_increase_gb": cpu_mem_increase,
    }
    with open("outputs/memit_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\n性能统计已保存至 outputs/memit_stats.json")

if __name__ == "__main__":
    run_memit_batch_edit()