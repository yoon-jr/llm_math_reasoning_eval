'''
코드 실행 방법:
(사전)
pip install torch torchvision torchaudio 
pip install transformers accelerate pillow 

(실행)
.venv/bin/python3 main_openllm.py --model '모델명' --data 'json 데이터 파일명' --out_dir '결과 저장할 폴더(./results/모델)' --max-new-tokens 512 --temperature 0.7 --top_p 0.9 --top_k 40

thinking 모드를 켜고 싶다면 .venv/bin/python3 main_openllm.py --model '모델명' --data 'json 데이터 파일명' --out_dir '결과 저장할 폴더(./results/모델)' --max-new-tokens 512 --temperature 0.7 --top_p 0.9 --top_k 40 --thinking
'''

"""
- Models supported:
  - Qwen2-VL-7B
  - Mistral-Pixtral-12B
  - OpenGVLab-InternVL3-8B
  - gemma-3-12B
"""

import argparse, json as jsonlib, os, time, re
from pathlib import Path
from typing import Any, Dict, List

import torch

# 사용 GPU 번호 지정
# 서술형 추론 : 0, 그림형 추론 : 1, 초등 그림 수학 : 2, 한국형 차트 : 3
import os
TARGET_GPU = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(TARGET_GPU)

# --- 공통 유틸 ---
from _common_eval import is_correct
from _common_log import RunLogger

# ---------- 데이터 로드 ----------
def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(jsonlib.loads(line))
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = jsonlib.load(f)
            if isinstance(payload, list):
                data = payload
            else:
                raise ValueError("JSON top-level must be a list")
    else:
        raise ValueError("Only .json or .jsonl supported")
    return data

# ---------- 프롬프트 ----------
PROMPT_KO = '''당신은 수학 전문가입니다. 주어진 문제를 단계별로 해결하세요.
- 조건과 알려진 값을 식으로 정리하세요.
- 필요한 공식을 적용해 풀이하세요.
- 단계별로 계산하고 중간 결과를 적으세요.
- 마지막에 정답이 맞는지 다시 검산하세요.
- 출력은 반드시 JSON 형식만 작성해야 합니다. 설명, 불필요한 텍스트, "출력:" 같은 접두어는 절대 포함하지 마세요.
- JSON은 반드시 { 로 시작하고 } 로 끝나야 하며, 키는 아래와 같이 두 개만 사용하세요:
{
"StepByStepSolution": "1) ... 2) ... 3) ...",
"FinalAnswer": <정답>
}
- FinalAnswer는 반드시 정수만 출력하세요. 단위(예: km, 개), null, None, 빈 문자열, 불가능 같은 값은 절대 쓰지 마세요.
- 출력은 반드시 한국어로 작성하세요.
- FinalAnswer이 나왔다면 생성을 중단하고 출력하세요.
- 풀이가 틀렸다고 생각되면 이전 단계로 돌아가 다시 생각하여 정확하게 계산하고 반드시 정수 FinalAnswer를 출력하세요.
- 풀이 과정은 논리적으로 생각하되, 풀이 과정 생성은 최대한 간결하게 핵심만 작성하세요.
- 근거 없는 추정은 하지 말고 반드시 주어진 정보를 토대로 풀이하세요.
- 새로운 문제를 만들어내지 말고, 주어진 문제만 풀어야 합니다.
'''

# ---------- 정수 추출 ----------
def extract_final_answer_as_int(text_out: str) -> int | None:
    """모델 출력에서 FinalAnswer 정수만 추출"""
    try:
        cand = text_out.strip()
        if cand.startswith("```"):
            cand = re.sub(r"^```(?:json)?", "", cand).strip()
            if cand.endswith("```"): cand = cand[:-3].strip()
        parsed = jsonlib.loads(cand)
        for k in ["FinalAnswer","final_answer","answer","final","Final"]:
            if k in parsed:
                return int(parsed[k])
    except Exception:
        pass
    lines = [ln for ln in text_out.split("\n") if ln.strip()]
    if lines:
        match = re.search(r"-?\d+", lines[-1])
        if match:
            return int(match.group())
    return None

def extract_pred_raw(text_out: str) -> str:
    """모델 출력에서 FinalAnswer 문자열로 추출"""
    cand = text_out.strip()
    if cand.startswith("```"):
        cand = re.sub(r"^```(?:json)?", "", cand).strip()
        if cand.endswith("```"):
            cand = cand[:-3].strip()
    try:
        parsed = jsonlib.loads(cand)
        for k in ["FinalAnswer", "final_answer", "answer", "final", "Final"]:
            if k in parsed:
                return str(parsed[k])
    except Exception:
        pass
    lines = [ln.strip() for ln in cand.split("\n") if ln.strip()]
    for ln in reversed(lines):
        match = re.search(r"-?\d+", ln)
        if match:
            return match.group()
    return ""

# ---------- 메인 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[
        "Qwen3-8B",
        "Mistral-7B-Instruct-v0.2",
        "gemma-3-12B",
    ])
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--thinking", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    logger = RunLogger(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    records = load_json_or_jsonl(Path(args.data))

    # ---------------- 모델별 처리 ----------------
    if args.model == "Qwen3-8B":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_id = "Qwen/Qwen3-8B"
        tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(hf_id, device_map="auto").eval()

        for i, ex in enumerate(records, 1):
            ex_id = ex.get("id", i)
            problem = ex["problem"]
            gold = ex["answer"]

            messages = [
                {"role": "system", "content": PROMPT_KO},
                {"role": "user", "content": problem}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.thinking)
            inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)

            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            dt = time.perf_counter() - t0

            trimmed = out_ids[:, inputs["input_ids"].shape[1]:]
            out_toks = trimmed.shape[-1]
            text_out = tokenizer.decode(trimmed[0], skip_special_tokens=True)

            pred_raw = extract_pred_raw(text_out)
            correct = is_correct(pred_raw, gold)

            logger.log_generation(ex_id=ex_id, problem=problem,
                                  gold_answer=gold, model_output=text_out)
            logger.log_sample_metrics(ex_id=ex_id, latency_sec=dt, output_tokens=out_toks, correct=correct)

            if torch.cuda.is_available():
                logger.update_peak_mem(torch.cuda.max_memory_allocated(torch_device))

            print(f"[Qwen3-8B {i}/{len(records)}] id={ex_id} latency={dt:.2f}s tokens={out_toks} correct={correct}")

    elif args.model == "Mistral-7B-Instruct-v0.2":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_id = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(hf_id, device_map="auto").eval()

        for i, ex in enumerate(records, 1):
            ex_id = ex.get("id", i)
            problem = ex["problem"]
            gold = ex["answer"]

            messages = [
                {"role": "system", "content": PROMPT_KO},
                {"role": "user", "content": problem}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)

            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            dt = time.perf_counter() - t0

            trimmed = out_ids[:, inputs["input_ids"].shape[1]:]
            out_toks = trimmed.shape[-1]
            text_out = tokenizer.decode(trimmed[0], skip_special_tokens=True)

            pred_raw = extract_pred_raw(text_out)
            correct = is_correct(pred_raw, gold)

            logger.log_generation(ex_id=ex_id, problem=problem,
                                  gold_answer=gold, model_output=text_out)
            logger.log_sample_metrics(ex_id=ex_id, latency_sec=dt, output_tokens=out_toks, correct=correct)

            if torch.cuda.is_available():
                logger.update_peak_mem(torch.cuda.max_memory_allocated(torch_device))

            print(f"[Mistral-7B {i}/{len(records)}] id={ex_id} latency={dt:.2f}s tokens={out_toks} correct={correct}")

    elif args.model == "gemma-3-12B":
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        hf_id = "google/gemma-3-12b-it"
        processor = AutoProcessor.from_pretrained(hf_id, padding_side="left")
        model = Gemma3ForConditionalGeneration.from_pretrained(hf_id, device_map="auto").eval()

        for i, ex in enumerate(records, 1):
            ex_id = ex.get("id", i)
            problem = ex["problem"]
            gold = ex["answer"]

            messages = [
                {"role": "system", "content": PROMPT_KO},
                {"role": "user", "content": problem}
            ]
            
            prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
            inputs = processor(text=[prompt], return_tensors="pt", padding=True)
            inputs = {k: v.to(torch_device) for k, v in inputs.items()}

            t0 = time.perf_counter()
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            dt = time.perf_counter() - t0

            trimmed = out_ids[:, inputs["input_ids"].shape[1]:]
            out_toks = trimmed.shape[-1]
            text_out = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

            pred_raw = extract_pred_raw(text_out)
            correct = is_correct(pred_raw, gold)

            logger.log_generation(ex_id=ex_id, problem=problem,
                                  gold_answer=gold, model_output=text_out)
            logger.log_sample_metrics(ex_id=ex_id, latency_sec=dt, output_tokens=out_toks, correct=correct)

            if torch.cuda.is_available():
                logger.update_peak_mem(torch.cuda.max_memory_allocated(torch_device))

            print(f"[Gemma3-PT {i}/{len(records)}] id={ex_id} latency={dt:.2f}s tokens={out_toks} correct={correct}")

    logger.finalize(model_name=args.model)

if __name__ == "__main__":
    main()