'''
코드 실행 방법:
(사전)
pip install torch torchvision torchaudio 
pip install transformers accelerate pillow 

(실행)
.venv/bin/python3 main_openllm.py --model '모델명' --data 'json 데이터 파일명' --out_dir '결과 저장할 폴더(./results/모델)' --max-new-tokens 512 --temperature 0.7 --top_p 0.9 --top_k 40
'''

"""
- Models supported:
        gpt-4.1
        gpt-4o-mini
        gemini-2.5-flash
        claude-sonnet-4
"""

import os
import argparse
import json as jsonlib
import time
from pathlib import Path
import re
import torch

# 사용 GPU 번호 지정
# 서술형 추론 : 0, 그림형 추론 : 1, 초등 그림 수학 : 2, 한국형 차트 : 3
import os
TARGET_GPU = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(TARGET_GPU)

# --- API 라이브러리 ---
import openai
import anthropic
#import google.generativeai as genai
from google import genai
from google.genai import types

# --- 공통 유틸 ---
from _common_eval import is_correct
from _common_log import RunLogger

# ---------- 데이터 로드 ----------
def load_json_or_jsonl(path: Path):
    data = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
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
def extract_pred_raw(text_out: str) -> str:
    cand = text_out.strip()
    if cand.startswith("```"):
        cand = re.sub(r"^```(?:json)?", "", cand).strip()
        if cand.endswith("```"):
            cand = cand[:-3].strip()
    try:
        parsed = jsonlib.loads(cand)
        for k in ["FinalAnswer","final_answer","answer","final","Final"]:
            if k in parsed:
                return str(parsed[k])
    except Exception:
        pass
    # fallback: 마지막 줄 중 숫자 포함
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
        "gpt-4.1", "gpt-4o-mini",
        "gemini-2.5-flash",
        "claude-sonnet-4"
    ])
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    # === API 키 설정 ===
    openai.api_key = os.getenv("OPENAI_API_KEY") 
    #genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    anthropic_client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    out_dir = Path(args.out_dir)
    logger = RunLogger(out_dir)

    records = load_json_or_jsonl(Path(args.data))

    for i, ex in enumerate(records, 1):
        ex_id = ex.get("id", i)
        problem = ex["problem"]
        gold = ex["answer"]

        messages = [
            {"role": "system", "content": PROMPT_KO},
            {"role": "user", "content": problem}
        ]

        text_out = ""
        t0 = time.perf_counter()

        # --- 공통 빈 문제 체크 ---
        if not problem.strip():
            print(f"⚠️ 입력이 비어 있음 (id={ex_id}) → 건너뜀")
            text_out = ""  # 또는 "(empty input)" 같은 기본 문자열

        # --- OpenAI GPT ---
        elif args.model in ["gpt-4.1", "gpt-4o-mini"]:
            resp = openai.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                response_format={"type": "json_object"}
            )
            text_out = resp.choices[0].message.content
            if not text_out:
                    print(f"⚠️ gpt 출력이 없거나 차단됨 (id={ex_id})")

        # --- Gemini (Structured Output) ---
        elif args.model == "gemini-2.5-flash":
            client = genai.Client(
                api_key=os.environ["GEMINI_API_KEY"]
            )

            # --- prompt 합치기 ---
            prompt_text = f"{PROMPT_KO}\n\n문제:\n{problem}"
            parts = [prompt_text] 

            def _extract_text(resp) -> str:
                texts = []
                for cand in getattr(resp, "candidates", []) or []:
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    if hasattr(content, "parts"):
                        for p in content.parts:
                            if hasattr(p, "text") and p.text:
                                texts.append(p.text)
                return "\n".join(texts).strip()

            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=parts,
                    config=types.GenerateContentConfig(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_output_tokens=args.max_new_tokens,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_budget=0),  # <-- Thinking 모드 OFF
                    ),
                )
                text_out = _extract_text(resp)
                if not text_out:
                    print(f"⚠️ Gemini 출력이 없거나 차단됨 (id={ex_id})")

            except Exception as e:
                print(f"⚠️ Gemini 호출 중 오류 발생 (id={ex_id}): {type(e).__name__}: {e}")
                text_out = ""

        # --- Claude ---
        elif args.model == "claude-sonnet-4":
            resp = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                system=PROMPT_KO,
                messages=[{"role": "user", "content": problem}],
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            text_out = "".join(block.text for block in resp.content if block.type == "text")
            if not text_out:
                print(f"⚠️ claude 출력이 없거나 차단됨 (id={ex_id})")

        dt = time.perf_counter() - t0

        pred_raw = extract_pred_raw(text_out)
        correct = is_correct(pred_raw, gold)

        logger.log_generation(ex_id=ex_id, problem=problem,
                              gold_answer=gold, model_output=text_out)
        logger.log_sample_metrics(ex_id=ex_id, latency_sec=dt, output_tokens=len(text_out.split()), correct=correct)

        print(f"[{args.model} {i}/{len(records)}] id={ex_id} latency={dt:.2f}s tokens={len(text_out.split())} correct={correct}")

    logger.finalize(model_name=args.model)

if __name__ == "__main__":
    main()