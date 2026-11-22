import os
import time
import json
import re
import psutil
from openai import OpenAI

# ===== Hugging Face API 토큰 세팅 =====
# ⚠️ 아래에 본인 Hugging Face 토큰 입력
HF_TOKEN = os.environ["HF_TOKEN"]

# ===== Hugging Face API 클라이언트 생성 =====
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"]
)

# ===== 문제 데이터 로드 =====
with open("GSM8k_sample.json", "r", encoding="utf-8") as f:
    problems = json.load(f)

# ===== 평가용 변수 초기화 =====
correct_count = 0
latencies = []
total_tokens = 0
results = []
process = psutil.Process(os.getpid())
peak_memory = 0.0

# ===== FinalAnswer 추출 함수 =====
num_pat = re.compile(r"[-+]?\d+(?:\.\d+)?")

def extract_final_answer(text):
    m = re.search(r'\"?FinalAnswer\"?\s*:\s*\"?([^\"]+?)\"?(?:[\n,}])', text)
    if m:
        n = num_pat.search(m.group(1))
        if n: 
            return float(n.group(0))
    nums = num_pat.findall(text)
    if nums:
        return float(nums[-1])
    return None

# ===== 문제별 평가 =====
PROMPT_TEMPLATE = '''당신은 수학 전문가입니다. 주어진 문제를 단계별로 해결하세요.

- 조건과 알려진 값을 식으로 정리하세요.
- 필요한 공식을 적용해 풀이하세요.
- 단계별로 계산하고 중간 결과를 적으세요.
- 마지막에 정답이 맞는지 다시 검산하세요.

출력 형식(JSON):
{{
"StepByStepSolution": "1) ... 2) ... 3) ...",
"FinalAnswer": <정답>
}}

Question: {question}
'''

for item in problems:
    problem = item['problem']
    answer = float(item['answer'])
    prompt = PROMPT_TEMPLATE.format(question=problem)

    start_time = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B:featherless-ai",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=256
    )
    end_time = time.time()

    latency = end_time - start_time
    latencies.append(latency)

    output_text = response.choices[0].message.content
    final_answer = extract_final_answer(output_text)
    is_correct = (final_answer == answer)
    if is_correct: 
        correct_count += 1

    # 토큰 수 대략 계산 (공백 기준)
    completion_tokens = len(output_text.split())
    total_tokens += len(prompt.split()) + completion_tokens

    mem = process.memory_info().rss / (1024*1024)
    peak_memory = max(peak_memory, mem)

    results.append({
        'id': item.get('id'),
        'problem': problem,
        'ground_truth': answer,
        'model_output': output_text,
        'final_answer': final_answer,
        'correct': int(is_correct),
        'latency_sec': latency,
        'memory_mb': mem
    })

# ===== 종합 결과 =====
n = len(problems)
accuracy_percent = (correct_count / n) * 100
avg_latency = sum(latencies)/n
avg_throughput = total_tokens / sum(latencies) if sum(latencies) > 0 else 0

summary = {
    'accuracy_percent': accuracy_percent,
    'avg_latency_sec': avg_latency,
    'avg_throughput_tokens_per_sec': avg_throughput,
    'peak_memory_mb': peak_memory
}

# ===== 결과 저장 =====
with open("eval_results_hf_api.json", "w", encoding="utf-8") as f:
    json.dump({'results': results, 'summary': summary}, f, ensure_ascii=False, indent=4)

print("✅ 완료! eval_results_hf_api.json 저장됨")
print(summary)