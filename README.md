# LLM Math Reasoning Evaluation

수학 문제에 대한 **상용 API 모델과 오픈 모델의 응답·정답 여부·처리 지연을 문항 단위로 기록**한 평가 코드와 실험 결과입니다. 모델별 요약 수치에서 원본 응답과 개별 문항 지표까지 따라갈 수 있도록 구성합니다.

현재 공개된 결과 폴더 8종에는 각각 5,000문항의 지표가 있습니다. 아래 수치는 저장된 실행 기록이며, 동일 하드웨어·동일 출력 예산의 통제된 모델 순위를 뜻하지 않습니다.

## 먼저 볼 자료

- [상용 API 평가 코드](main_commercial_API.py): GPT·Gemini·Claude 호출과 정답 추출
- [오픈 모델 평가 코드](main_openllm.py): 로컬 모델 추론과 평가 기록
- [GPT-4.1 문항별 지표](gpt-4.1/per_sample_metrics.csv) → [원본 응답](gpt-4.1/generations.jsonl) → [요약](gpt-4.1/metrics_summary.json)
- [Gemini thinking-on 응답](gemini_thinking_on/generations.jsonl): 출력 상태·추출 실패 사례를 확인할 수 있는 자료
- [MathVerse 프롬프트 개선 실험](https://github.com/yoon-jr/llm_math_reasoning_grpo): 별도 데이터·실험으로 구분

## 평가 흐름

1. JSON 또는 JSONL에서 문제(`problem`)와 정답(`answer`)을 읽습니다.
2. 단계별 풀이와 최종 정답을 요청하는 프롬프트로 응답을 생성합니다.
3. JSON의 `FinalAnswer` 등에서 정답을 추출하고, 파싱 실패 시 보조 추출을 시도합니다.
4. 정답 비교 결과와 문항별 지연·출력량을 저장합니다.
5. 개별 기록을 집계해 모델별 요약 JSON을 생성합니다.

입력은 [GSM8k_sample.json](GSM8k_sample.json) 등 저장소에 포함된 자료를 사용합니다. 이 평가 집합을 공식 벤치마크 전체 또는 공식 평가 프로토콜과 동일한 것으로 간주하지 않습니다. MathVerse 100문항 실험과도 별개입니다.

## 저장된 평가 결과

| 기록의 모델 표기 | 문항 수 | 전체 정확도 | 요약 파일 |
|---|---:|---:|---|
| Mistral-7B-Instruct-v0.2 | 5,000 | 18.78% | [Mistral](Mistral/metrics_summary.json) |
| Qwen3-8B | 5,000 | 77.94% | [Qwen](Qwen/metrics_summary.json) |
| gemma-3-12B | 5,000 | 79.40% | [gemma](gemma/metrics_summary.json) |
| gpt-4o-mini | 5,000 | 81.48% | [gpt-4o-mini](gpt-4o-mini/metrics_summary.json) |
| gpt-4.1 | 5,000 | 88.82% | [gpt-4.1](gpt-4.1/metrics_summary.json) |
| claude-sonnet-4 | 5,000 | 91.56% | [claude](claude/metrics_summary.json) |
| gemini-2.5-flash · thinking off 기록 | 5,000 | 89.68% | [gemini_thinking_off_with_error](gemini_thinking_off_with_error/metrics_summary.json) |
| gemini-2.5-flash · thinking on 기록 | 5,000 | 69.04% | [gemini_thinking_on](gemini_thinking_on/metrics_summary.json) |

각 수치는 해당 폴더의 `per_sample_metrics.csv` 5,000행을 재집계한 결과와 일치합니다. 모델·thinking 설정 표기는 **저장된 실행 기록의 이름**입니다. 현재 스크립트의 선택지·공급자 설정과 과거 모든 결과가 일대일로 일치한다고 보장하지 않습니다. 특히 `gemini_thinking_off_with_error`는 오류가 포함된 실행 기록임을 폴더 이름에 명시하고 있습니다.

## 지표 정의와 비교 범위

| 지표 | 의미와 주의점 |
|---|---|
| `accuracy_percent` | 기록된 정답 수 / 전체 평가 문항 수 × 100 |
| `average_latency_sec_per_request` | 요청 처리 구간의 평균 경과시간. API 서버·네트워크 지연과 로컬 추론을 같은 조건으로 통제한 값은 아님 |
| `output_tokens` | 상용 API 코드는 `len(text_out.split())`로 계산한 공백 분할 개수. 로컬 토크나이저 토큰 수와 단위가 다름 |
| `throughput_tokens_per_sec` | 위 출력량 정의에 영향을 받으므로 상용/로컬 모델 간 동일 단위 속도 비교에 사용하지 않음 |
| `peak_gpu_memory_gb` | 로컬에서 기록 가능한 메모리. API 결과의 0은 원격 서버 GPU 사용량이 0이라는 의미가 아님 |

빈 응답·정답 추출 실패·형식 오류를 구분하지 않은 정확도만으로 추론 능력의 차이를 단정하지 않습니다. 처리량 비교에는 각 공급자의 실제 출력 토큰 사용량, 실행 환경, 최대 출력 길이를 일치시킨 후속 기록이 필요합니다.

## 결과 확인과 실행 상태

**결과 열람은 바로 가능하지만, 신규 평가 실행 전 공통 모듈 복원이 필요합니다.** 현재 저장소에는 두 main 파일이 import하는 `_common_eval.py`, `_common_log.py`가 포함되어 있지 않습니다. 원래 사용한 정답 비교·로깅 구현을 복원해야 하며, 빈 파일이나 임의의 비교 함수로 대체하면 기존 지표와 달라질 수 있습니다.

주요 의존성은 Python, PyTorch, Transformers, Accelerate, Pillow, OpenAI SDK, Anthropic SDK, Google Gen AI SDK, psutil입니다. 고정 버전의 requirements와 검증된 CUDA 환경은 아직 제공하지 않습니다. [eval.py](eval.py)는 Hugging Face 경유의 별도 스크립트이며 main 파일의 공통 진입점이 아닙니다.

환경변수:

| 변수 | 사용하는 경로 |
|---|---|
| `OPENAI_API_KEY` | GPT 상용 API |
| `ANTHROPIC_API_KEY` | Claude API 및 현재 상용 스크립트의 클라이언트 초기화 |
| `GEMINI_API_KEY` | Gemini API |
| `HF_TOKEN` | `eval.py`의 Hugging Face API |

현재 상용 스크립트는 모델 분기 전에 Anthropic 클라이언트를 생성합니다. GPT만 실행할 때도 초기화 오류가 발생할 수 있으므로, 선택한 공급자만 초기화하도록 정리하거나 해당 초기화 조건을 충족해야 합니다. 키를 소스에 기록하지 않습니다.

공통 모듈·의존성·키 설정을 완료한 뒤 사용할 **상용 평가 명령 예시**입니다. API 이용료가 발생하므로 작은 입력으로 먼저 확인합니다.

```bash
python main_commercial_API.py --model gpt-4.1 --data GSM8k_sample.json --out_dir results/new_gpt41_run --max-new-tokens 256 --temperature 0 --top_p 1
```

입력 형식 예시:

```json
[{"id": 1, "problem": "사과 2개에 3개를 더하면 몇 개인가?", "answer": "5"}]
```

위 명령은 사용법 예시이며, 공개된 5,000문항 결과의 프롬프트·공급자 버전·모든 실행 조건을 완전히 복원한 명령은 아닙니다.

## 코드와 결과의 범위

| 파일·폴더 | 내용 |
|---|---|
| [main_commercial_API.py](main_commercial_API.py) | 상용 모델 선택, 호출, 최종 답 추출 |
| [main_openllm.py](main_openllm.py) | 로컬 오픈 모델 로딩·추론 |
| [eval.py](eval.py) | Hugging Face API 경유 평가 |
| 모델별 폴더 | `generations.jsonl`, `per_sample_metrics.csv`, `metrics_summary.json` |

모델마다 실행 하드웨어·프롬프트·출력 길이·호출일을 연결하는 실험 설정표, 응답 실패 유형, 데이터 출처와 변환 과정은 추가 문서화 대상입니다. 저장소에 포함된 결과는 평가 기록이며, 서비스 배포나 모델 학습 성과를 뜻하지 않습니다.

---
문서 기준: 2026-11-22 · [검토한 코드 버전](https://github.com/yoon-jr/llm_math_reasoning_eval/tree/70b161829804e090994697741ee68b8a03ed19b0). 수치는 저장된 실험 결과이며, 새 실행에서는 환경과 설정에 따라 달라질 수 있습니다.
