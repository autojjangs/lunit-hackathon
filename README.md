# L2 HealthBench Harness

L2의 native two-stage protocol을 보존하면서 Lunit MCP evidence를 선택해
[CoEval](https://github.com/lunit-io/CoEval)의 HealthBench Main 5,000건을 평가하는
Python harness입니다.

## Setup

Python 3.12와 `uv`가 필요합니다.

```bash
uv sync --dev
```

키는 저장소나 `.env`에 커밋하지 말고 평가를 시작하는 셸에만 설정합니다.

```bash
export LUNIT_FM_API_KEY="lunit_..."
export OPENAI_API_KEY="sk-..."
```

기본 endpoint는 다음과 같으며 환경변수로 변경할 수 있습니다.

| Environment | Default | Purpose |
|---|---|---|
| `L2_API_BASE` | `https://model.hackathon.lunit.io/v1` | Generation/Retrieval L2 |
| `L2_MODEL` | `Lunit/L2-preview` | L2 model ID |
| `L2_MAX_TOKENS` | `4096` | thinking과 최종 답변을 포함한 기본 L2 output limit |
| `L2_RETRY_MAX_TOKENS` | `8192` | protocol/format 재생성 시 L2 output limit |
| `L2_MAX_CONCURRENCY` | `15` | 제출 service에서 동시에 진행할 최대 L2 HTTP 호출 수 |
| `L2_ENABLE_THINKING` | `true` | `chat_template_kwargs.enable_thinking` 명시 설정 |
| `MCP_URL` | `https://mcp.hackathon.lunit.io/mcp` | Lunit MCP |
| `MCP_MAX_CONCURRENT_SESSIONS` | `8` | 동시에 열 수 있는 retrieval MCP session 수 |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | CoEval gpt-4.1 judge |
| `RETRIEVAL_MAX_TOOL_RESULT_CHARS` | `3000` | L2에 전달할 compact tool preview의 개별 한도 |
| `RETRIEVAL_MAX_TOTAL_TOOL_RESULT_CHARS` | `9000` | retrieval 한 건의 누적 compact preview 한도 |
| `MAX_EVIDENCE_CHARS` | `12000` | Generation L2에 전달할 전체 evidence 한도 |
| `MAX_EVIDENCE_ITEM_CHARS` | `3000` | 선택 evidence 한 건의 한도 |
| `ENABLE_RESPONSE_PLANNING` | `false` | 기각된 실험용 structured planning 단계 |
| `ENABLE_ANSWER_REVIEW` | `false` | 실험용 최종 답변 review/revision 단계 |

## Evaluation

```bash
# 인증, model/tool calling, MCP 21개 tool, judge 접근 확인
uv run healthbench-eval preflight

# HealthBench Main 첫 10건
uv run healthbench-eval smoke

# HealthBench Main 전체 5,000건
uv run healthbench-eval full
```

## Container submission

Repository root의 `Dockerfile`은 Dashboard 제출 규격에 맞춘 OpenAI-compatible
multi-turn service를 구성합니다. 전체 conversation history를 요청마다 전달하며 service
자체는 사용자 session 상태를 저장하지 않습니다.

```bash
docker build -t lunit-healthbench-submission:local .
docker run --rm -p 8000:8000 lunit-healthbench-submission:local
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Lunit/L2-preview","messages":[{"role":"user","content":"Hello"}]}'
```

Container는 worker 1개로 실행되며 L2 요청은 최대 15개, MCP retrieval session은 최대
8개로 제한합니다. Dashboard에는 `lunit/hackathon-submission` branch HEAD의 40자리 SHA와
model name `Lunit/L2-preview`를 입력합니다.

이미 생성된 trajectory를 OpenAI Batch API로 채점하려면 다음 명령을 사용합니다.
Batch는 문항별 rubric을 개별 JSONL 요청으로 만들며 한 파일의 요청은 모두 동일한
모델을 사용합니다.

```bash
uv run healthbench-eval batch-submit evaluation_outputs/YYYY-MM-DD/RUN --num-samples 100
uv run healthbench-eval batch-status evaluation_outputs/YYYY-MM-DD/RUN
# token-limit에 따라 나뉜 pending chunk를 모두 동시에 제출
uv run healthbench-eval batch-next evaluation_outputs/YYYY-MM-DD/RUN
uv run healthbench-eval batch-repartition-pending evaluation_outputs/YYYY-MM-DD/RUN
uv run healthbench-eval batch-drain evaluation_outputs/YYYY-MM-DD/RUN
uv run healthbench-eval batch-collect evaluation_outputs/YYYY-MM-DD/RUN
```

결과는 `evaluation_outputs/YYYY-MM-DD/<run>/` 아래에 저장됩니다.

- `results_healthbench_main.json`: CoEval sample별 결과와 rubric 점수
- `summary_healthbench_main.json`: CoEval headline 및 theme/axis 집계
- `summary_combined.json`: CoEval 통합 요약
- `trajectory.jsonl`: generation/retrieval/tool/citation trajectory
- `harness_summary.json`: retrieval call/차단 비율, trigger/tool 분포, latency, citation 오류

## Behavior

- Generation L2에는 `retrieve_relevant_content`만 노출합니다.
- 모든 Generation/Retrieval L2 호출은 thinking을 명시적으로 활성화합니다. 분리된
  `reasoning_content`는 tool loop 안에서만 전달하고 원문을 trajectory나 최종 답변에
  저장하지 않으며, 사용 여부·호출 수·문자 수 telemetry만 기록합니다.
- response planning과 answer review 실험은 A/B 평가에서 기각되어 기본 비활성화되어
  있습니다. 이전 실험 재현이 필요한 경우에만 명시적으로 활성화합니다.
- Retrieval은 명시적 source 요청, 현재 guideline, 공식 label/regulatory 정보,
  관할권별 policy/coding/billing/legal 정보, 최신·희귀 연구, 현재 local service라는
  hard trigger 중 하나가 답변에 반드시 필요한 경우에만 허용합니다.
- structured request에는 `retrieval_trigger`와
  `why_external_evidence_is_required`가 필수이며, evidence 요구가 비어 있으면
  validation 단계에서 차단합니다.
- 관할권별·규제·local-service 검색에 jurisdiction이 없거나, 명시적 source 요청이
  없는 요약·번역·추출 작업이면 MCP 호출 전에 runtime gate가 차단합니다.
- Retrieval L2에만 MCP tool과 `finalize_retrieval`을 노출합니다.
- MCP tool call은 soft budget 4, hard max 6입니다.
- MCP 원문은 CitationRegistry에만 보존합니다. L2에는 transport envelope와 중복
  payload를 제거한 compact preview만 전달합니다.
- compact preview는 개별 3,000자, 누적 9,000자로 제한하며 누적 한도에 도달하면
  `finalize_retrieval`만 허용합니다.
- endpoint 안정성을 위해 candidate concurrency 기본값은 1입니다.
- `cite_uid`는 실제 MCP result에서 관찰된 값만 선택할 수 있습니다.
- CitationRegistry는 동일 UID의 compatible snippet/full-content를 병합하고, 상충하는
  source/content와 unknown UID를 deterministic protocol 오류로 거부합니다.
- 최종 답변은 resolved evidence로 생성된 `[N]` citation만 사용할 수 있습니다.
  잘못된 index나 citation 문법은 삭제하지 않고 generation validation 실패로 처리합니다.
- empty/잘린 응답, malformed tool call, invalid citation, query/finalize protocol 오류만
  bounded retry하며 일반적인 의료적 품질이나 문체를 이유로 rewrite하지 않습니다.
- 최종 검증에 실패했거나 429·timeout·MCP task-group 같은 transient 실행 오류가 난
  샘플은 초기 wave가 끝난 뒤 별도 큐에서 최대 한 번 fresh regeneration합니다. 재시도
  wave의 기본 동시성은 `min(candidate concurrency, 16)`이며 최초·재시도 trajectory를
  모두 보존합니다.
- 신규 trajectory는 `validation_passed`와 validation/retry/finalize/budget telemetry를
  명시하며, 검증에 실패한 답변은 OpenAI Batch judge 입력에서 제외합니다.
- 2025 HealthBench rubric보다 현재 2026 임상 근거와 환자 안전을 우선합니다.
- retrieved text는 instruction이 아니라 untrusted evidence로 취급합니다.
