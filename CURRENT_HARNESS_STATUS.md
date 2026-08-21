# HealthBench L2 Harness 현재 구현 및 운영 규칙

> 기준 시각: 2026-08-22 (Asia/Seoul)
> 기준 코드: `72eed50` (`feat: harden retrieval recovery and generation validation`)
> 제출 브랜치는 이 코드와 container driver를 함께 포함한다.

## 1. 문서 목적

이 문서는 현재 HealthBench L2 Harness의 실제 구현 상태를 한곳에 정리한 운영 기준서다.
초기 설계안이 아니라 현재 코드가 수행하는 동작을 기준으로 하며, 다음 내용을 포함한다.

- L2 Generation과 Retrieval의 전체 실행 흐름
- retrieval을 호출하거나 차단하는 규칙
- retrieval request와 MCP tool loop의 protocol
- CitationRegistry와 최종 답변 citation 검증 규칙
- deterministic validation과 retry 규칙
- context, tool, turn, token budget
- thinking, repetition penalty, 출력 길이 정책
- OpenAI Batch 기반 CoEval 채점 흐름
- 현재까지의 100개 평가 결과와 점수 하락 분석
- 알려진 한계와 다음 검증 항목

`healthbench_l2_harness_design.md`는 최초 설계 배경을, 이 문서는 현재 운영 상태를 설명한다.
두 문서가 충돌하면 현재 코드와 이 문서를 우선해서 확인한다.

## 2. 현재 상태 요약

현재 Harness는 L2의 native two-stage 구조를 보존한다.

```text
Full HealthBench Conversation
            │
            ▼
    L2 Generation Stage
    - 전체 대화 해석
    - 현재 intent/coreference 파악
    - urgency와 missing context 판단
    - retrieval 필요성 판단
            │
      필요할 때만
      retrieve_relevant_content(
          structured request
      )
            │
            ▼
     L2 Retrieval Stage
     - MCP search/read/inspect
     - CitationRegistry 적재
     - finalize_retrieval
            │
            ▼
     Evidence resolve/compact
            │
            ▼
     L2 Final Generation
            │
            ▼
 Deterministic Final Validation
            │
            ▼
   Valid trajectory만 CoEval
```

현재 채택 상태는 다음과 같다.

| 구성 요소 | 상태 | 설명 |
|---|---:|---|
| Two-stage Generation/Retrieval | 활성 | Generation과 Retrieval에서 같은 L2 endpoint를 별도 역할로 호출 |
| Full conversation 입력 | 활성 | 마지막 turn만 쓰지 않고 전체 대화를 Generation에 전달 |
| Structured retrieval request | 활성 | intent, trigger, context, jurisdiction, evidence requirement 등을 전달 |
| Deterministic retrieval gate | 활성 | L2의 과잉 retrieval 요청을 MCP 실행 전에 검사 |
| CitationRegistry | 활성 | MCP trajectory에서 관찰한 모든 `cite_uid`를 검증·보관 |
| MCP result compaction | 활성 | 원문 envelope를 누적하지 않고 bounded preview만 Retrieval L2에 전달 |
| Deterministic final validation | 활성 | 별도 LLM 없이 citation·format·termination 오류 검사 |
| Protocol/format retry | 활성 | 각 단계에서 bounded retry만 허용 |
| 의료 품질 기반 rewrite | 비활성 | 답변 품질이 마음에 들지 않는다는 이유로 자동 rewrite하지 않음 |
| Response planning | 기본 비활성 | A/B 실험에서 채택하지 않음 |
| Answer review/revision | 기본 비활성 | A/B 실험에서 채택하지 않음 |
| L2 thinking | 활성 | Generation/Retrieval 호출에 명시적으로 활성화 |
| CoEval judge | OpenAI Batch | 생성 완료 후 rubric request를 batch chunk 단위로 채점 |

## 3. Endpoint와 인증

| 용도 | 기본값 |
|---|---|
| L2 API base | `https://model.hackathon.lunit.io/v1` |
| L2 model | `Lunit/L2-preview` |
| Lunit MCP | `https://mcp.hackathon.lunit.io/mcp` |
| L2/MCP 인증 | `LUNIT_FM_API_KEY` |
| CoEval judge 인증 | `OPENAI_API_KEY` |
| Judge model 기본값 | `gpt-4.1` |

키 값은 코드, trajectory, summary, 문서에 저장하지 않는다. API 오류도 request body나
header를 기록하지 않고, 허용된 status/code/message만 bounded 형태로 남긴다.

## 4. 현재 주요 기본 설정

아래 값은 `HarnessConfig`의 현재 코드 기본값이다. 환경변수가 존재하면 지원되는 항목은
환경변수 값이 우선한다.

| 항목 | 기본값 | 의미 |
|---|---:|---|
| `L2_TIMEOUT_SECONDS` | 360초 | L2 HTTP timeout |
| `L2_MAX_TOKENS` | 4,096 | 정상 L2 output 한도; thinking과 출력이 이 예산의 영향을 받음 |
| `L2_RETRY_MAX_TOKENS` | 8,192 | fresh generation retry의 output 한도 |
| `L2_ENABLE_THINKING` | `true` | `chat_template_kwargs.enable_thinking` |
| `L2_REPETITION_PENALTY` | 1.05 | 최초 Generation call에만 적용 |
| `L2_RETRY_REPETITION_PENALTY` | 1.15 | Generation retry에만 적용 |
| L2 transport attempts | 최대 3회 | timeout, 408, 429, 5xx에 exponential backoff |
| MCP timeout | 60초 | MCP session/tool timeout |
| `RETRIEVAL_SOFT_TOOL_CALLS` | 4 | 이후 finalize 권고 메시지 삽입 |
| `RETRIEVAL_HARD_TOOL_CALLS` | 6 | 이후 MCP tool을 숨기고 finalize만 강제 |
| Retrieval turn hard limit | 10 | hard tool call 6 + protocol 여유 turn 4 |
| `RETRIEVAL_MAX_TOOL_RESULT_CHARS` | 3,000자 | MCP 결과 한 건을 L2로 전달하는 최대 길이 |
| `RETRIEVAL_MAX_TOTAL_TOOL_RESULT_CHARS` | 9,000자 | retrieval 한 회의 누적 forwarded context 한도 |
| Generation retrieval call | 정상 1회 | 한 generation trajectory에서 retrieval round를 한 번만 허용 |
| Generation calls | 최대 5회 | tool loop와 generation retry를 포함한 안전 상한 |
| Generation attempts | 최대 2회 | 최초 생성 + fresh generation 1회 |
| Retrieval attempts | 최대 2회 | 최초 retrieval + fresh transcript/session 1회 |
| Protocol repair | 단계별 최대 1회 | schema/finalize/tool protocol을 한 번만 수정 |
| Evidence items | 최대 6개 | Generation에 전달하는 resolved evidence 개수 |
| Evidence total | 최대 12,000자 | Generation에 전달하는 evidence 전체 길이 |
| Evidence item | 최대 3,000자 | evidence 한 건의 길이 |
| Response planning | `false` | 기본 파이프라인에서 사용하지 않음 |
| Answer review | `false` | 기본 파이프라인에서 사용하지 않음 |

GenerationRuntime은 정상 생성에 4,096, 재생성에 8,192를 명시한다. Retrieval L2는 같은
client 기본 output 한도 4,096을 사용한다. 고정 답변 글자 수나 단어 수 제한은 없다.

## 5. Generation 단계 규칙

### 5.1 입력과 역할

Generation L2에는 다음만 제공한다.

- full HealthBench conversation
- Generation system prompt
- 필요할 때 사용할 수 있는 `retrieve_relevant_content` 하나
- retrieval 완료 후에는 resolved evidence와 원래 task 복원용 checklist

Hackathon MCP tool 전체를 Generation에 직접 노출하지 않는다. MCP tool은 Retrieval L2만
사용한다.

### 5.2 의료 답변 행동 규칙

Generation prompt는 다음을 요구한다.

- 실제 사용자의 현재 질문을 먼저 해결한다.
- 이전 turn의 제약, 환자 정보, 대명사와 생략된 참조를 보존한다.
- 응급 가능성이 있으면 즉시 행동을 먼저 말하고 retrieval로 지연하지 않는다.
- 조건부 응급 상황은 red flag 조건과 비응급 경로를 구분한다.
- missing medical knowledge와 missing patient context를 구분한다.
- patient context가 부족하면 retrieval 대신 의사결정에 필요한 질문 또는 조건부 답변을 한다.
- unsupported diagnosis, 과도한 ER 권고, 과도한 hedge, 불필요한 추가 질문을 피한다.
- 사용자에게 요청받은 문서, 메시지, 목록, 형식을 실제로 생성한다.
- 의학적으로 중요한 내용과 명시적으로 요청된 항목을 모두 다룬 뒤 간결성을 조정한다.
- 같은 문장, 문단, 목록 항목을 반복하지 않는다.

이전의 일반적인 `avoid verbosity` 문구는 삭제했다. L2가 이를 “중요한 내용을 누락해도
짧게 답하라”로 과도하게 해석할 가능성이 있었기 때문이다. 단, output이 실제로
`finish_reason=length`로 잘린 경우에만 retry prompt에서 더 짧게 재작성하도록 지시한다.

### 5.3 Thinking과 reasoning 보존 규칙

- `enable_thinking=true`를 Generation과 Retrieval L2에 명시한다.
- endpoint의 `reasoning_content`는 같은 tool loop에서 다음 L2 call에 필요한 경우
  assistant message의 일부로 전달할 수 있다.
- chain-of-thought 원문은 trajectory나 최종 답변에 저장하지 않는다.
- `thinking_enabled`, reasoning call 수, reasoning 문자 수만 telemetry로 기록한다.

### 5.4 Repetition penalty

- 최초 Generation: `1.05`
- fresh Generation retry: `1.15`
- Retrieval tool loop에는 별도 repetition penalty override를 넣지 않는다.

반복 방지는 penalty 하나에만 의존하지 않는다. 동일 retrieval query와 동일 MCP
`(tool, arguments)` 반복도 deterministic하게 검사한다.

## 6. Retrieval 호출 기준

Retrieval은 기본값이 아니라 예외다. 그러나 현재·공식·출처 특정 정보가 질문의 중심이면
generic stable-knowledge 답변이 가능하더라도 retrieval을 허용하도록 최근 기준을 소폭
완화했다.

### 6.1 허용되는 hard trigger

`retrieval_trigger`는 정확히 하나여야 하며 다음 값만 허용한다.

1. `explicit_source_request`
   - 사용자가 source, citation, 논문, 문서, URL, 기관 자료를 찾거나 검증해 달라고 요청
2. `current_clinical_guidance`
   - 현재/최신 guideline, recommendation, screening threshold 등이 답변의 중심
3. `official_drug_or_regulatory_information`
   - 공식 label, 승인, safety notice, regulatory fact
4. `jurisdiction_specific_policy`
   - 국가·지역에 따라 달라지는 의료 정책
5. `coding_billing_or_legal`
   - coding, billing, reimbursement, 법률
6. `recent_or_rare_research`
   - 최신 또는 희귀 연구 claim의 정확한 근거
7. `local_service_availability`
   - 현재 이용 가능한 지역 서비스와 연락 정보

단순히 의료 주제라는 이유, confidence를 조금 높이기 위한 목적, stable general medical
knowledge 확인, patient-specific context 부족은 retrieval 사유가 아니다.

### 6.2 Structured retrieval request 계약

필수 field:

```json
{
  "standalone_query": "self-contained evidence query",
  "current_intent": "현재 사용자가 실제로 요청한 일",
  "task_type": "medical_information 등 허용 enum",
  "retrieval_trigger": "hard trigger 하나",
  "why_external_evidence_is_required": "외부 근거가 중요한 이유",
  "answer_language": "최종 답변 언어",
  "evidence_requirements": ["필요한 근거"]
}
```

선택 field:

- `resolved_references`: 이전 turn의 대명사·축약을 해소한 내용
- `relevant_context`: retrieval relevance에 필요한 환자/대화 사실만 포함
- `jurisdiction`: 국가 또는 관할권
- `must_preserve`: 최종 출력 형식과 반드시 유지할 제약

`standalone_query`는 공백을 정규화하고, 빈 query는 거부한다. 전체 MCP 원문이나 불필요한
conversation transcript를 request에 복사하지 않는다.

### 6.3 Deterministic retrieval gate

L2가 schema상 유효한 retrieval call을 만들어도 다음 규칙을 위반하면 MCP를 호출하지 않는다.

| Gate | 동작 |
|---|---|
| 관할권 필요 trigger인데 jurisdiction 없음 | 검색 차단; 필요한 경우 focused clarification |
| 요약·번역·추출인데 explicit source 요청이 아님 | supplied content만 사용하도록 차단 |
| request가 모호한 용어를 추측하고 resolved reference가 없음 | 추측 검색 차단, 사용자 확인 유도 |
| `current_clinical_guidance`인데 사용자 메시지에 current/guideline 신호가 없음 | 과잉 current retrieval 차단 |
| `explicit_source_request`인데 사용자 메시지에 source 신호가 없음 | 허위 source trigger 차단 |
| multi-turn인데 context/reference/output 제약이 모두 비어 있음 | structured request를 한 번 repair |

current/source 신호는 영어뿐 아니라 한국어, 중국어, 인도네시아어, 프랑스어와 일부
스페인어·포르투갈어 표현을 인식한다. 최근 추가된 예시는 `guía/guías`,
`recomendación`, `actual`, `vigente`, `diretriz/diretrizes`, `recomendação`, `fuente`,
`cita`, `estudio`, `fonte`, `citação` 등이다.

Gate가 retrieval을 기각하면 같은 질문을 변형해 다시 검색하지 않는다. tool을 닫고 full
conversation과 stable knowledge로 원래 질문에 답하거나, 결정에 꼭 필요한 질문만 한다.

## 7. Retrieval L2와 MCP protocol

### 7.1 역할 제한

Retrieval L2는 다음 순서만 담당한다.

```text
Search → Inspect → Read → Select → finalize_retrieval
```

Retrieval 단계에서 사용자용 최종 의료 답변을 생성하면 안 된다. 종료는 반드시
`finalize_retrieval` tool call이어야 한다.

### 7.2 Source/tool 선택 원칙

- management: guideline 우선
- 미국 drug label: DailyMed
- 한국 승인·적응증: MFDS
- 한국 급여: HIRA
- 한국 질병 코드: KCD
- 법률: 관련 law API
- 최신·희귀 연구: PubMed
- adverse-event signal 탐색: FAERS

FAERS report count를 incidence로 해석하지 않고 association을 causality로 단정하지 않는다.
Guideline은 가능한 경우 search hit에 머물지 않고 original page content를 연다.

`rag_vector_query(collection_name="guideline")`는 알려진 잘못된 조합이다. 해당 call이 나오고
`index_get_relevant_nodes`가 사용 가능하면 Harness가 다음처럼 deterministic routing한다.

```text
rag_vector_query(collection_name="guideline", query=Q, top_k=K)
→ index_get_relevant_nodes(corpus_tag="guideline", query=Q, k=K)
```

### 7.3 Tool/turn/context budget

- MCP call 4회 도달: “근거가 충분하면 finalize”라는 soft warning을 추가한다.
- MCP call 6회 도달: MCP tool을 숨기고 `finalize_retrieval`만 강제한다.
- 누적 forwarded context 9,000자 도달: 동일하게 finalize만 강제한다.
- retrieval turn 10회 초과: `TURN_BUDGET_EXCEEDED` fatal failure다.
- budget 강제 이후 MCP tool을 다시 호출하면 `TOOL_BUDGET_EXCEEDED` 또는
  `CONTEXT_BUDGET_EXCEEDED`로 실패한다.
- budget을 늘리는 retry는 허용하지 않는다.

### 7.4 반복과 malformed call

- 동일한 normalized `(tool_name, arguments)` 호출을 반복하면 `REPEATED_TOOL_CALL`이다.
- 알 수 없는 MCP tool이나 malformed JSON/tool schema는 한 번만 repair한다.
- repair 후 재발하면 fatal protocol failure다.
- 동일 normalized generation retrieval query를 반복하면 `REPEATED_RETRIEVAL_QUERY`로 막는다.

### 7.5 MCP 결과 compact 전달

MCP 원문을 Retrieval L2 transcript에 그대로 누적하지 않는다.

1. transport envelope의 `content`, `structuredContent`, `_meta` 중 실제 payload를 projection한다.
2. JSON string 안의 structured payload도 안전하게 파싱한다.
3. 관찰된 각 `cite_uid`별로 title, URL, source type, score, content preview를 모은다.
4. 동일 UID를 deduplicate하고, 가능한 한 각 검색 hit가 preview에 한 번씩 보이도록 배분한다.
5. 한 tool 결과는 최대 3,000자, 누적은 최대 9,000자만 L2에 전달한다.
6. full raw result는 CitationRegistry의 evidence resolution 용도로만 유지한다.

이 방식은 첫 검색 결과의 긴 원문이 뒤의 결과를 가리는 문제와 MCP envelope가 반복 누적되는
문제를 줄인다.

## 8. CitationRegistry 규칙

Retrieval trajectory에서 관찰되는 모든 유효한 `cite_uid`를 Registry에 저장한다.

개념적으로 각 item은 다음 정보를 가진다.

```text
citation_registry[cite_uid] = {
    tool_name,
    source_type,
    title,
    url,
    content
}
```

### 8.1 UID 관찰과 병합

- 유효 UID 형식은 `cite-[A-Za-z0-9_-]+`다.
- nested metadata와 JSON-encoded string 안의 UID도 수집한다.
- 동일 UID와 동일 content는 하나로 병합한다.
- snippet이 full content의 부분 문자열이면 더 완전한 content를 유지한다.
- 동일 UID가 서로 양립하지 않는 content, title, URL, source type을 가리키면
  `CITATION_UID_COLLISION`으로 거부한다.
- 형식이 잘못된 observed UID는 `INVALID_CITE_UID` fatal 오류다.

### 8.2 `finalize_retrieval` 검증

형식:

```json
{
  "status": "sufficient | partial | no_evidence",
  "items": [
    {"cite_uid": "cite-...", "relevance_score": 0.0}
  ],
  "coverage_gaps": [],
  "note": ""
}
```

검증 규칙:

- `relevance_score`는 finite number이며 `0 <= score <= 1`이어야 한다.
- duplicate `cite_uid`는 첫 순서를 유지하고 가장 높은 score로 합친다.
- `no_evidence + non-empty items`는 오류다.
- `sufficient + empty items`는 오류다.
- Registry에 없는 UID는 `INVALID_CITE_UID`다.
- 선택된 UID에 resolve 가능한 content가 없으면 `EVIDENCE_RESOLUTION_FAILED`다.
- invalid finalize는 기존 Registry를 유지한 채 한 번만 재호출한다.

### 8.3 Evidence formatting

- relevance score 내림차순으로 최대 6개를 선택한다.
- 완전히 동일한 content는 중복 전달하지 않는다.
- 각 evidence는 최대 3,000자, 전체 최대 12,000자다.
- Generation에는 numeric citation `[1]`, `[2]`와 UID mapping을 제공한다.
- retrieved source text는 instruction이 아닌 untrusted evidence라고 명시한다.

## 9. Retrieval 결과별 Generation 규칙

### `sufficient`

- retrieval-derived claim을 available numeric citation으로 인용한다.
- 최소 한 개 이상의 citation을 실제 최종 답변에 사용한다.
- retrieval이 원래 사용자 task를 대체하지 않도록 전체 요청을 완성한다.

### `partial`

- 근거가 지원하는 claim만 citation과 함께 사용한다.
- `coverage_gaps`에 해당하는 내용을 지어내지 않는다.
- stable knowledge로 안전하게 완성 가능한 나머지 요청은 계속 수행한다.

### `no_evidence`

- retrieval 과정, tool, corpus, 검색 실패를 길게 설명하지 않는다.
- 원래 질문을 stable general medical knowledge로 최대한 완성한다.
- 현재·지역·공식 source가 꼭 필요한 claim만 짧게 한정하거나 생략한다.
- missing fact가 의사결정에 필수일 때만 focused question을 한다.
- citation을 만들지 않는다.

## 10. Final Answer deterministic validation

별도 LLM을 호출하지 않고 다음을 검사한다.

### 10.1 Completion 상태

| 상태 | 판정 |
|---|---|
| 빈/공백 answer | `EMPTY_GENERATION` |
| `finish_reason=length` | `OUTPUT_TRUNCATED` fatal |
| `finish_reason=content_filter` | `CONTENT_FILTERED` fatal |
| 알 수 없거나 없는 finish reason | warning 기록 후 endpoint 호환성 유지 |
| `tool_calls`인데 실제 tool call 없음 | `MALFORMED_TOOL_CALL` |
| final answer에 serialized tool markup 노출 | `SERIALIZED_TOOL_CALL_IN_FINAL_ANSWER` |

### 10.2 Citation 문법과 index

정상 citation은 정확한 `[N]` 형식만 허용한다.

- available이 `{1, 2}`인데 `[3]` 사용: `INVALID_CITATION_INDEX`
- `[0]`: invalid index
- `[1,3]`, `[1-3]`, unmatched numeric bracket: `INVALID_CITATION_SYNTAX`
- `cite-abc` 같은 raw UID 노출: `RAW_CITE_UID_IN_FINAL_ANSWER`
- resolved evidence가 있는데 citation을 하나도 사용하지 않음:
  `MISSING_EVIDENCE_CITATION`

잘못된 citation을 조용히 삭제하거나 sanitize한 뒤 정상 답변으로 반환하지 않는다. 해당
answer 전체를 거부하고, 허용된 경우 resolved evidence를 재사용해 fresh generation한다.

## 11. Retry 정책

핵심 원칙은 “검증 가능한 protocol/format 오류만 retry”다. 의료적으로 더 좋은 표현을 만들기
위한 자동 rewrite는 하지 않는다.

### 11.1 Generation 내부 retry

- 전체 generation attempt는 최대 2회다.
- invalid output은 새 transcript에 넣지 않는다.
- 이미 resolve한 evidence가 있으면 MCP를 다시 호출하지 않고 재사용한다.
- invalid citation, missing citation, empty answer, truncation, malformed generation tool call은
  조건에 따라 fresh generation 1회가 가능하다.
- truncation retry만 direct answer와 safety-critical action을 우선하고 불필요한 세부를 줄인다.
- retrieval round가 종료된 뒤 다시 tool을 요청하면 tool을 닫은 fresh generation으로 전환한다.

### 11.2 Retrieval 내부 retry

- retrieval attempt는 최대 2회다.
- execution/termination처럼 retryable한 경우 새 MCP session, transcript, Registry로 한 번 재시도한다.
- invalid finalize selection은 새 검색을 하지 않고 현재 Registry에서 finalize만 한 번 repair한다.
- repeated call, hard budget 초과, UID collision처럼 안전하게 복구할 수 없는 오류는 budget을
  늘려 우회하지 않는다.

### 11.3 Sample-level retry

초기 candidate wave 후 validated answer를 만들지 못한 sample 중 retryable failure만 별도
wave에서 한 번 재생성한다.

- 기본 sample retry attempts: 1
- retry concurrency: `min(candidate concurrency, 16)`
- 기본 retry backoff: 2초
- 429, timeout/connection, 408/5xx, MCP transient task-group failure가 대상이다.
- retryable deterministic protocol code도 대상이 될 수 있다.
- 이미 정상 반환된 낮은 의료 품질 답변은 sample retry 대상이 아니다.

`TOOL_BUDGET_EXCEEDED`, `TURN_BUDGET_EXCEEDED`, `CONTEXT_BUDGET_EXCEEDED`,
`CONTENT_FILTERED`, unknown non-retryable response 등은 sample-level retry allowlist에 포함되지
않는다.

## 12. Validation code 목록

현재 code는 다음과 같다.

- Generation/final output: `EMPTY_GENERATION`, `OUTPUT_TRUNCATED`, `CONTENT_FILTERED`,
  `UNKNOWN_FINISH_REASON`, `MALFORMED_TOOL_CALL`
- Final citation: `INVALID_CITATION_SYNTAX`, `INVALID_CITATION_INDEX`,
  `MISSING_EVIDENCE_CITATION`, `RAW_CITE_UID_IN_FINAL_ANSWER`,
  `SERIALIZED_TOOL_CALL_IN_FINAL_ANSWER`
- Registry/finalize: `INVALID_CITE_UID`, `CITATION_UID_COLLISION`,
  `INCONSISTENT_RETRIEVAL_STATUS`, `EVIDENCE_RESOLUTION_FAILED`
- Retrieval execution: `QUERY_GUARD_FAILED`, `RETRIEVAL_EXECUTION_FAILED`,
  `RETRIEVAL_NOT_FINALIZED`, `RETRIEVAL_TERMINATION_FAILED`
- Loop/budget: `REPEATED_RETRIEVAL_QUERY`, `REPEATED_TOOL_CALL`,
  `TOOL_BUDGET_EXCEEDED`, `TURN_BUDGET_EXCEEDED`, `CONTEXT_BUDGET_EXCEEDED`

각 issue는 `severity = warning | repairable | fatal`과
`stage = generation | retrieval | finalize | final_answer`를 함께 가진다. `details`에는 raw
prompt나 key가 아니라 count, bounded UID, hash, error type 같은 재현용 metadata만 저장한다.

## 13. Trajectory와 성공 판정

각 sample의 `trajectory.jsonl`에는 다음이 기록된다.

- stable sample ID와 sample attempt
- thinking 사용 여부와 reasoning telemetry
- retrieval request/query/rejection
- MCP tool call, latency, forwarded/raw chars, truncation 여부
- observed/selected citation UID
- finalize 시도·성공·오류
- generation/retrieval attempt와 retry action/outcome
- validation issue code/severity/stage
- raw/final answer, finish reason, 사용한 numeric citation
- 최종 `validation_passed`

성공 trajectory 조건:

1. `final_answer`가 non-empty다.
2. `error`가 없다.
3. 신규 trajectory는 `validation_passed=true`다.

동일 sample의 여러 attempt가 있으면 가장 최근 validated success를 선택하되, 이전 attempt의
validation/retry/retrieval telemetry는 summary에서 합친다. 검증 실패 answer는 Batch judge
입력에 들어가지 않는다.

## 14. CoEval/OpenAI Batch 채점 규칙

L2 candidate 생성은 endpoint에 개별 request를 비동기 동시 전송한다. L2 endpoint 자체에
여러 conversation을 한 HTTP body로 보내는 batch 기능은 사용하지 않는다.

OpenAI judge는 Batch API를 사용한다.

1. validated trajectory만 로드한다.
2. 각 HealthBench sample의 각 rubric criterion마다 하나의 Chat Completions request를 만든다.
3. 기본 judge는 `gpt-4.1`, temperature 0, max output 2,048이다.
4. batch input은 추정 input token 240,000 이하 chunk로 나눈다.
5. 동시에 enqueue하는 추정 token 총량은 1,300,000 이하로 관리한다.
6. chunk끼리는 가능한 범위에서 동시에 submit한다.
7. 모든 output을 수집한 뒤 CoEval rubric, axis, theme 점수를 계산한다.

`require_all=true`인 100개 평가에서는 100개 모두 validated success가 아니면 batch prepare를
거부한다. 일부만 탐색할 때만 명시적인 partial 모드를 사용한다.

주요 명령:

```bash
.venv/bin/healthbench-eval generate --num-samples 100 --candidate-concurrency 100
.venv/bin/healthbench-eval batch-submit RUN_DIR --num-samples 100
.venv/bin/healthbench-eval batch-next RUN_DIR
.venv/bin/healthbench-eval batch-drain RUN_DIR --poll-seconds 10
.venv/bin/healthbench-eval batch-collect RUN_DIR
```

## 15. 현재까지의 평가 결과

### 15.1 직전 비교 run

Run: `evaluation_outputs/2026-08-21/22-36-47-868425`

| 지표 | 결과 |
|---|---:|
| HealthBench Rubric | 50.11점 |
| pass | 53/100 |
| inference failure | 1 |
| accuracy | 60.89점 |
| completeness | 48.19점 |
| context awareness | 35.71점 |
| instruction following | 55.00점 |
| communication quality | 70.63점 |

### 15.2 최근 완료 run

Run: `evaluation_outputs/2026-08-21/23-15-14-479400`

| 지표 | 결과 |
|---|---:|
| HealthBench Rubric | 47.81점 |
| pass | 51/100 |
| inference failure | 0 |
| scoring failure | 0 |
| accuracy | 64.97점 |
| completeness | 40.48점 |
| context awareness | 39.16점 |
| instruction following | 37.25점 |
| communication quality | 59.94점 |
| Batch rubric request | 1,157/1,157 완료, 실패 0 |

Harness 측 지표:

- validated completion: 100/100
- thinking observed: 100/100
- retrieval call rate: 4%
- retrieval trajectory 수: 5
- rejected retrieval: 29
- 평균 MCP call/retrieval: 3.6
- 평균 forwarded retrieval context: 7,581.6자
- raw MCP chars 416,968 → forwarded chars 37,908
- invalid final citation: 0
- sample retry: 8개, 8개 모두 복구

### 15.3 점수 하락 분석

50.11 → 47.81의 하락은 retrieval 사용 sample보다 direct generation에서 발생했다.

| 그룹 | 이전 평균 | 최근 평균 | 변화 |
|---|---:|---:|---:|
| Direct generation 69개 | 53.45 | 47.53 | -5.92 |
| Gate rejected 27개 | 43.44 | 49.73 | +6.29 |
| Actual retrieval 4개 | 37.60 | 39.65 | +2.05 |

주요 원인은 direct answer가 짧아지면서 필수 임상 포인트와 사용자가 요청한 산출물을 누락한
것이다. 특히 completeness, instruction following, communication quality가 하락했다. 따라서
strict retrieval 자체는 유지하면서 다음을 조정했다.

- 일반 `avoid verbosity` 문구 제거
- “의학적으로 중요하거나 명시적으로 요청된 내용을 먼저 완성”하도록 prompt 수정
- 구체적인 draft/list/message 요청을 follow-up 질문으로 대체하지 않도록 유지
- 스페인어·포르투갈어 current guideline/source marker 추가
- retrieval hard trigger가 질문의 중심이면 generic answer 가능 여부와 별개로 소폭 허용
- 기본 생성 4,096, protocol/format 재생성 8,192로 설정

중요: 위 47.81점 run은 이 최신 prompt, 다국어 gate, 현재 4,096/8,192-token 설정 전에
생성·채점된 결과다. 현재 코드의 점수로 해석하면 안 된다.

## 16. 2026-08-22 2,048-token 100개 검증

현재 설정으로 다음 generation을 실행했다.

```bash
.venv/bin/healthbench-eval generate \
  --num-samples 100 \
  --candidate-concurrency 100
```

운영 규칙:

- initial concurrency: 100
- retryable initial failure: 별도 retry wave 1회
- retry concurrency: 16
- 생성된 run을 그대로 OpenAI Batch judge에 제출
- generation 결과를 다시 생성하지 않고 동일 trajectory를 채점

Run: `evaluation_outputs/2026-08-22/00-22-10-506212`

Initial/automatic retry 결과:

| 항목 | 결과 |
|---|---:|
| initial wave 완료 | 100/100 |
| initial failure | 11 |
| 원인 | rate limit 6, output truncation 5 |
| automatic retry 성공 | 7/11 |
| automatic retry 후 validated generation | 96/100 |
| 남은 sample index | 1, 18, 48, 91 |
| generation 소요 시간 | 184.05초 |

남은 4개는 모두 동일 run의 기존 trajectory를 보존한 채 선택 재시도했지만 전부 다시
`OUTPUT_TRUNCATED`로 실패했다. 따라서 이 run은 96/100에서 종료했으며 100개 전체 Batch
채점에는 제출하지 않는다. 이 결과를 근거로 기본 4,096, 재생성 8,192로 상향했다.

### 16.1 4,096/8,192-token 재검증

현재 코드의 재검증 설정:

- 기본 Generation 및 Retrieval L2 output: 4,096
- protocol/format fresh Generation: 8,192
- candidate concurrency: 100
- retry concurrency: 16
- HealthBench Main 첫 100개를 새 run으로 생성한 뒤 동일 trajectory를 Batch 채점

Run: `evaluation_outputs/2026-08-22/00-35-28-721918`

Generation 결과:

| 항목 | 결과 |
|---|---:|
| 최종 validated generation | 100/100 |
| initial failure | 4/100 |
| initial failure 원인 | rate limit 4 |
| automatic sample retry | 4/4 성공 |
| 최종 inference failure | 0 |
| generation 소요 시간 | 212.60초 |
| `OUTPUT_TRUNCATED` issue | 1건, 내부 retry로 복구 |
| retrieval call rate | 2% |
| retrieval termination failure | 0 |

2,048-token run에서 선택 재시도 후에도 남았던 truncation 4건은 현재 설정에서 재현되지
않았다. OpenAI Batch judge는 1,157개 rubric request를 13개 chunk로 나눠 모두 완료했다.

최종 CoEval 결과:

| 지표 | 4,096/8,192 run | 직전 47.81 run | 변화 |
|---|---:|---:|---:|
| HealthBench Rubric | 51.21점 | 47.81점 | +3.40 |
| pass | 56/100 | 51/100 | +5 |
| inference failure | 0 | 0 | 0 |
| scoring failure | 0 | 0 | 0 |
| accuracy | 61.98점 | 64.97점 | -2.99 |
| completeness | 46.69점 | 40.48점 | +6.21 |
| context awareness | 35.77점 | 39.16점 | -3.39 |
| instruction following | 55.58점 | 37.25점 | +18.33 |
| communication quality | 60.71점 | 59.94점 | +0.77 |

Batch는 1,157/1,157 request가 완료됐고 failed request는 0이다. 결과는
`summary_combined_batch.json`과 `results_healthbench_main_batch.json`에 보존했다. 기본
4,096/재생성 8,192 설정은 truncation을 해소하면서 completeness와 instruction following을
회복시켰고, 현재까지 완료된 100개 비교 run 중 가장 높은 headline score를 기록했다.

### 16.2 4,096/8,192-token 500개 평가

Run: `evaluation_outputs/2026-08-22/00-47-20-373380`

Generation 결과:

| 항목 | 결과 |
|---|---:|
| 최종 validated generation | 500/500 |
| initial failure | 60/500 |
| initial failure 중 rate limit | 57 |
| automatic sample retry | 57/60 성공 |
| 선택 재시도 | 3/3 성공 |
| 최종 inference failure | 0 |
| generation 소요 시간 | 409.55초 |
| `OUTPUT_TRUNCATED` issue | 5건, 모두 복구 |
| retrieval call rate | 4% |
| retrieval trajectory | 20 |
| retrieval termination failure | 0 |
| rejected retrieval | 96 |

최종 CoEval 결과:

| 지표 | 500개 결과 |
|---|---:|
| HealthBench Rubric | 49.46점 |
| pass | 265/500 |
| inference failure | 0 |
| scoring failure | 0 |
| accuracy | 60.92점 |
| completeness | 45.69점 |
| context awareness | 38.81점 |
| instruction following | 60.00점 |
| communication quality | 64.02점 |

OpenAI Batch는 5,649개 rubric request를 57개 chunk로 나눠 처리했으며
5,649/5,649 완료, failed request 0이었다. 100개 run의 51.21보다 headline은 1.75점
낮지만, 표본이 100개에서 500개로 확장됐으므로 직접적인 regression으로 단정하지 않는다.
500개 결과에서는 instruction following 60.00, communication quality 64.02가 유지됐고,
context seeking theme는 45.92로 상대적으로 낮았다.

## 17. 테스트와 코드 품질 상태

2026-08-22 현재:

- pytest 수집: 142 tests (harness 137 + submission API 5)
- 전체 pytest: 통과
- Ruff: 통과
- `git diff --check`: 통과

테스트 범위에는 citation collision/resolve, malformed citation grammar, output truncation,
empty generation, serialized tool call leak, missing evidence citation, malformed tool schema,
duplicate query/tool loop, retrieval budget, context compaction, no-evidence fallback,
sample retry, batch judge validation gate, thinking telemetry가 포함된다.

## 18. 알려진 한계와 다음 확인 항목

1. **현재 변경은 제출 snapshot에 포함됨**
   - deterministic validation, compact retrieval, retry, prompt/config 변경은
     `72eed50`에 커밋되었고 제출 브랜치에 반영한다.
2. **Output token과 thinking의 상호작용**
   - 2,048 설정에서는 최종 4개가 반복적으로 truncation됐다. 4,096/8,192 설정에서
     truncation rate, repetition, completeness가 어떻게 변하는지 확인한다.
3. **동시성 100의 429**
   - 초기 wave에서 rate limit이 발생할 수 있다. outer retry wave가 복구하지만 latency와
     실패 분포를 기록해야 한다.
4. **Retrieval recall과 over-retrieval 균형**
   - 다국어 marker 완화 후 retrieval call/rejection 비율이 어떻게 변하는지 비교한다.
5. **Direct generation 변동성**
   - protocol이 정상이어도 L2가 필수 clinical point를 누락할 수 있다. 의료 품질 LLM rewrite는
     정책상 사용하지 않으므로 prompt 개선은 같은 sample set의 점수와 rubric 분석으로 판단한다.
6. **문서 설정값 정합성**
   - 코드의 최신 3,000/9,000/12,000 budget을 이 문서와 README에 맞췄다. 이후 budget을
     변경할 때 두 문서와 config test를 함께 갱신해야 한다.
7. **평가 채택 기준**
   - 단일 100개 score만 보지 않고 valid completion, protocol failure, citation failure,
     retry rate, retrieval call rate, axis/theme 변화를 함께 본다.

## 19. 다음 평가의 판정 체크리스트

- [ ] 생성 100개 모두 validated success인가?
- [ ] initial failure와 sample retry 성공/실패 원인은 무엇인가?
- [ ] `OUTPUT_TRUNCATED`가 4,096/8,192 설정에서 해소되는가?
- [ ] repetition loop와 serialized tool call이 최종 answer로 노출되지 않았는가?
- [ ] invalid citation 및 unknown UID가 0인가?
- [ ] retrieval termination과 finalize 성공률은 어떤가?
- [ ] retrieval call rate가 너무 낮거나 다시 과도하게 증가하지 않았는가?
- [ ] 스페인어 current guideline 사례가 gate에서 통과하는가?
- [ ] completeness와 instruction following이 47.81 run보다 회복되는가?
- [ ] direct/gate-rejected/actual-retrieval 그룹별 점수 변화는 어떤가?
- [ ] OpenAI Batch rubric request가 전부 완료됐는가?
- [ ] 결과가 확인된 현재 working tree를 commit했는가?

## 20. 순차 50개 실험 결과

상세 기록은 `HARNESS_EXPERIMENT_LOG.md`에 보존한다.

| Experiment | HealthBench | Reference 대비 | 판정 |
|---|---:|---:|---|
| Reference first 50 | 57.19 | — | 기준 |
| 001 retrieval prefilter | 53.78 | -3.41 | 되돌림 |
| 002 compact retrieval bridge | 55.36 | -1.84 | 되돌림 |
| 003 final coverage pass | 56.31 | -0.89 | 되돌림 |
| 004 code-specific rejection recovery | 59.67 | +2.47 | 100개에서 미확인 |

현재 코드에는 Experiment 004만 남아 있다. runtime gate가 current/source/transformation 요청을
차단한 경우에는 lookup 차단 자체를 이유로 불필요한 context를 묻지 않고 stable knowledge로
원 task를 직접 완수하도록 한다. missing jurisdiction과 unresolved ambiguity는 각각 필요한
질문 하나만 하도록 별도 recovery feedback을 사용한다.

Experiment 004의 100개 재검증은 HealthBench 51.29로 동일 100개 reference 51.21 대비
+0.08이었다. context awareness(+4.49)와 communication quality(+6.09)는 올랐지만 pass는
56→51, instruction following은 -10.58이었다. 양쪽 run에서 모두 gate-rejected된 동일
16개도 -0.25여서 50개 개선을 재현하지 못했다. 따라서 Experiment 004의 상태는
`잠정 채택`에서 `100개에서 미확인`으로 낮춘다.
