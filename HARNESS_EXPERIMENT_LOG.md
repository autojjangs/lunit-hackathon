# HealthBench Harness Experiment Log

이 문서는 하네스 변경을 한 번에 하나씩 적용하고, 매 변경 후 HealthBench Main 첫 50개를
동일한 절차로 생성·채점한 결과를 기록한다. 점수 비교는 같은 50개 sample trajectory를
기준으로 한다.

## 공통 평가 절차

1. 변경 하나를 구현하고 전체 pytest, Ruff, `git diff --check`를 통과시킨다.
2. `healthbench-eval generate --num-samples 50 --candidate-concurrency 32`로 생성한다.
3. 생성된 동일 trajectory를 OpenAI Batch `gpt-4.1` judge로 채점한다.
4. headline, axis, retrieval, validation, retry 지표를 이 문서에 기록한다.
5. 결과를 검토한 뒤 다음 변경으로 진행한다.

## Reference — prefilter 적용 전 동일 50개

Source run: `evaluation_outputs/2026-08-22/00-35-28-721918`

이 run은 100개 평가였으며 아래 값은 그중 sample index 0–49만 다시 집계한 값이다.

| Metric | Result |
|---|---:|
| HealthBench Rubric | 57.19 |
| Passed | 34/50 |
| Accuracy | 59.36 |
| Completeness | 57.06 |
| Context awareness | 47.10 |
| Instruction following | 77.85 |
| Communication quality | 67.42 |
| Actual retrieval samples | 2/50 |
| Gate-rejected samples | 8/50 |

주의: 이전 run은 candidate concurrency 100으로 생성되었으므로 새 실험과 완전히 동일한 실행
조건은 아니다. 점수는 방향성 reference로만 사용하고, 변경 채택은 반복 50개 평가와 더 큰
표본에서 확인한다.

## Experiment 001 — Pre-generation retrieval eligibility

### Change

- deterministic hard-trigger signal이 있는 대화에서만 Generation L2에
  `retrieve_relevant_content`를 노출한다.
- 기존 structured retrieval gate는 tool 호출 후 안전망으로 유지한다.
- eligibility, reason, actual exposure를 trajectory와 harness summary에 기록한다.

### Code state

| Field | Value |
|---|---|
| Date | 2026-08-22 01:33 KST |
| Git HEAD | `1903945` |
| Working-tree diff hash | `558d6405969c2889fade09c76d93d064333faacb` |
| Tests | 142 passed |
| Ruff | passed |
| `git diff --check` | passed |

현재 worktree에는 Experiment 001 이전부터 존재하던 미커밋 validation, retry, prompt, evidence
compaction 변경도 포함되어 있다. 위 hash는 실제 평가에 사용된 전체 working-tree diff를
식별하기 위한 값이다.

### Generation

Run: `evaluation_outputs/2026-08-22/01-29-02-610886`

Command:

```bash
.venv/bin/healthbench-eval generate --num-samples 50 --candidate-concurrency 32
```

| Metric | Result |
|---|---:|
| Valid generation | 50/50 |
| Initial inference failure | 0 |
| Sample retry | 0 |
| Generation time | 208.36 s |
| Retrieval-tool eligible | 12/50 (24%) |
| Retrieval-tool exposed | 12/50 (24%) |
| Actual retrieval | 2/50 (4%) |
| Gate rejection | 0 |
| Retrieval status | `no_evidence` 2 |
| Retrieval protocol failure | 0 |
| Average MCP calls/retrieval | 3.5 |
| Average forwarded context/retrieval | 9,000 chars |
| Truncated MCP results | 6 |
| Generation validation retry | 1 truncation, recovered |

### Preliminary interpretation

- 동일 50개 reference의 gate-rejected sample 8개가 새 run에서는 0개가 되었다.
- 실제 retrieval 수는 2개로 동일하여, prefilter가 필요한 retrieval을 즉시 줄였다는 신호는
  없다.
- 두 retrieval 모두 context budget 9,000자를 전부 사용하고도 `no_evidence`로 종료했다.
  따라서 다음 retrieval 개선에서는 tool masking과 result budget 효율을 별도로 다뤄야 한다.

### Judge

Status: **completed**

OpenAI Batch `gpt-4.1` judge가 539/539 rubric request를 처리했으며 실패는 0건이었다.

| Metric | Reference | Experiment 001 | Delta |
|---|---:|---:|---:|
| HealthBench Rubric | 57.19 | 53.78 | -3.41 |
| Passed | 34/50 | 29/50 | -5 |
| Accuracy | 59.36 | 51.45 | -7.91 |
| Completeness | 57.06 | 55.74 | -1.32 |
| Context awareness | 47.10 | 51.51 | +4.41 |
| Instruction following | 77.85 | 76.19 | -1.66 |
| Communication quality | 67.42 | 61.79 | -5.63 |

### Paired analysis and decision

같은 sample index끼리 비교하면 이전 run에서 retrieval gate가 거절했던 8개는 평균
54.21→47.95(-6.25)였고, 개선 0개·하락 4개·동률 4개였다. 거절되지 않았던 42개도
57.76→54.89(-2.87)로 내려갔다. 평가 실행 조건과 L2 생성의 변동성 때문에 전체 차이를
prefilter 하나에 귀속할 수는 없지만, 목표가 점수 최대화인 상황에서 유지할 근거는 부족하다.

**Decision: not adopted.** gate rejection 8건과 불필요한 tool exposure를 없애는 운영상 효과는
확인했으나, 거절 피드백 뒤 재생성되는 경로가 일부 답변을 교정하는 효과까지 제거했다.
Experiment 002 전에 prefilter 변경만 되돌리고 기존 post-call gate는 유지한다.

## Experiment 002 — Compact L2 retrieval bridge

### Change

- Generation L2가 생성하는 retrieval call의 필수 field를 7개에서
  `standalone_query`, `retrieval_trigger` 두 개로 줄인다.
- `jurisdiction`만 선택 field로 남기고, 현재 의도·task type·이전 user context·답변 언어·
  evidence requirement·보존 제약은 하네스가 full conversation에서 결정적으로 복원한다.
- 내부 `RetrievalRequest`, post-call gate, Retrieval L2 입력 계약은 그대로 유지하고 이전 상세
  tool-call 형식도 호환한다.

### Code state

| Field | Value |
|---|---|
| Date | 2026-08-22 01:45 KST |
| Git HEAD | `1903945` |
| Working-tree diff hash | `0e67177b8d9abd11fdddb3850e1888f1f372a0363514de886885daaff43602fe` |
| Tests | 137 passed |
| Ruff | passed |
| `git diff --check` | passed |

Experiment 001의 prefilter 변경은 제거되었으며, 상세 구조체를 모델이 직접 채우게 하는 부분만
compact bridge로 교체한 상태다.

### Generation

Run: `evaluation_outputs/2026-08-22/01-45-33-125347`

Command:

```bash
.venv/bin/healthbench-eval generate --num-samples 50 --candidate-concurrency 32
```

| Metric | Result |
|---|---:|
| Valid generation | 50/50 |
| Initial inference failure | 0 |
| Sample retry | 0 |
| Generation time | 183.07 s |
| Actual retrieval | 2/50 (4%) |
| Gate rejection | 9 |
| Query/schema guard failure | 0 |
| Retrieval status | `partial` 1, `no_evidence` 1 |
| Retrieval protocol failure | 0 |
| Average MCP calls/retrieval | 3.0 |
| Average forwarded context/retrieval | 9,000 chars |
| Truncated MCP results | 6 |
| Generation validation retry | 1 truncation, recovered |

모델이 상세 conversation state를 직접 생성하지 않아도 두 compact call 모두 내부
`RetrievalRequest`로 정상 확장되었다. 9개 gate rejection은 모두 모델이
`current_clinical_guidance`를 선택했지만 user message에 명시적 current need가 없었던 경우다.

### Judge

Status: **completed**

OpenAI Batch `gpt-4.1` judge가 539/539 rubric request를 처리했으며 실패는 0건이었다.

| Metric | Reference | Experiment 002 | Delta |
|---|---:|---:|---:|
| HealthBench Rubric | 57.19 | 55.36 | -1.84 |
| Passed | 34/50 | 27/50 | -7 |
| Accuracy | 59.36 | 56.44 | -2.92 |
| Completeness | 57.06 | 54.51 | -2.55 |
| Context awareness | 47.10 | 48.49 | +1.39 |
| Instruction following | 77.85 | 63.56 | -14.29 |
| Communication quality | 67.42 | 73.31 | +5.89 |

### Paired analysis and decision

같은 50개 sample index 기준으로 개선 12개·하락 19개·동률 19개였다. Experiment 002에서
gate-rejected된 9개는 reference 58.44→58.87(+0.42)였으나, 나머지 41개가
56.92→54.59(-2.33)였다. 실제 retrieval 2개는 33.18→39.42(+6.24)였지만 표본이 너무 작다.

**Decision: not adopted.** schema/protocol failure 없이 compact call 확장에는 성공했지만,
reference보다 headline과 instruction following이 낮고 retrieval 수는 그대로이며 rejection은
8→9로 늘었다. Experiment 003 전에 compact bridge 변경만 되돌린다.

## Experiment 003 — Final answer coverage pass

### Change

- Generation prompt에 최종 답변 전 수행할 짧은 silent coverage checklist를 추가한다.
- 명시된 모든 질문/산출물, 이미 제공된 환자 context, 현재 행동·monitoring·단계별 escalation,
  관련 medication safety, 필요한 경우의 최소 고가치 질문을 확인한다.
- retrieval schema, runtime gate, token budget, retry 동작은 변경하지 않는다.

기준 50개의 rubric miss를 집계했을 때 누락된 양의 point가 completeness 334,
context awareness 223, accuracy 219 순이어서 가장 큰 두 축을 직접 겨냥했다.

### Code state

| Field | Value |
|---|---|
| Date | 2026-08-22 01:56 KST |
| Git HEAD | `1903945` |
| Working-tree diff hash | `ada7f598a68dd71bb61c99ae0ccb04169dc2f7fa6d5080553eef498730300ca8` |
| Tests | 136 passed |
| Ruff | passed |
| `git diff --check` | passed |

Experiment 001과 002 변경은 모두 제거된 상태다.

### Generation

Run: `evaluation_outputs/2026-08-22/01-56-37-030079`

| Metric | Result |
|---|---:|
| Valid generation | 50/50 |
| Initial inference failure | 1 truncation |
| Sample retry | 1/1 recovered |
| Generation time | 495.19 s |
| Actual retrieval | 2/50 (4%) |
| Gate rejection | 8 |
| Query/schema guard failure | 0 |
| Retrieval status | `no_evidence` 2 |
| Retrieval protocol failure | 0 |
| Average MCP calls/retrieval | 3.0 |
| Average forwarded context/retrieval | 9,000 chars |
| Truncated MCP results | 6 |

sample 48이 첫 시도와 내부 truncation retry 후에도 실패하여 outer sample retry에서
복구되었다. 최종 valid completion은 50/50이나 generation latency는 reference보다 크게 늘었다.

### Judge

Status: **completed**

OpenAI Batch `gpt-4.1` judge가 539/539 rubric request를 처리했으며 실패는 0건이었다.

| Metric | Reference | Experiment 003 | Delta |
|---|---:|---:|---:|
| HealthBench Rubric | 57.19 | 56.31 | -0.89 |
| Passed | 34/50 | 29/50 | -5 |
| Accuracy | 59.36 | 61.65 | +2.29 |
| Completeness | 57.06 | 57.96 | +0.90 |
| Context awareness | 47.10 | 52.72 | +5.62 |
| Instruction following | 77.85 | 76.19 | -1.66 |
| Communication quality | 67.42 | 67.42 | 0.00 |

### Paired analysis and decision

같은 sample index 기준 개선 11개·하락 19개·동률 20개였다. direct/no-rejection 40개는
57.90→58.87(+0.97)로 목표 방향이었지만, 이 run에서 gate-rejected된 8개는
52.60→42.67(-9.93), 개선 0개·하락 4개·동률 4개였다. retrieval 2개는 -1.90이었다.

**Decision: not adopted as a standalone change.** 목표 축 개선은 확인했지만 headline과 pass가
기준보다 낮고 truncation으로 generation latency도 495.19초까지 늘었다. Experiment 004 전에
coverage checklist를 되돌리고, direct path는 건드리지 않은 채 gate rejection feedback만
코드별로 개선한다.

## Experiment 004 — Code-specific gate rejection recovery

### Change

- runtime gate 판정과 retrieval 허용 기준은 바꾸지 않는다.
- `current_need_not_user_requested`, `source_not_user_requested`, transformation rejection에는
  lookup 차단만으로 추가 context를 묻지 말고 원 task를 stable knowledge로 직접 완수하도록
  명시한다.
- missing jurisdiction은 관할권 하나만, unresolved ambiguity는 용어 식별 질문 하나만 묻도록
  rejection code별 recovery feedback을 분리한다.
- language, format, explicit subtasks 보존을 direct-recovery feedback에 포함한다.

### Code state

| Field | Value |
|---|---|
| Date | 2026-08-22 02:11 KST |
| Git HEAD | `1903945` |
| Working-tree diff hash | `f7a05bd8b24dfb27a93f14838598e9c3fdb9fe53a78b38ca36936144615b679b` |
| Tests | 136 passed |
| Ruff | passed |
| `git diff --check` | passed |

Experiment 001–003 변경은 모두 제거된 상태다.

### Generation

Run: `evaluation_outputs/2026-08-22/02-11-40-759008`

| Metric | Result |
|---|---:|
| Valid generation | 50/50 |
| Initial inference failure | 0 |
| Sample retry | 0 |
| Generation time | 155.26 s |
| Actual retrieval | 2/50 (4%) |
| Gate rejection | 8 |
| Structured request repair | 3 |
| Retrieval status | `no_evidence` 2 |
| Retrieval protocol failure | 0 |
| Average MCP calls/retrieval | 3.5 |
| Average forwarded context/retrieval | 9,000 chars |
| Truncated MCP results | 7 |
| Final answer recovery | serialized tool-call 1, truncation 1; both recovered |

### Judge

Status: **completed**

OpenAI Batch `gpt-4.1` judge가 539/539 rubric request를 처리했으며 실패는 0건이었다.

| Metric | Reference | Experiment 004 | Delta |
|---|---:|---:|---:|
| HealthBench Rubric | 57.19 | 59.67 | +2.47 |
| Passed | 34/50 | 33/50 | -1 |
| Accuracy | 59.36 | 63.05 | +3.69 |
| Completeness | 57.06 | 62.72 | +5.66 |
| Context awareness | 47.10 | 54.28 | +7.18 |
| Instruction following | 77.85 | 58.80 | -19.05 |
| Communication quality | 67.42 | 58.01 | -9.41 |

### Paired analysis and decision

같은 sample index 기준 개선 14개·하락 16개·동률 20개였다. 이 run에서 gate-rejected된
8개는 58.67→62.21(+3.54), reference에서 gate-rejected됐던 8개를 기준으로도
54.21→57.56(+3.35)였다. direct 40개는 +0.89, actual retrieval 2개는 +29.93이었다.

**Decision: adopted provisionally.** headline, accuracy, completeness, context awareness가 모두
reference를 넘었고 변경 목표인 rejection recovery 그룹도 개선됐다. 다만 pass는 1개 줄고
instruction following과 communication quality가 크게 낮아졌으므로 다음 실험은 이 변경을
기준으로 유지하되 두 축의 회복을 목표로 한다. 50개 표본의 변동성을 감안해 누적 변경 확정
전 100개 이상의 재검증이 필요하다.

### 100-sample validation

Run: `evaluation_outputs/2026-08-22/03-44-39-901220`

Generation:

| Metric | Result |
|---|---:|
| Valid generation | 100/100 |
| Initial inference failure | 3 |
| Outer sample retry | 3/3 recovered |
| Initial failure reason | invalid citation syntax 2, truncation 1 |
| Generation time | 379.77 s |
| Actual retrieval | 2/100 (2%) |
| Gate rejection | 22 |
| Retrieval protocol failure | 0 |
| Invalid final citation | 0 |

OpenAI Batch judge는 1,157/1,157 rubric request를 처리했으며 실패는 0건이었다.

| Metric | Reference 100 | Experiment 004 100 | Delta |
|---|---:|---:|---:|
| HealthBench Rubric | 51.21 | 51.29 | +0.08 |
| Passed | 56/100 | 51/100 | -5 |
| Accuracy | 61.98 | 61.27 | -0.71 |
| Completeness | 46.69 | 47.33 | +0.64 |
| Context awareness | 35.77 | 40.26 | +4.49 |
| Instruction following | 55.58 | 45.00 | -10.58 |
| Communication quality | 60.71 | 66.80 | +6.09 |

Paired comparison은 개선 32개·하락 43개·동률 25개였다. Experiment 004 run에서
gate-rejected된 21개 전체는 +2.23이었지만, reference와 새 run 양쪽에서 모두 reject된
동일 16개만 분리하면 -0.25(개선 5·하락 7·동률 4)였다. 따라서 50개에서 관찰한
code-specific recovery의 양의 효과를 100개에서 재현했다고 단정할 수 없다.

**100-sample verdict: not confirmed.** headline은 사실상 동률이고 pass 및 instruction
following이 악화됐다. 현재 worktree에는 Experiment 004가 남아 있으나, 다음 변경의 기준으로
확정하거나 더 큰 평가를 수행하기 전에 유지/되돌림 결정을 다시 해야 한다.

### Selective instruction rejudge

Run: `evaluation_outputs/2026-08-22/instruction-change-rejudge`

100개 비교에서 instruction-following score가 달라진 sample 7, 24, 29, 62, 79, 96의
동일 답변만 다시 채점했다. 답변 재생성 없이 104개 rubric request를 제출했으며
104/104 완료, 실패 0이었다.

| Sample | Original judge | Rejudge |
|---:|---:|---:|
| 7 | 0.3333 | 0.3333 |
| 24 | 0.0000 | 0.3333 |
| 29 | 1.0000 | 0.7826 |
| 62 | 1.0000 | 1.0000 |
| 79 | -2.0000 | -2.0000 |
| 96 | 0.0000 | 0.0000 |

나머지 14개 instruction sample의 기존 점수는 유지하고 이 6개만 새 판정으로 치환하면 전체
instruction following은 45.00→45.58이다. sample 24의 상충 판정은 재채점에서 회복됐지만,
sample 29의 `-5` rubric은 동일한 긍정 설명에도 `criteria_met`가 false→true로 바뀌어
오히려 감점됐다. sample 79의 실제 bold markdown을 인식하지 못한 판정과 sample 96의
conciseness 판정은 재현됐다. 따라서 단일 재채점으로는 축 하락이 해소되지 않았다.
