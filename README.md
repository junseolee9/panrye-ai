---
title: PanRye AI
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: false
license: mit
---

# PanRye.ai — 한국 판례 기반 법률 상담 RAG 챗봇

**판례 기반 법률 상담 AI.** 상황을 설명하면 가장 유사한 실제 판례를 검색하여 답변합니다.

## 아키텍처

```
사용자 입력
    ↓
[Agent 1] Zero-Shot 도메인 분류 (형사/민사/가족/행정/노동/부동산)
    ↓
[Agent 2] HyDE 쿼리 재작성 (구어체 → 법률 용어)
    ↓
[Agent 3] Hybrid 검색: Dense(ko-sroberta) + BM25 → RRF → Cross-encoder Rerank
    ↓
[Agent 4] 판례 요약 (kobart-summarization)
    ↓
[Agent 5] 최종 답변 생성 (Groq llama-3.3-70b / Gemini 1.5 Flash)
    ↓
[RAGAS Eval] Faithfulness / Relevancy 자동 평가
```

Orchestration: **LangGraph** state machine

## HuggingFace Tasks 활용

| Task | 모델 | 용도 |
|------|------|------|
| Feature Extraction | `jhgan/ko-sroberta-multitask` | 판례 임베딩 |
| Sentence Similarity | 동일 | 유사 판례 검색 |
| Zero-Shot Classification | `joeddav/xlm-roberta-large-xnli` | 법적 영역 분류 |
| Summarization | `digit82/kobart-summarization` | 판례 요약 |
| Text Generation | Groq / Gemini (free) | 최종 답변 |

## 기술 스택 (전부 무료)

- **Vector DB**: ChromaDB (로컬)
- **BM25**: rank_bm25 + kiwipiepy 형태소 토큰화
- **Reranker**: Dongjin-kr/ko-reranker (cross-encoder)
- **Orchestration**: LangGraph
- **Eval**: RAGAS
- **API**: FastAPI + SSE
- **UI**: Gradio → HF Spaces

## 데이터 소스

- **법제처 국가법령정보 OpenAPI** (`open.law.go.kr`)
- **HuggingFace Datasets**: `lawcompany/KLAID`

## 실행 방법

```bash
# 1. 환경 설정
cp .env.example .env
# .env에 API 키 입력 (GROQ_API_KEY 필수, LAW_API_KEY 선택)

# 2. 의존성 설치 (Python 3.10+ 필요)
# conda 사용 시:
conda activate rag  # 또는 python3.10+ 환경
pip install -r requirements.txt

# 3. 데이터 파이프라인 (판례 수집 → 청킹 → 인덱싱)
python run_data_pipeline.py

# 4a. Gradio UI 실행
python ui/app.py

# 4b. FastAPI 서버 실행
uvicorn api.main:app --reload --app-dir .

# 5. RAGAS 평가
python eval/ragas_runner.py
```

## 성능 목표

| 메트릭 | 목표 |
|--------|------|
| RAGAS Faithfulness | ≥ 0.80 |
| Answer Relevancy | ≥ 0.75 |
| Context Precision | ≥ 0.70 |
| Retrieval Latency | < 2초 |

## 프로젝트 구조

```
panrye-ai/
├── data/           # 수집·청킹·인덱싱
├── agents/         # 5개 전문 에이전트
├── graph/          # LangGraph 파이프라인
├── eval/           # RAGAS 평가 하네스
├── api/            # FastAPI 백엔드
├── ui/             # Gradio 프론트엔드
└── logs/           # SQLite 로깅
```

---

⚠️ **면책 고지**: 본 서비스는 AI 참고 정보이며 정식 법률 자문을 대체하지 않습니다.
