"""
판례 데이터 수집 — 법제처 국가법령정보 OpenAPI (open.law.go.kr).

설계:
- 도메인별 검색 키워드로 lawSearch.do 페이지네이션 → lawService.do 상세 조회
- 라벨은 core.domains.classify_precedent (참조법령 가중 스코어링) 결과 기준
- 사건번호/판례일련번호 이중 dedupe, 진행상태 저장으로 중단 후 재개 가능
- rate limit 0.2s 준수
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Generator
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from panrye.config import get_settings
from panrye.core.domains import classify_precedent

logger = logging.getLogger(__name__)

# 도메인별 law.go.kr 검색 키워드. 약세 도메인(부동산/노동/가족법)은 세분화된 쟁점 키워드로 보강.
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "형사": ["절도", "사기", "폭행", "살인", "강도", "횡령", "배임", "마약",
             "보이스피싱", "음주운전"],
    "민사": ["손해배상", "대여금", "계약해제", "채권양도", "불법행위", "부당이득",
             "구상금", "보증채무"],
    "가족법": ["이혼", "상속", "친권", "양육비", "위자료", "재산분할",
               "유류분반환", "상속회복", "면접교섭", "사실혼"],
    "행정": ["행정처분", "과세처분", "취소소송", "영업정지", "면허취소",
             "국가배상", "정보공개", "인허가"],
    "노동": ["부당해고", "임금체불", "퇴직금", "산업재해", "통상임금",
             "부당노동행위", "직위해제", "취업규칙", "해고예고"],
    "부동산": ["전세보증금", "임대차보증금", "명도소송", "임차권등기", "소유권이전등기",
               "유치권", "재개발", "분양권", "가등기", "근저당", "공인중개사", "집합건물"],
}

# 도메인별 수집 목표 (최종 라벨 기준)
DEFAULT_TARGETS: dict[str, int] = {
    "형사": 800, "민사": 800, "부동산": 800,
    "노동": 700, "가족법": 600, "행정": 600,
}

PAGE_SIZE = 20
RATE_LIMIT_SEC = 0.2
SAVE_EVERY = 50

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_html(s: str) -> str:
    """law.go.kr 응답의 <br/> 등 HTML 태그 제거."""
    if not s:
        return ""
    s = s.replace("<br/>", "\n").replace("<br>", "\n")
    s = _TAG_RE.sub("", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def _format_date(raw: str) -> str:
    """'20260129' → '2026.01.29'"""
    raw = (raw or "").strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}.{raw[4:6]}.{raw[6:]}"
    return raw


def _extract_verdict(content: str) -> str:
    """판례내용에서 【주 문】 섹션 추출."""
    m = re.search(r"【주\s*문】\s*(.+?)(?=【|$)", content, re.S)
    return m.group(1).strip()[:300] if m else ""


def _parse_statutes(raw: str) -> list[str]:
    """참조조문 문자열 → 개별 조문 목록. '[1]' 마커 제거, '/'와 ',' 분리."""
    if not raw:
        return []
    raw = re.sub(r"\[\d+\]", " ", raw)
    seen: set[str] = set()
    out = []
    for part in re.split(r"[/,]", raw):
        part = part.strip()
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return out


@dataclass
class Precedent:
    case_id: str
    case_name: str
    case_number: str  # 사건번호 (예: 2023두54914)
    court: str
    date: str
    domain: str
    summary: str
    full_text: str
    statutes: list[str]
    verdict: str
    source: str


class LawAPIClient:
    """법제처 국가법령정보 OpenAPI 클라이언트."""

    BASE_URL = "https://www.law.go.kr/DRF"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def search_precedents(
        self, query: str, page: int = 1, page_size: int = PAGE_SIZE
    ) -> list[dict]:
        params = {
            "OC": self.api_key,
            "target": "prec",
            "type": "JSON",
            "query": query,
            "page": page,
            "display": page_size,
        }
        resp = self.session.get(f"{self.BASE_URL}/lawSearch.do", params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("PrecSearch", {}).get("prec", [])
        if isinstance(items, dict):
            items = [items]
        return items

    def get_precedent_detail(self, case_id: str) -> dict:
        params = {"OC": self.api_key, "target": "prec", "ID": case_id, "type": "JSON"}
        resp = self.session.get(f"{self.BASE_URL}/lawService.do", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("PrecService", {})

    def _parse_precedent(self, raw: dict, searched_domain: str) -> Precedent:
        # 상세 조회 응답 필드: 판례정보일련번호/판례내용/판시사항/판결요지/참조조문 (HTML 포함)
        content = _clean_html(raw.get("판례내용", ""))
        summary = _clean_html(raw.get("판시사항", "")) or _clean_html(raw.get("판결요지", ""))
        statutes = _parse_statutes(_clean_html(raw.get("참조조문", "")))
        case_name = (raw.get("사건명") or "").strip()
        domain = classify_precedent(
            statutes,
            text=content,
            case_name=case_name,
            case_type=raw.get("사건종류명"),
            default=searched_domain,
        )
        return Precedent(
            case_id=str(raw.get("판례정보일련번호", "")),
            case_name=case_name,
            case_number=(raw.get("사건번호") or "").strip(),
            court=raw.get("법원명", ""),
            date=_format_date(raw.get("선고일자", "")),
            domain=domain,
            summary=summary,
            full_text=content,
            statutes=statutes,
            verdict=_extract_verdict(content),
            source="law.go.kr",
        )

    def iter_keyword(
        self,
        domain: str,
        keyword: str,
        start_page: int = 1,
        skip_case_ids: set[str] | None = None,
    ) -> Generator[tuple[Precedent | None, int], None, None]:
        """키워드 하나를 페이지네이션. (판례 or None, 현재 페이지) yield.
        skip_case_ids에 있는 판례는 상세 조회 없이 건너뜀 (재개/교차 키워드 중복 절약)."""
        page = start_page
        while True:
            try:
                items = self.search_precedents(keyword, page=page)
            except Exception as e:
                logger.error(f"검색 실패 [{keyword} p{page}]: {e}")
                return
            if not items:
                return

            for item in items:
                case_id = str(item.get("판례일련번호", ""))
                if not case_id:
                    continue
                if skip_case_ids and case_id in skip_case_ids:
                    continue
                try:
                    detail = self.get_precedent_detail(case_id)
                    parsed = self._parse_precedent(detail, domain)
                    time.sleep(RATE_LIMIT_SEC)
                    if parsed.case_id and parsed.full_text:
                        yield parsed, page
                except Exception as e:
                    logger.warning(f"판례 {case_id} 파싱 실패: {e}")

            if len(items) < PAGE_SIZE:
                return
            page += 1


class IngestStore:
    """수집 결과 + 진행상태 영속화. 중단 후 재개 지원."""

    def __init__(self, raw_dir: Path | None = None):
        self.raw_dir = raw_dir or get_settings().raw_data_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.raw_dir / "law_precedents.json"
        self.state_path = self.raw_dir / "ingest_state.json"
        self.precedents: list[dict] = []
        self.seen_case_ids: set[str] = set()
        self.seen_case_numbers: set[str] = set()
        self.state: dict = {"cursors": {}, "done_keywords": []}
        self._load()

    def _load(self) -> None:
        if self.data_path.exists():
            with open(self.data_path, encoding="utf-8") as f:
                self.precedents = json.load(f)
            self.seen_case_ids = {p["case_id"] for p in self.precedents}
            self.seen_case_numbers = {p["case_number"] for p in self.precedents if p["case_number"]}
            logger.info(f"기존 수집 데이터 로드: {len(self.precedents)}건")
        if self.state_path.exists():
            with open(self.state_path, encoding="utf-8") as f:
                self.state = json.load(f)

    def is_duplicate(self, prec: Precedent) -> bool:
        if prec.case_id in self.seen_case_ids:
            return True
        if prec.case_number and prec.case_number in self.seen_case_numbers:
            return True
        return False

    def add(self, prec: Precedent) -> None:
        self.precedents.append(asdict(prec))
        self.seen_case_ids.add(prec.case_id)
        if prec.case_number:
            self.seen_case_numbers.add(prec.case_number)

    def domain_counts(self) -> Counter:
        return Counter(p["domain"] for p in self.precedents)

    def save(self) -> None:
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self.precedents, f, ensure_ascii=False, indent=2)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def export_all(self) -> Path:
        """청커 입력 파일(all_precedents.json)로 내보내기."""
        out = self.raw_dir / "all_precedents.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.precedents, f, ensure_ascii=False, indent=2)
        logger.info(f"내보내기 완료: {out} ({len(self.precedents)}건)")
        return out


def collect_domain(domain: str, target: int, store: IngestStore | None = None) -> IngestStore:
    """
    도메인 하나를 목표 건수까지 수집 (최종 라벨 기준 카운트).
    키워드 순회, 진행상태/데이터 주기 저장.
    """
    settings = get_settings()
    if not settings.law_api_key:
        raise ValueError("LAW_API_KEY 미설정. .env 확인하세요.")
    if domain not in DOMAIN_KEYWORDS:
        raise ValueError(f"알 수 없는 도메인: {domain}")

    store = store or IngestStore()
    client = LawAPIClient(settings.law_api_key)
    added_since_save = 0

    for keyword in DOMAIN_KEYWORDS[domain]:
        cursor_key = f"{domain}:{keyword}"
        if cursor_key in store.state["done_keywords"]:
            continue
        if store.domain_counts()[domain] >= target:
            break

        start_page = store.state["cursors"].get(cursor_key, 1)
        logger.info(f"수집: [{domain}] '{keyword}' (page {start_page}부터, "
                    f"현재 {store.domain_counts()[domain]}/{target}건)")

        exhausted = True
        for prec, page in client.iter_keyword(
            domain, keyword, start_page, skip_case_ids=store.seen_case_ids
        ):
            store.state["cursors"][cursor_key] = page
            if store.is_duplicate(prec):
                continue
            store.add(prec)
            added_since_save += 1
            if added_since_save >= SAVE_EVERY:
                store.save()
                added_since_save = 0
            if store.domain_counts()[domain] >= target:
                exhausted = False
                break

        if exhausted:
            store.state["done_keywords"].append(cursor_key)
        store.save()

    store.save()
    logger.info(f"[{domain}] 수집 종료: {store.domain_counts()[domain]}건 (목표 {target})")
    return store


def collect_all(targets: dict[str, int] | None = None) -> IngestStore:
    """모든 도메인 순차 수집 후 all_precedents.json 내보내기."""
    targets = targets or DEFAULT_TARGETS
    store = IngestStore()
    for domain, target in targets.items():
        collect_domain(domain, target, store)
    store.export_all()
    dist = store.domain_counts()
    logger.info(f"전체 수집 완료: {sum(dist.values())}건 — {dict(dist.most_common())}")
    return store
