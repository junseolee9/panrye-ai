"""
판례 데이터 수집 모듈.
소스 1: 법제처 국가법령정보 OpenAPI (대법원/헌재/하급심)
소스 2: HuggingFace lawcompany/KLAID 데이터셋
"""
import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Generator
from dataclasses import dataclass, asdict

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from data.domain_rules import classify_precedent

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LAW_API_KEY = os.getenv("LAW_API_KEY", "")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
RAW_DATA_DIR.mkdir(exist_ok=True)

# law.go.kr 검색 키워드 (도메인 라벨은 agents/classifier.py와 동일 체계)
LEGAL_DOMAINS = {
    "형사": ["절도", "사기", "폭행", "살인", "강도", "횡령", "배임", "마약"],
    "민사": ["손해배상", "계약", "임대차", "채권", "물권", "불법행위"],
    "가족법": ["이혼", "상속", "친권", "양육", "위자료"],
    "행정": ["행정처분", "과세", "허가", "취소", "행정소송"],
    "노동": ["임금", "해고", "부당해고", "산재", "근로계약"],
    "부동산": ["매매", "등기", "전세", "임대", "건축"],
}

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


@dataclass
class Precedent:
    case_id: str
    case_name: str
    case_number: str  # 사건번호 (예: 2023두54914). KLAID는 없음.
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

    def search_precedents(self, query: str, page: int = 1, page_size: int = 20) -> dict:
        params = {
            "OC": self.api_key,
            "target": "prec",
            "type": "JSON",
            "query": query,
            "page": page,
            "display": page_size,
        }
        resp = self.session.get(f"{self.BASE_URL}/lawSearch.do", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_precedent_detail(self, case_id: str) -> dict:
        params = {
            "OC": self.api_key,
            "target": "prec",
            "ID": case_id,
            "type": "JSON",
        }
        resp = self.session.get(f"{self.BASE_URL}/lawService.do", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def iter_domain_precedents(
        self, domain: str, keywords: list[str], max_per_keyword: int = 50
    ) -> Generator[Precedent, None, None]:
        for keyword in keywords:
            logger.info(f"수집 중: [{domain}] {keyword}")
            page = 1
            collected = 0

            while collected < max_per_keyword:
                try:
                    result = self.search_precedents(keyword, page=page, page_size=20)
                    items = result.get("PrecSearch", {}).get("prec", [])
                    if not items:
                        break

                    if isinstance(items, dict):
                        items = [items]

                    for item in items:
                        case_id = item.get("판례일련번호", "")
                        if not case_id:
                            continue

                        try:
                            detail = self.get_precedent_detail(case_id)
                            prec = detail.get("PrecService", {})
                            parsed = self._parse_precedent(prec, domain)
                            if parsed.case_id and parsed.full_text:
                                yield parsed
                                collected += 1
                            time.sleep(0.2)  # rate limit 준수
                        except Exception as e:
                            logger.warning(f"판례 {case_id} 파싱 실패: {e}")

                    page += 1
                    if len(items) < 20:
                        break

                except Exception as e:
                    logger.error(f"검색 실패 [{keyword} p{page}]: {e}")
                    break

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


class KLAIDLoader:
    """HuggingFace lawcompany/KLAID 데이터셋 로더."""

    def load(self, max_samples: int = 20000) -> list[Precedent]:
        logger.info("KLAID 데이터셋 로딩 중...")
        ds = load_dataset("lawcompany/KLAID", "ljp", split="train")
        precedents = []

        for i, item in enumerate(tqdm(ds, total=min(max_samples, len(ds)))):
            if i >= max_samples:
                break
            fact_text = item.get("fact", "")
            laws = item.get("laws_service", "")
            statutes = [s.strip() for s in laws.split(",") if s.strip()]
            domain = classify_precedent(statutes, text=fact_text)

            # 사건명: 법조문 나열 대신 대표 법령명으로 (예: "도로교통법 위반 사건")
            law_name = statutes[0].split(" 제")[0].strip() if statutes else ""
            case_name = f"{law_name} 위반 사건" if law_name else "KLAID 판례"

            precedents.append(
                Precedent(
                    case_id=f"klaid_{i:05d}",
                    case_name=case_name,
                    case_number="",
                    court="",
                    date="",
                    domain=domain,
                    summary=fact_text[:500],
                    full_text=fact_text,
                    statutes=statutes,
                    verdict="",
                    source="KLAID",
                )
            )

        logger.info(f"KLAID: {len(precedents)}건 로드 완료")
        return precedents


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


def save_raw(precedents: list[Precedent], filename: str) -> Path:
    out = RAW_DATA_DIR / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in precedents], f, ensure_ascii=False, indent=2)
    logger.info(f"저장 완료: {out} ({len(precedents)}건)")
    return out


def ingest_all(
    use_law_api: bool = True,
    use_klaid: bool = True,
    max_per_keyword: int = 30,
) -> list[Precedent]:
    all_precedents: list[Precedent] = []

    if use_law_api and LAW_API_KEY:
        client = LawAPIClient(LAW_API_KEY)
        for domain, keywords in LEGAL_DOMAINS.items():
            for prec in client.iter_domain_precedents(domain, keywords, max_per_keyword):
                all_precedents.append(prec)
        save_raw(all_precedents, "law_api_precedents.json")
    else:
        if use_law_api:
            logger.warning("LAW_API_KEY 미설정. law.go.kr 수집 건너뜀.")

    if use_klaid:
        loader = KLAIDLoader()
        klaid_data = loader.load(max_samples=3000)
        all_precedents.extend(klaid_data)
        save_raw(klaid_data, "klaid_precedents.json")

    # 중복 제거 (case_id 기준)
    seen = set()
    unique = []
    for p in all_precedents:
        if p.case_id not in seen:
            seen.add(p.case_id)
            unique.append(p)

    save_raw(unique, "all_precedents.json")
    logger.info(f"수집 완료: 총 {len(unique)}건")
    return unique


if __name__ == "__main__":
    precedents = ingest_all(use_law_api=bool(LAW_API_KEY), use_klaid=True)
    print(f"\n수집 완료: {len(precedents)}건")
    print(f"도메인 분포:")
    from collections import Counter
    dist = Counter(p.domain for p in precedents)
    for domain, cnt in dist.most_common():
        print(f"  {domain}: {cnt}건")
