import os
import re
import time
from importlib import import_module
from io import BytesIO
from collections import defaultdict, deque
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from legal_rag import add_uploaded_document_chunks, search_knowledge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index.html")

app = FastAPI(title="SaulGPT API", version="3.4.0")

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,null",
    ).split(",")
    if origin.strip()
]
ALLOW_ALL_ORIGINS = os.getenv("ALLOW_ALL_ORIGINS", "1") == "1"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ALL_ORIGINS else ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
MODEL_CANDIDATES = [
    name.strip()
    for name in os.getenv(
        "OLLAMA_MODEL_CANDIDATES",
        "mistral,llama3.1:8b-instruct,qwen2.5:7b-instruct",
    ).split(",")
    if name.strip()
]
if OLLAMA_MODEL not in MODEL_CANDIDATES:
    MODEL_CANDIDATES.insert(0, OLLAMA_MODEL)
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "520"))
GENERATE_MAX_TOKENS = int(os.getenv("GENERATE_MAX_TOKENS", "900"))
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "30"))
API_KEY = os.getenv("SAULGPT_API_KEY")
MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
CHAT_CACHE_TTL_SECONDS = int(os.getenv("CHAT_CACHE_TTL_SECONDS", "900"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "10485760"))  # 10 MB

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}

_rate_limit_store: Dict[str, deque] = defaultdict(deque)
_chat_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

LAW_FOCUS_GUIDANCE = (
    "I focus on Indian legal information. If you describe the issue in legal terms, "
    "I can explain the likely legal category and general context."
)

NON_ADVICE_NOTE = (
    "This is general legal information, not legal advice or a lawyer-client opinion."
)

LAW_KEYWORDS = {
    "law", "legal", "case", "ipc", "crpc", "evidence", "fir", "complaint", "petition",
    "bail", "section", "court", "judge", "advocate", "lawyer", "police", "crime", "fraud",
    "forgery", "theft", "defamation", "conspiracy", "intimidation", "contract", "notice",
    "litigation", "arrest", "rights", "remedy", "punishment", "offence", "offense",
    "charge", "cheating", "breach", "trust", "property", "civil", "criminal", "procedure",
    "appeal", "warrant", "summons", "injunction", "damages", "liability", "mediation",
    "divorce", "custody", "maintenance", "anticipatory", "quash", "affidavit", "agreement",
    "draft", "strategy", "accused", "respondent", "complainant", "chargesheet", "charge-sheet",
    "citizen", "civilian", "tenant", "landlord", "property", "consumer", "employment",
    "employer", "employee", "salary", "wage", "wages", "workplace",
    "service", "notice", "agreement", "family", "inheritance", "succession",
    "rent", "rental", "lease", "owner", "ownership", "land", "title", "deed",
    "mutation", "registry", "registration", "encumbrance", "partition", "probate", "will",
    "eviction", "tenancy", "stamp", "conveyance", "sublease", "allotment",
}

LEGAL_CATEGORIES: Dict[str, set[str]] = {
    "Property or tenancy issue": {
        "property", "tenant", "tenancy", "landlord", "rent", "lease", "owner",
        "ownership", "eviction", "land", "deed", "mutation", "registry", "title",
        "sublease", "allotment", "flat", "house", "apartment", "plot", "real estate",
        "rera", "builder", "possession", "encumbrance", "stamp duty", "registration",
        "conveyance", "partition", "probate", "will", "inheritance", "succession",
        "trespass", "dispossession", "mortgage",
    },
    "Contract or payment dispute": {
        "contract", "agreement", "payment", "deposit", "refund", "invoice",
        "breach", "dues", "advance", "money", "service", "cheque", "bounce",
        "dishonour", "negotiable", "promissory", "arbitration", "settlement",
        "vendor", "supplier", "buyer", "seller", "transaction", "deal",
    },
    "Fraud or cheating concern": {
        "fraud", "cheat", "cheating", "forgery", "scam", "deception", "fake",
        "misrepresentation", "impersonation", "ponzi", "embezzlement",
        "420", "criminal breach", "trust", "dishonest",
    },
    "Consumer issue": {
        "consumer", "defect", "deficiency", "warranty", "product", "seller",
        "ecommerce", "replacement", "return", "refund", "amazon", "flipkart",
        "service provider", "complaint forum", "district commission",
    },
    "Employment dispute": {
        "employment", "salary", "termination", "dismissal", "workplace", "hr",
        "harassment", "wages", "bonus", "notice period", "pf", "gratuity",
        "labour", "industrial", "retrenchment", "unfair dismissal", "employer",
    },
    "Family or relationship dispute": {
        "divorce", "custody", "maintenance", "marriage", "domestic", "inheritance",
        "succession", "family", "will", "alimony", "dowry", "matrimonial",
        "guardian", "adoption", "child", "spouse", "husband", "wife",
    },
    "Cyber or online harm": {
        "cyber", "online", "phishing", "hacked", "social media", "digital",
        "data", "account", "otp", "upi", "payment fraud", "bank fraud",
        "email", "whatsapp", "instagram", "deepfake", "blackmail", "extortion online",
        "it act", "cybercrime", "identity theft",
    },
    "Defamation or reputation issue": {
        "defamation", "reputation", "false statement", "public post", "libel", "slander",
        "character", "honour", "dignity",
    },
    "Traffic or road incident": {
        "traffic", "road", "vehicle", "accident", "driving", "license", "challan",
        "motor", "car", "truck", "mva", "insurance claim", "hit and run",
    },
    "FIR and police complaint": {
        "fir", "first information report", "police complaint", "police station",
        "register complaint", "lodge complaint", "section 154", "cognizable",
        "non-cognizable", "chargesheet", "charge sheet", "challan",
        "investigation", "arrest", "zero fir", "complaint to police",
        "magistrate", "crpc", "bnss",
    },
    "Bail and custody": {
        "bail", "anticipatory bail", "regular bail", "remand", "custody",
        "detention", "arrested", "locked up", "jail", "prison",
        "section 436", "section 437", "section 438", "section 439",
        "bailable", "non-bailable", "surety",
    },
    "General criminal concern": {
        "crime", "offence", "offense", "threat", "violence", "assault", "police",
        "arrest", "intimidation", "murder", "theft", "robbery", "kidnapping",
        "rape", "harassment", "criminal", "ipc", "bns", "warrant", "quash",
        "acquittal", "conviction", "sentence", "punishment", "accused",
    },
}


class CaseRequest(BaseModel):
    case_type: str = Field(min_length=3, max_length=120)
    incident: str = Field(min_length=10, max_length=3000)
    amount: Optional[str] = Field(default="", max_length=80)

    @field_validator("case_type", "incident", "amount", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        return _normalize_text(value)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=3000)

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: Any) -> str:
        return _normalize_text(value)


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=3000)
    history: List[ChatTurn] = Field(default_factory=list, max_length=20)

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: Any) -> str:
        return _normalize_text(value)


class Citation(BaseModel):
    kind: str
    id: str
    section: str
    text: str
    source: str
    effective_date: str
    score: float


class GenerateResponse(BaseModel):
    draft: str
    citations: List[Citation]
    disclaimer: str


class ChatResponse(BaseModel):
    reply: str
    citations: List[Citation]
    redirected_to_law: bool
    disclaimer: str


class IngestFileResult(BaseModel):
    filename: str
    document_type: Optional[str] = None
    status: Literal["ingested", "failed"]
    chunks: int = 0
    detail: str


class IngestResponse(BaseModel):
    total_files: int
    ingested_files: int
    failed_files: int
    total_chunks: int
    results: List[IngestFileResult]


class ReportExportRequest(BaseModel):
    title: str = "SaulGPT Legal Report"
    content: str = Field(min_length=1, max_length=20000)
    format: Literal["pdf", "docx"] = "pdf"
    filename: Optional[str] = None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("must be a string")
    return " ".join(value.split()).strip()


def _check_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def _check_rate_limit(client_id: str) -> None:
    now = time.time()
    window_start = now - 60
    bucket = _rate_limit_store[client_id]

    while bucket and bucket[0] < window_start:
        bucket.popleft()

    if len(bucket) >= REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    bucket.append(now)


def _is_law_topic(text: str) -> bool:
    lower = text.lower()
    words = set(re.findall(r"[a-zA-Z]+", lower))
    if words.intersection(LAW_KEYWORDS):
        return True

    # Handle real-life citizen case narratives without explicit legal keywords.
    if re.search(r"\b(i|my|me|we|our)\b", lower) and re.search(
        r"\b(partner|landlord|tenant|deposit|money|payment|loan|took|stole|fraud|cheat|threat|forgery|notice|police|case)\b",
        lower,
    ):
        return True

    legal_patterns = [
        r"section\s+\d+",
        r"ipc\s*\d*",
        r"fir",
        r"criminal",
        r"civil",
        r"rental?\s+law",
        r"rent\s+act",
        r"owner(ship)?",
        r"land",
        r"property",
        r"title\s+deed",
        r"mutation",
        r"tenant|tenancy|landlord|lease|eviction",
        r"legal\s+notice",
        r"what\s+is\s+the\s+law",
        r"can\s+i\s+file",
        r"how\s+to\s+file",
        r"draft\s+(complaint|petition|notice|report)",
        r"accused",
    ]
    if any(re.search(pattern, lower) for pattern in legal_patterns):
        return True

    # Bias toward legal for citizen-law style questions.
    if re.search(r"\b(act|law|rule|code|regulation|right|remedy|owner|tenant|rent|land)\b", lower):
        return True
    return False


def _sanitize_reference_text(text: str, collapse_whitespace: bool = True) -> str:
    cleaned = re.sub(r"\b(?:section|sec\.?|article)\s*\d+[a-zA-Z]?\b", "a relevant provision", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:ipc|crpc|cpc|bns|bnss|bsa)\s*\d*[a-zA-Z]?\b", "the legal framework", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bIndian Penal Code\b", "criminal law framework", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bCode of Criminal Procedure\b", "criminal procedure framework", cleaned, flags=re.IGNORECASE)
    if collapse_whitespace:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    else:
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _infer_legal_category(message: str, retrieval: List[Dict[str, Any]]) -> str:
    combined = message.lower()
    for doc in retrieval[:4]:
        combined += f" {str(doc.get('text', '')).lower()} {str(doc.get('section', '')).lower()}"

    best_category = "General legal issue"
    best_score = 0
    for category, keywords in LEGAL_CATEGORIES.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > best_score:
            best_score = score
            best_category = category
    return best_category


def _category_context(category: str) -> str:
    contexts = {
        "Property or tenancy issue": "This usually concerns rights and obligations around possession, use of property, rent terms, and documentary proof of ownership or occupancy.",
        "Contract or payment dispute": "This usually concerns promises between parties, whether terms were fulfilled, and whether money or services were delivered as agreed.",
        "Fraud or cheating concern": "This usually concerns alleged deception, false representation, and financial or practical loss caused by that conduct.",
        "Consumer issue": "This usually concerns product or service quality, fair trade expectations, and grievance resolution channels for buyers.",
        "Employment dispute": "This usually concerns working conditions, wages/benefits, termination process, and employer-employee obligations.",
        "Family or relationship dispute": "This usually concerns personal status, support obligations, family rights, and documentary family records.",
        "Cyber or online harm": "This usually concerns digital evidence, unauthorized access, online impersonation, or misuse of personal information.",
        "Defamation or reputation issue": "This usually concerns allegedly harmful statements and whether they caused reputational impact.",
        "Traffic or road incident": "This usually concerns compliance with road-safety rules, incident facts, and responsibility allocation.",
        "FIR and police complaint": (
            "An FIR (First Information Report) is a written document prepared by the police when they receive information about a cognizable offence. "
            "Under the Bharatiya Nagarik Suraksha Sanhita 2023 (earlier CrPC), any person can report a cognizable offence at the nearest police station. "
            "If police refuse to register an FIR, you can approach the Superintendent of Police or file a complaint directly before a Magistrate. "
            "A 'Zero FIR' can be filed at any police station regardless of jurisdiction, which is then transferred to the appropriate station."
        ),
        "Bail and custody": (
            "Bail is the conditional release of an arrested person pending trial. "
            "For bailable offences, bail is a right. For non-bailable offences, bail is discretionary and granted by a Magistrate or Sessions Court. "
            "Anticipatory bail (pre-arrest bail) can be sought from the Sessions Court or High Court when there is a reasonable apprehension of arrest. "
            "The court considers factors like nature of the offence, prior criminal record, flight risk, and likelihood of tampering with evidence."
        ),
        "General criminal concern": "This usually concerns allegations of wrongful conduct, evidence quality, and procedural safeguards under Indian criminal law.",
    }
    return contexts.get(category, "This appears to involve a legal disagreement where facts, documents, and timeline are central to understanding the issue.")



def _document_observations(retrieval: List[Dict[str, Any]]) -> str:
    uploaded_items = [d for d in retrieval if d.get("kind") == "uploaded"][:2]
    if not uploaded_items:
        return ""

    observations: List[str] = []
    for item in uploaded_items:
        text = _sanitize_reference_text(str(item.get("text", "")))
        if text:
            observations.append(f"- Uploaded document: {text[:180]}")
    if not observations:
        return ""
    return "Document observations:\n" + "\n".join(observations) + "\n\n"


def _general_next_steps(category: str) -> str:
    common = [
        "- Build a dated timeline of events in plain language.",
        "- Gather relevant documents (communications, receipts, agreements, identity records, and proof of payment if applicable).",
        "- Verify process details on official government portals or authority websites.",
        "- Discuss the facts with a qualified lawyer or legal-aid service before taking decisions.",
    ]

    if category == "Cyber or online harm":
        common.insert(2, "- Preserve screenshots, transaction logs, account alerts, and device/email headers.")
    elif category == "Property or tenancy issue":
        common.insert(2, "- Organize property papers, rent records, correspondence, and proof of possession/occupancy.")
    elif category == "Employment dispute":
        common.insert(2, "- Compile offer letters, salary records, HR emails, attendance logs, and any policy documents.")

    return "\n".join(common)


def _build_information_reply(message: str, retrieval: List[Dict[str, Any]]) -> str:
    category = _infer_legal_category(message, retrieval)
    uploaded_items = [d for d in retrieval if d.get("kind") == "uploaded"][:2]
    law_items = [
        d
        for d in retrieval
        if d.get("kind") == "law" and float(d.get("score", 0.0)) >= 0.65
    ][:2]
    context_points = []
    for item in uploaded_items + law_items:
        text = _sanitize_reference_text(str(item.get("text", "")))
        if text:
            if item.get("kind") == "uploaded":
                context_points.append(f"- Uploaded document context: {text[:180]}")
            else:
                context_points.append(f"- General legal context: {text[:180]}")
    if not context_points:
        context_points.append("- No closely matching legal text was found, so this is a broad orientation only.")

    return (
        f"Likely legal category:\n{category}\n\n"
        f"General legal context:\n{_category_context(category)}\n\n"
        "High-level guidance:\n"
        + "\n".join(context_points)
        + "\n\n"
        + _document_observations(retrieval)
        + "General next steps:\n"
        + _general_next_steps(category)
        + "\n\n"
        + f"Important note:\n{NON_ADVICE_NOTE}"
    )


def _detect_document_type(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    if ext in SUPPORTED_UPLOAD_EXTENSIONS:
        return ext.lstrip(".")
    raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


def _extract_txt_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _extract_pdf_text(data: bytes) -> str:
    try:
        pypdf_module = import_module("pypdf")
    except ImportError as exc:
        raise RuntimeError("PDF parsing dependency missing. Install pypdf.") from exc
    reader = pypdf_module.PdfReader(BytesIO(data))
    parts: List[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _extract_docx_text(data: bytes) -> str:
    try:
        docx_module = import_module("docx")
    except ImportError as exc:
        raise RuntimeError("DOCX parsing dependency missing. Install python-docx.") from exc
    document = docx_module.Document(BytesIO(data))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _extract_upload_text(filename: str, data: bytes) -> tuple[str, str]:
    document_type = _detect_document_type(filename)
    if document_type == "txt":
        text = _extract_txt_text(data)
    elif document_type == "pdf":
        text = _extract_pdf_text(data)
    else:
        text = _extract_docx_text(data)
    normalized = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    return document_type, normalized


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip().lower()).strip("-")
    return cleaned or "saulgpt-report"


def _build_pdf_bytes(title: str, content: str) -> bytes:
    try:
        canvas_module = import_module("reportlab.pdfgen.canvas")
        pagesizes_module = import_module("reportlab.lib.pagesizes")
    except ImportError as exc:
        raise RuntimeError("PDF export dependency missing. Install reportlab.") from exc

    buffer = BytesIO()
    canvas = canvas_module.Canvas(buffer, pagesize=pagesizes_module.A4)
    width, height = pagesizes_module.A4
    margin_x, margin_y = 48, 56
    text_obj = canvas.beginText(margin_x, height - margin_y)
    text_obj.setFont("Helvetica-Bold", 14)
    text_obj.textLine(title)
    text_obj.moveCursor(0, 10)
    text_obj.setFont("Helvetica", 11)

    max_chars = max(50, int((width - (2 * margin_x)) / 6))
    for paragraph in content.splitlines():
        wrapped = [paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)]
        if not wrapped:
            wrapped = [""]
        for line in wrapped:
            if text_obj.getY() <= margin_y:
                canvas.drawText(text_obj)
                canvas.showPage()
                text_obj = canvas.beginText(margin_x, height - margin_y)
                text_obj.setFont("Helvetica", 11)
            text_obj.textLine(line)
        text_obj.textLine("")
    canvas.drawText(text_obj)
    canvas.save()
    return buffer.getvalue()


def _build_docx_bytes(title: str, content: str) -> bytes:
    try:
        docx_module = import_module("docx")
    except ImportError as exc:
        raise RuntimeError("DOCX export dependency missing. Install python-docx.") from exc

    document = docx_module.Document()
    document.add_heading(title, level=1)
    for paragraph in content.split("\n\n"):
        text = paragraph.strip()
        if text:
            document.add_paragraph(text)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _build_context_block(context_docs: List[Dict[str, Any]]) -> str:
    if not context_docs:
        return "No directly matching references were retrieved."

    context_lines = []
    for i, doc in enumerate(context_docs, start=1):
        safe_text = _sanitize_reference_text(str(doc.get("text", "")))
        context_lines.append(
            f"{i}. ({str(doc.get('kind', 'reference')).upper()}) {safe_text} "
            f"(source: {doc.get('source', 'unknown')}, score: {doc.get('score', 0)})"
        )
    return "\n".join(context_lines)


def _call_ollama_text(prompt: str, num_predict: int) -> str:
    last_error: Optional[Exception] = None

    for model_name in MODEL_CANDIDATES:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": num_predict,
            },
        }

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
                response.raise_for_status()
                raw = response.json().get("response", "").strip()
                if raw:
                    return raw
                raise ValueError(f"empty model response from {model_name}")
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == MAX_RETRIES:
                    break
                time.sleep(0.4 * (attempt + 1))

    raise HTTPException(status_code=502, detail=f"model call failed: {last_error}")


def _chat_prompt(message: str, history: List[ChatTurn], context_docs: List[Dict[str, Any]]) -> str:
    recent_turns = history[-8:]
    history_lines = [f"{turn.role.upper()}: {turn.content}" for turn in recent_turns]
    history_block = "\n".join(history_lines) if history_lines else "No prior context."

    return f"""
You are SaulGPT, a legal information assistant for Indian law.
Follow these rules strictly:
- Do NOT provide specific legal advice.
- Do NOT cite exact sections, statute numbers, or legal provisions.
- Do NOT instruct the user to take specific legal action.
- Do NOT present yourself as a lawyer or legal authority.

Use these references first:
{_build_context_block(context_docs)}

Conversation so far:
{history_block}

User message:
{message}

Output format (plain text):
Likely legal category:
General legal context:
High-level guidance:
General next steps:
Important note:
""".strip()


def _build_generate_prompt(data: CaseRequest, context_docs: List[Dict[str, Any]]) -> str:
    return f"""
You are SaulGPT, a legal information assistant for Indian law.
You must provide only high-level legal information.
Do not provide specific legal advice.
Do not cite exact section numbers or statute provisions.
Do not recommend specific legal actions.

Retrieved legal context:
{_build_context_block(context_docs)}

Case details:
- Case type: {data.case_type}
- Incident: {data.incident}
- Amount involved: {data.amount or "Not specified"}

Output format:
- Likely legal category
- General legal context
- High-level guidance
- General next steps
- Important note that this is not legal advice
""".strip()


def _fallback_chat_reply(message: str, retrieval: List[Dict[str, Any]]) -> str:
    return _build_information_reply(message, retrieval)


def _needs_more_facts(message: str) -> bool:
    lower = message.lower()
    scenario_markers = {
        "my ", "i ", "we ", "our ", "accused", "incident", "happened", "paid", "payment",
        "took", "threat", "dispute", "arrest", "transaction", "landlord", "tenant",
        "notice received", "refusing", "stopped responding", "police refused",
    }
    if any(marker in lower for marker in scenario_markers):
        return True
    if re.search(r"\b(i|my|me|we|our)\b", lower):
        return True
    # General doctrinal/citizen-law Q&A usually does not require a missing-facts block.
    general_markers = {
        "explain", "difference", "what is", "what are", "overview", "define",
        "meaning", "penalty", "procedure", "rights", "remedies", "law for",
    }
    if any(marker in lower for marker in general_markers):
        return False
    return False


def _fallback_draft(data: CaseRequest, retrieval: List[Dict[str, Any]]) -> str:
    query = f"{data.case_type}. {data.incident}. Amount: {data.amount or 'Not specified'}"
    return _build_information_reply(query, retrieval)


def _clean_reply(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = _sanitize_reference_text(cleaned, collapse_whitespace=False)
    cleaned = re.sub(
        r"\b(file|register|lodge|initiate)\b[^.\n]*\b(complaint|fir|case|petition|suit)\b",
        "contact the appropriate authority after getting professional legal advice",
        cleaned,
        flags=re.IGNORECASE,
    )
    if NON_ADVICE_NOTE.lower() not in cleaned.lower():
        cleaned = f"{cleaned}\n\nImportant note:\n{NON_ADVICE_NOTE}".strip()
    return cleaned


def _finalize_chat_reply(message: str, reply: str) -> str:
    cleaned = _clean_reply(reply)
    if "Likely legal category:" not in cleaned:
        cleaned = _build_information_reply(message, [])
    return cleaned


def _chat_cache_key(message: str, history: List[ChatTurn]) -> str:
    recent = history[-6:]
    hist = "|".join(f"{t.role}:{t.content}" for t in recent)
    return f"{message.strip().lower()}||{hist.strip().lower()}"


def _get_cached_chat(message: str, history: List[ChatTurn]) -> Optional[ChatResponse]:
    key = _chat_cache_key(message, history)
    hit = _chat_cache.get(key)
    if not hit:
        return None
    ts, payload = hit
    if time.time() - ts > CHAT_CACHE_TTL_SECONDS:
        _chat_cache.pop(key, None)
        return None
    return ChatResponse(**payload)


def _set_cached_chat(message: str, history: List[ChatTurn], response: ChatResponse) -> None:
    key = _chat_cache_key(message, history)
    _chat_cache[key] = (time.time(), response.model_dump())


def _is_reply_grounded(reply: str, retrieval: List[Dict[str, Any]]) -> bool:
    if not retrieval:
        return True
    lower = reply.lower()
    for doc in retrieval[:5]:
        doc_id = str(doc.get("id", "")).lower()
        section = str(doc.get("section", "")).lower()
        if doc_id and doc_id in lower:
            return True
        if section and section in lower:
            return True
        numbers = re.findall(r"\d+[a-zA-Z]?", f"{doc_id} {section}")
        if any(n.lower() in lower for n in numbers):
            return True
    return False


def _disclaimer() -> str:
    return (
        "SaulGPT provides general legal information only and does not provide legal advice. "
        "For decisions in your specific matter, consult a qualified lawyer."
    )


@app.get("/")
def home() -> FileResponse:
    return FileResponse(INDEX_FILE)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_candidates": MODEL_CANDIDATES,
        "timeout_seconds": OLLAMA_TIMEOUT_SECONDS,
        "chat_cache_ttl_seconds": CHAT_CACHE_TTL_SECONDS,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(data: CaseRequest, request: Request, x_api_key: Optional[str] = Header(default=None)) -> GenerateResponse:
    _check_api_key(x_api_key)
    client_id = request.client.host if request.client else "unknown"
    _check_rate_limit(client_id)

    query = f"{data.case_type} {data.incident} {data.amount or ''}".strip()
    retrieval = search_knowledge(query, top_k=7)
    draft = _build_information_reply(query, retrieval)

    return GenerateResponse(
        draft=_clean_reply(draft),
        citations=[],
        disclaimer=_disclaimer(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest, request: Request, x_api_key: Optional[str] = Header(default=None)) -> ChatResponse:
    _check_api_key(x_api_key)
    client_id = request.client.host if request.client else "unknown"
    _check_rate_limit(client_id)

    cached = _get_cached_chat(data.message, data.history)
    if cached:
        return cached

    if not _is_law_topic(data.message):
        response = ChatResponse(
            reply=(
                f"{LAW_FOCUS_GUIDANCE} "
                "You can ask things like: 'What type of legal issue is this?', "
                "'What is the general legal context?', or 'What documents should I gather?'"
            ),
            citations=[],
            redirected_to_law=True,
            disclaimer=_disclaimer(),
        )
        _set_cached_chat(data.message, data.history, response)
        return response

    user_history_context = " ".join(
        turn.content for turn in data.history[-6:] if turn.role == "user"
    )
    retrieval_query = f"{user_history_context} {data.message}".strip()
    retrieval = search_knowledge(retrieval_query, top_k=7)
    reply = _build_information_reply(data.message, retrieval)

    response = ChatResponse(
        reply=_finalize_chat_reply(data.message, reply),
        citations=[],
        redirected_to_law=False,
        disclaimer=_disclaimer(),
    )
    _set_cached_chat(data.message, data.history, response)
    return response


@app.post("/api/chat", response_model=ChatResponse)
def chat_api_alias(
    data: ChatRequest, request: Request, x_api_key: Optional[str] = Header(default=None)
) -> ChatResponse:
    return chat(data, request, x_api_key)


@app.post("/documents/ingest", response_model=IngestResponse)
@app.post("/api/documents/ingest", response_model=IngestResponse)
async def documents_ingest(
    request: Request,
    files: List[UploadFile] = File(...),
    x_api_key: Optional[str] = Header(default=None),
) -> IngestResponse:
    _check_api_key(x_api_key)
    client_id = request.client.host if request.client else "unknown"
    _check_rate_limit(client_id)

    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    results: List[IngestFileResult] = []
    ingested_files = 0
    total_chunks = 0

    for file in files:
        filename = (file.filename or "uploaded_document.txt").strip()
        try:
            raw = await file.read()
            if not raw:
                raise ValueError("File is empty.")
            if len(raw) > MAX_UPLOAD_BYTES:
                raise ValueError("File is too large. Max supported size is 10 MB.")

            document_type, text = _extract_upload_text(filename, raw)
            if not text:
                raise ValueError("No readable text found in file.")

            chunks = _chunk_text(text)
            if not chunks:
                raise ValueError("Could not split document into chunks.")

            added = add_uploaded_document_chunks(filename, document_type, chunks)
            ingested_files += 1
            total_chunks += added
            results.append(
                IngestFileResult(
                    filename=filename,
                    document_type=document_type,
                    status="ingested",
                    chunks=added,
                    detail="Indexed successfully.",
                )
            )
        except Exception as exc:
            results.append(
                IngestFileResult(
                    filename=filename,
                    document_type=None,
                    status="failed",
                    chunks=0,
                    detail=str(exc),
                )
            )

    if ingested_files == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents were ingested. Check file type and parsing dependencies.",
        )

    return IngestResponse(
        total_files=len(files),
        ingested_files=ingested_files,
        failed_files=len(files) - ingested_files,
        total_chunks=total_chunks,
        results=results,
    )


@app.post("/reports/export")
@app.post("/api/reports/export")
def export_report(
    data: ReportExportRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
) -> Response:
    _check_api_key(x_api_key)
    client_id = request.client.host if request.client else "unknown"
    _check_rate_limit(client_id)

    title = data.title.strip() or "SaulGPT Legal Report"
    content = data.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Report content cannot be empty.")

    base_name = _safe_filename(data.filename or title)

    try:
        if data.format == "pdf":
            payload = _build_pdf_bytes(title, content)
            media_type = "application/pdf"
            filename = f"{base_name}.pdf"
        else:
            payload = _build_docx_bytes(title, content)
            media_type = (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            filename = f"{base_name}.docx"
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(
        content=payload,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
