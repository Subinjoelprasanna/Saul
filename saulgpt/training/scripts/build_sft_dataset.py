#!/usr/bin/env python3
"""Build an SFT dataset for SaulGPT legal assistant fine-tuning.

Outputs ChatML-style JSONL with `messages` field.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

SYSTEM_PROMPT = (
    "You are SaulGPT, a legal information assistant for Indian law. "
    "Be concise and practical. Do not provide legal advice. "
    "Do not cite exact legal sections, acts, or statute numbers. "
    "Identify the legal category, explain general context, ask only essential follow-up questions, "
    "and generate structured reports only after enough facts are available."
)

CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Property or tenancy issue": (
        "property", "tenant", "landlord", "lease", "rent", "deed", "title", "ownership", "mutation", "encroachment"
    ),
    "Employment dispute": (
        "salary", "employment", "employer", "employee", "dues", "termination", "hr", "payslip"
    ),
    "Consumer issue": (
        "consumer", "defect", "seller", "service", "refund", "warranty", "purchase", "delivery"
    ),
    "Family or relationship dispute": (
        "family", "maintenance", "custody", "marriage", "inheritance", "domestic", "separation", "will"
    ),
    "Cyber or online harm": (
        "cyber", "upi", "otp", "phishing", "hacked", "online", "digital", "account", "impersonation"
    ),
    "Fraud or cheating concern": (
        "fraud", "cheat", "scam", "forgery", "deception", "misrepresentation"
    ),
    "Contract or payment dispute": (
        "contract", "agreement", "breach", "payment", "vendor", "settlement", "invoice"
    ),
    "General criminal concern": (
        "assault", "threat", "criminal", "theft", "intimidation", "violence", "police"
    ),
}

DOC_CHECKLIST: Dict[str, List[str]] = {
    "Property or tenancy issue": [
        "ownership/tenancy documents",
        "rent or lease records",
        "payment proofs",
        "messages/emails with counterparty",
    ],
    "Employment dispute": [
        "offer/appointment documents",
        "salary records and bank credits",
        "attendance/work proof",
        "HR and manager communication",
    ],
    "Consumer issue": [
        "invoice/order details",
        "warranty or service terms",
        "defect/service-failure proof",
        "complaint and response trail",
    ],
    "Family or relationship dispute": [
        "family/relationship records",
        "financial support records",
        "communications and timeline notes",
        "identity and residence proofs",
    ],
    "Cyber or online harm": [
        "transaction IDs and account alerts",
        "screenshots with timestamps",
        "device/app/email logs",
        "communication with bank/platform",
    ],
    "Fraud or cheating concern": [
        "transaction/payment trail",
        "representations/promises made",
        "identity/account details of counterparty",
        "chat/call/email records",
    ],
    "Contract or payment dispute": [
        "agreement/contract copy",
        "invoices and payment trail",
        "breach communications",
        "loss/impact computation",
    ],
    "General criminal concern": [
        "incident timeline",
        "injury/loss evidence",
        "witness details",
        "communication and complaint records",
    ],
}

FOLLOWUPS: Dict[str, List[str]] = {
    "Property or tenancy issue": [
        "What is the property/tenancy detail and location?",
        "What documents and payment proofs do you currently have?",
    ],
    "Employment dispute": [
        "What is your employment period and role?",
        "Which months are unpaid and what payroll records do you have?",
    ],
    "Consumer issue": [
        "When was the purchase/service and what failed?",
        "Do you have invoice, warranty terms, and complaint responses?",
    ],
    "Family or relationship dispute": [
        "What is the relationship context and timeline?",
        "What records support your version of events?",
    ],
    "Cyber or online harm": [
        "What exact digital event happened and when?",
        "Which transaction/account evidence is available?",
    ],
    "Fraud or cheating concern": [
        "What representation was made and how did reliance occur?",
        "What payment and communication trail do you have?",
    ],
    "Contract or payment dispute": [
        "Which contract term was breached?",
        "What loss occurred and what proof supports it?",
    ],
    "General criminal concern": [
        "When and where did the incident occur?",
        "What evidence and witnesses are available?",
    ],
}

NEXT_STEPS: Dict[str, List[str]] = {
    "Property or tenancy issue": [
        "Prepare a clean date-wise timeline.",
        "Organize ownership/tenancy and payment records.",
        "Keep all communication in one evidence folder.",
    ],
    "Employment dispute": [
        "Prepare month-wise dues calculation.",
        "Collect payroll, bank, and HR records.",
        "Document all communication dates clearly.",
    ],
    "Consumer issue": [
        "Prepare purchase-to-complaint timeline.",
        "Keep invoice, warranty, and defect proof ready.",
        "Store all complaint responses with dates.",
    ],
    "Family or relationship dispute": [
        "Prepare chronology of major events.",
        "Collect family and financial support records.",
        "Preserve communication and residence evidence.",
    ],
    "Cyber or online harm": [
        "Preserve digital evidence without editing.",
        "Track every transaction ID and timestamp.",
        "Keep bank/platform complaint references.",
    ],
    "Fraud or cheating concern": [
        "Map promise -> payment -> loss timeline.",
        "Collect identity/account and transaction records.",
        "Preserve all communications and screenshots.",
    ],
    "Contract or payment dispute": [
        "Map obligations and breach events date-wise.",
        "Compile invoices, payments, and notices.",
        "Compute loss with supporting records.",
    ],
    "General criminal concern": [
        "Prepare short factual chronology.",
        "Organize evidence and witness details.",
        "Maintain dated record of all follow-ups.",
    ],
}

REPORT_SCENARIOS = [
    {
        "category": "Property or tenancy issue",
        "title": "Tenant Deposit Dispute",
        "background": "Tenant vacated a rented flat; landlord withheld security deposit without clear accounting.",
        "evidence": "Lease copy, payment receipts, handover messages, move-out photos.",
        "witness": "Building caretaker observed handover condition.",
        "timeline": "Lease start, notice to vacate, handover date, deposit demand communications.",
    },
    {
        "category": "Employment dispute",
        "title": "Unpaid Salary and Settlement",
        "background": "Employee resigned; two months salary and final settlement remain unpaid.",
        "evidence": "Offer letter, payslips, bank statements, resignation and HR emails.",
        "witness": "Team lead aware of final handover and pending payroll.",
        "timeline": "Joining date, salary due months, resignation date, settlement follow-ups.",
    },
    {
        "category": "Consumer issue",
        "title": "Defective Product Refund",
        "background": "Consumer received defective product; seller denied replacement/refund despite repeated requests.",
        "evidence": "Invoice, product photos/videos, warranty terms, complaint ticket history.",
        "witness": "Family member saw unboxing and defect.",
        "timeline": "Purchase date, defect discovery, complaint dates, denial responses.",
    },
    {
        "category": "Cyber or online harm",
        "title": "UPI Scam Loss",
        "background": "User was induced into a fraudulent UPI transfer through social engineering.",
        "evidence": "Transaction IDs, call logs, screenshots, account alerts.",
        "witness": "Friend present during call and transfer.",
        "timeline": "Fraud contact time, transfer time, bank/platform reporting times.",
    },
    {
        "category": "Family or relationship dispute",
        "title": "Maintenance and Support Conflict",
        "background": "Dispute over financial support obligations after separation.",
        "evidence": "Income indicators, expense records, communication trail.",
        "witness": "Relative aware of support discussions.",
        "timeline": "Separation date, support requests, partial payments, subsequent non-payment.",
    },
    {
        "category": "Contract or payment dispute",
        "title": "Vendor Non-Performance",
        "background": "Service provider failed to deliver agreed milestones after receiving advance payment.",
        "evidence": "Contract, invoices, payment proofs, milestone communications.",
        "witness": "Project manager tracked missed milestones.",
        "timeline": "Agreement date, payment date, missed milestone dates, demand notices.",
    },
]

OUT_OF_SCOPE = [
    "What is the best protein powder for gym?",
    "Can you suggest vacation spots in Europe?",
    "Write a birthday poem for my friend.",
    "What phone should I buy under 30000?",
]

GENERAL_PROMPTS = [
    "Explain this in simple terms: {topic}",
    "Help me understand this legal issue: {topic}",
    "What does this mean in practical language: {topic}?",
    "I am confused about this issue: {topic}. Explain simply.",
]

CATEGORY_PROMPTS = [
    "What category of legal issue is this: {topic}?",
    "Classify this issue for me: {topic}",
    "Is this mostly property, consumer, family, criminal, or something else: {topic}?",
]

DOCUMENT_PROMPTS = [
    "What documents should I gather for this: {topic}?",
    "What records are usually useful in this type of issue: {topic}?",
    "Tell me the evidence checklist for: {topic}",
]

NEXT_STEP_PROMPTS = [
    "Give high-level next steps for this issue: {topic}",
    "What should I do next at a general level for: {topic}?",
    "How should I organize this matter before speaking to a lawyer: {topic}?",
]

REPORT_PREP_PROMPTS = [
    "I want a legal report for this issue: {topic}. What details do you need first?",
    "Before generating a case report on {topic}, ask me the essential questions only.",
]

FACT_PARTIALS: Dict[str, str] = {
    "Property or tenancy issue": "The dispute started in Jan 2026 in Bengaluru.",
    "Employment dispute": "I worked from 2022 to 2025 as a sales manager.",
    "Consumer issue": "I bought the product in Feb 2026 and it failed in one week.",
    "Family or relationship dispute": "The separation happened in 2025 and support discussions started after that.",
    "Cyber or online harm": "The transaction happened on 03 Mar 2026 through UPI.",
    "Fraud or cheating concern": "The promise was made in Dec 2025 and payment was done online.",
    "Contract or payment dispute": "The service agreement began in Nov 2025.",
    "General criminal concern": "The incident happened in the evening near my residence.",
}

FACT_COMPLETIONS: Dict[str, str] = {
    "Property or tenancy issue": "I have lease papers, rent receipts, and move-out chat records.",
    "Employment dispute": "Unpaid dues are for Nov and Dec; I have payslips, bank credits, and HR emails.",
    "Consumer issue": "I have invoice, warranty details, product photos, and complaint ticket responses.",
    "Family or relationship dispute": "I have financial records, key messages, and expense notes.",
    "Cyber or online harm": "I have transaction IDs, screenshots, call logs, and bank complaint reference.",
    "Fraud or cheating concern": "I have transfer records, chats, and account details used for payment.",
    "Contract or payment dispute": "I have contract copy, invoices, payment proofs, and breach emails.",
    "General criminal concern": "I have timeline notes, photos, and contact details of one witness.",
}


def sanitize_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\b(?:section|sec\.?|article)\s*\d+[a-zA-Z]?\b", "a relevant provision", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:ipc|crpc|cpc|bns|bnss|bsa)\s*\d*[a-zA-Z]?\b", "the legal framework", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b[A-Z][A-Za-z&' .-]{1,80}\s+(?:Act|Code|Rules?|Regulations?)\b", "the relevant legal framework", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def infer_category(text: str) -> str:
    lower = text.lower()
    best = "General legal issue"
    best_score = 0
    for category, kws in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in lower)
        if score > best_score:
            best_score = score
            best = category
    return best


def topic_from_record(section: str, text: str) -> str:
    for candidate in [section, text]:
        c = sanitize_text(candidate)
        if 8 <= len(c) <= 70:
            return c
    return sanitize_text(section)[:70]


def parse_law_files(files: List[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    for file_path in files:
        if not file_path.exists():
            continue
        for line in file_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [part.strip() for part in raw.split("|")]
            if len(parts) != 4:
                continue
            section, text, source, effective_date = parts
            key = (section, text, source, effective_date)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "section": section,
                    "text": sanitize_text(text),
                    "source": source,
                    "effective_date": effective_date,
                }
            )
    return rows


def compact_context(text: str, max_len: int = 220) -> str:
    cleaned = sanitize_text(text)
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "."


def concise_general_response(category: str, context: str, include_questions: bool = True, style: int = 0) -> str:
    context = compact_context(context)
    questions = FOLLOWUPS.get(category, FOLLOWUPS["Contract or payment dispute"])
    variant = style % 3
    if variant == 0:
        lines = [
            f"Category: {category}",
            f"General context: {context}",
        ]
        if include_questions:
            lines.append("To improve accuracy, share:")
            lines.append(f"1. {questions[0]}")
            lines.append(f"2. {questions[1]}")
        return "\n".join(lines)
    if variant == 1:
        lines = [
            f"Likely category: {category}",
            f"In general: {context}",
        ]
        if include_questions:
            lines.append("If you want a detailed report next, please add:")
            lines.append(f"- {questions[0]}")
            lines.append(f"- {questions[1]}")
        return "\n".join(lines)

    lines = [
        f"This appears to be a {category.lower()}.",
        f"General legal context: {context}",
    ]
    if include_questions:
        lines.append(f"Needed facts: {questions[0]} Also, {questions[1]}")
    return "\n".join(lines)


def checklist_response(category: str, context: str, style: int = 0) -> str:
    context = compact_context(context)
    docs = DOC_CHECKLIST.get(category, DOC_CHECKLIST["Contract or payment dispute"])
    variant = style % 3
    if variant == 0:
        lines = [
            f"Category: {category}",
            f"General context: {context}",
            "Useful records:",
        ]
        for doc in docs[:4]:
            lines.append(f"- {doc}")
        lines.append("If you want, I can convert your facts into a structured report.")
        return "\n".join(lines)

    if variant == 1:
        lines = [
            f"Likely category: {category}",
            f"Context: {context}",
            "Keep these ready:",
        ]
        for idx, doc in enumerate(docs[:4], start=1):
            lines.append(f"{idx}. {doc}")
        lines.append("Share what you already have and I can draft the report.")
        return "\n".join(lines)

    lines = [
        f"This looks like {category.lower()}.",
        "Evidence checklist:",
    ]
    for doc in docs[:4]:
        lines.append(f"- {doc}")
    lines.append("Optional: ask me to generate a structured case report from these facts.")
    return "\n".join(lines)


def next_steps_response(category: str, context: str, style: int = 0) -> str:
    context = compact_context(context)
    steps = NEXT_STEPS.get(category, NEXT_STEPS["Contract or payment dispute"])
    variant = style % 3
    if variant == 0:
        lines = [
            f"Category: {category}",
            f"General context: {context}",
            "High-level next steps:",
        ]
        for step in steps[:3]:
            lines.append(f"- {step}")
        lines.append("Disclaimer: General legal information only, not legal advice.")
        return "\n".join(lines)

    if variant == 1:
        lines = [
            f"Likely category: {category}",
            "Suggested high-level steps:",
        ]
        for idx, step in enumerate(steps[:3], start=1):
            lines.append(f"{idx}. {step}")
        lines.append("This is general information and not legal advice.")
        return "\n".join(lines)

    lines = [
        f"General context: {context}",
        "A practical sequence is:",
    ]
    for step in steps[:3]:
        lines.append(f"- {step}")
    lines.append("Disclaimer: informational support only.")
    return "\n".join(lines)


def report_intake_response(category: str, context: str, style: int = 0) -> str:
    context = compact_context(context)
    questions = FOLLOWUPS.get(category, FOLLOWUPS["Contract or payment dispute"])
    variant = style % 2
    if variant == 0:
        lines = [
            "Stage 2 - Collect key facts",
            f"Category: {category}",
            f"General context: {context}",
            "To generate a structured report now, I only need:",
            f"1. {questions[0]}",
            f"2. {questions[1]}",
        ]
        return "\n".join(lines)

    lines = [
        f"Category identified: {category}",
        "Before report generation, please confirm:",
        f"- {questions[0]}",
        f"- {questions[1]}",
        "Once shared, I will move directly to report drafting.",
    ]
    return "\n".join(lines)


def build_single_turn_examples(records: List[Dict[str, str]]) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    for idx, row in enumerate(records):
        section = row["section"]
        text = row["text"]
        category = infer_category(f"{section} {text}")
        topic = topic_from_record(section, text)

        prompt_modes = [
            ("general", GENERAL_PROMPTS[idx % len(GENERAL_PROMPTS)]),
            ("category", CATEGORY_PROMPTS[(idx + 1) % len(CATEGORY_PROMPTS)]),
            ("documents", DOCUMENT_PROMPTS[(idx + 2) % len(DOCUMENT_PROMPTS)]),
            ("next_steps", NEXT_STEP_PROMPTS[(idx + 3) % len(NEXT_STEP_PROMPTS)]),
        ]
        if idx % 4 == 0:
            prompt_modes.append(("report_prep", REPORT_PREP_PROMPTS[idx % len(REPORT_PREP_PROMPTS)]))

        for sub_idx, (mode, template) in enumerate(prompt_modes):
            user_prompt = template.format(topic=topic)
            style = idx + sub_idx
            if mode == "documents":
                assistant = checklist_response(category, text, style=style)
            elif mode == "next_steps":
                assistant = next_steps_response(category, text, style=style)
            elif mode == "report_prep":
                assistant = report_intake_response(category, text, style=style)
            else:
                assistant = concise_general_response(
                    category,
                    text,
                    include_questions=(style % 2 == 0),
                    style=style,
                )

            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant},
                    ]
                }
            )
    return examples


def build_structured_report(
    title: str,
    category: str,
    background: str,
    evidence: str,
    witness: str,
    timeline: str,
) -> str:
    return (
        f"Stage 3 - Generate structured report\n\n"
        f"Title\nSaulGPT Structured Legal Intake Report - {title}\n\n"
        f"Legal Category\n{category}\n\n"
        f"Background of the Issue\n- {background}\n\n"
        f"Summary of Evidence\n- {evidence}\n\n"
        f"Witness Information\n- {witness}\n\n"
        f"Document Evidence\n- Agreements, records, and communication trail provided by user.\n\n"
        f"Timeline of Events\n- {timeline}\n\n"
        f"General Legal Context\nThis issue usually concerns {category.lower()} and depends on documented facts and chronology.\n\n"
        f"Suggested High-Level Next Steps\n"
        f"- Build a date-wise chronology\n"
        f"- Organize all supporting records\n"
        f"- Seek professional legal review before decisions\n\n"
        f"Disclaimer\nGeneral legal information only, not legal advice."
    )


def build_report_examples() -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    for item in REPORT_SCENARIOS:
        category = item["category"]
        title = item["title"]
        report = build_structured_report(
            title=title,
            category=category,
            background=item["background"],
            evidence=item["evidence"],
            witness=item["witness"],
            timeline=item["timeline"],
        )

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Generate a legal report for this case: {item['background']} I have {item['evidence']}",
                    },
                    {"role": "assistant", "content": report},
            ]
        }
    )

    for idx, item in enumerate(REPORT_SCENARIOS):
        category = item["category"]
        followups = FOLLOWUPS.get(category, FOLLOWUPS["Contract or payment dispute"])
        partial = FACT_PARTIALS.get(category, "The issue started recently.")
        completion = FACT_COMPLETIONS.get(category, "I have key records and communications.")
        report = build_structured_report(
            title=item["title"],
            category=category,
            background=item["background"],
            evidence=item["evidence"],
            witness=item["witness"],
            timeline=item["timeline"],
        )
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"I need a report for this issue: {item['background']}"},
                    {
                        "role": "assistant",
                        "content": (
                            "Stage 2 - Collect key facts\n"
                            f"Category: {category}\n"
                            "Still needed:\n"
                            f"1. {followups[0]}\n"
                            f"2. {followups[1]}"
                        ),
                    },
                    {"role": "user", "content": partial},
                    {
                        "role": "assistant",
                        "content": (
                            "Stage 2 - Collect key facts\n"
                            f"Category: {category}\n"
                            "Captured your first update.\n"
                            "Still needed for report generation:\n"
                            f"1. {followups[1]}\n"
                            "Share this and I will generate the report."
                        ),
                    },
                    {"role": "user", "content": completion},
                    {"role": "assistant", "content": report},
                ]
            }
        )
        if idx % 2 == 0:
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Report now for this case: {item['background']} I already have {item['evidence']}",
                        },
                        {"role": "assistant", "content": report},
                    ]
                }
            )

    return examples


def build_multiturn_workflow_examples() -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []

    examples.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Generate report for landlord deposit dispute"},
                {
                    "role": "assistant",
                    "content": (
                        "Stage 2 - Collect key facts\n"
                        "Category: Property or tenancy issue\n"
                        "Still needed: Incident date, Incident location, Document evidence, Witness information\n"
                        "Answer these next:\n"
                        "1. What is the incident date and location?\n"
                        "2. What tenancy/payment documents do you have?"
                    ),
                },
                {
                    "role": "user",
                    "content": "Jan 2026 in Bengaluru. I have lease copy and payment receipts. No witnesses.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Stage 3 - Generate structured report\n\n"
                        "Title\nSaulGPT Structured Legal Intake Report - Tenant Deposit Dispute\n\n"
                        "Legal Category\nProperty or tenancy issue\n\n"
                        "Background of the Issue\n- Tenant states deposit was withheld after vacating premises.\n\n"
                        "Summary of Evidence\n- Lease copy and payment receipts provided.\n\n"
                        "Witness Information\n- No witnesses identified.\n\n"
                        "Document Evidence\n- Lease copy, receipts, communication records.\n\n"
                        "Timeline of Events\n- Jan 2026: dispute arose after move-out.\n\n"
                        "General Legal Context\nThis issue usually concerns tenancy obligations, possession status, and documentary payment trail.\n\n"
                        "Suggested High-Level Next Steps\n- Keep chronology and evidence indexed.\n- Preserve all communication records.\n- Seek legal review of dispute facts.\n\n"
                        "Disclaimer\nGeneral legal information only, not legal advice."
                    ),
                },
            ]
        }
    )

    examples.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What is section 420 IPC exactly?"},
                {
                    "role": "assistant",
                    "content": (
                        "Category: Fraud or cheating concern\n"
                        "General context: This usually concerns alleged deception causing financial or practical loss.\n"
                        "To improve accuracy, share:\n"
                        "1. What transaction or promise was made?\n"
                        "2. What payment and communication records are available?"
                    ),
                },
            ]
        }
    )

    examples.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Create a case report about unpaid salary by my employer"},
                {
                    "role": "assistant",
                    "content": (
                        "Stage 2 - Collect key facts\n"
                        "Category: Employment dispute\n"
                        "Still needed:\n"
                        "1. What is your employment period and role?\n"
                        "2. Which months are unpaid and what payroll records do you have?"
                    ),
                },
                {"role": "user", "content": "I worked from 2021 to 2025 as an analyst."},
                {
                    "role": "assistant",
                    "content": (
                        "Stage 2 - Collect key facts\n"
                        "Category: Employment dispute\n"
                        "Captured: employment period and role.\n"
                        "Still needed:\n"
                        "1. Which months are unpaid and what payroll records do you have?"
                    ),
                },
                {
                    "role": "user",
                    "content": "Salary for Nov and Dec is unpaid. I have offer letter, payslips, bank statements, and HR emails.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Stage 3 - Generate structured report\n\n"
                        "Title\nSaulGPT Structured Legal Intake Report - Unpaid Salary and Settlement\n\n"
                        "Legal Category\nEmployment dispute\n\n"
                        "Background of the Issue\n- User reports unpaid salary and pending settlement after resignation.\n\n"
                        "Summary of Evidence\n- Offer letter, payslips, bank statements, and HR emails.\n\n"
                        "Witness Information\n- No witness confirmed yet.\n\n"
                        "Document Evidence\n- Employment records and communication trail.\n\n"
                        "Timeline of Events\n- Employment period completed; unpaid months are Nov and Dec.\n\n"
                        "General Legal Context\nThis generally concerns wage obligations, separation settlement, and documentary payroll trail.\n\n"
                        "Suggested High-Level Next Steps\n- Keep month-wise due computation.\n- Preserve all payroll and HR records.\n- Seek professional legal review.\n\n"
                        "Disclaimer\nGeneral legal information only, not legal advice."
                    ),
                },
            ]
        }
    )

    examples.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "I uploaded rent_agreement.pdf and payment_receipts.docx. Help me."},
                {
                    "role": "assistant",
                    "content": (
                        "Category: Property or tenancy issue\n"
                        "General context: This usually involves tenancy terms, payment trail, and possession/handover facts.\n"
                        "I can use uploaded documents. To proceed, share:\n"
                        "1. Key dispute point (for example deposit withheld, eviction pressure, or rent claim)\n"
                        "2. Timeline (move-in, notice, handover, latest communication)"
                    ),
                },
                {
                    "role": "user",
                    "content": "Deposit is withheld after move-out. Timeline: moved out Jan 2026, requested refund in Feb and Mar.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Stage 3 - Generate structured report\n\n"
                        "Title\nSaulGPT Structured Legal Intake Report - Tenant Deposit Dispute\n\n"
                        "Legal Category\nProperty or tenancy issue\n\n"
                        "Background of the Issue\n- Deposit allegedly withheld after move-out.\n\n"
                        "Summary of Evidence\n- Uploaded rent agreement and payment receipts, plus user timeline.\n\n"
                        "Witness Information\n- Not provided.\n\n"
                        "Document Evidence\n- Rent agreement, receipts, and communication records.\n\n"
                        "Timeline of Events\n- Jan 2026 move-out; refund requests in Feb and Mar 2026.\n\n"
                        "General Legal Context\nThis generally concerns tenancy terms, settlement at handover, and documentary payment trail.\n\n"
                        "Suggested High-Level Next Steps\n- Preserve all landlord communication.\n- Keep date-wise chronology.\n- Seek legal review with complete records.\n\n"
                        "Disclaimer\nGeneral legal information only, not legal advice."
                    ),
                },
            ]
        }
    )

    return examples


def build_out_of_scope_examples() -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    reply = (
        "I can help with Indian legal issues and legal report drafting. "
        "Please reframe your question as a legal issue, and I will assist."
    )
    for prompt in OUT_OF_SCOPE:
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": reply},
                ]
            }
        )
    return examples


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SaulGPT SFT dataset")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--train-out", type=Path, default=None)
    parser.add_argument("--eval-out", type=Path, default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    law_files = [project_root / "laws.txt", project_root / "laws_expanded.txt"]

    rows = parse_law_files(law_files)
    examples = []
    examples.extend(build_single_turn_examples(rows))
    examples.extend(build_report_examples())
    examples.extend(build_multiturn_workflow_examples())
    examples.extend(build_out_of_scope_examples())

    random.seed(args.seed)
    random.shuffle(examples)

    eval_count = max(20, int(len(examples) * args.eval_ratio))
    eval_rows = examples[:eval_count]
    train_rows = examples[eval_count:]

    train_out = args.train_out or (project_root / "training" / "data" / "saulgpt_sft_train.jsonl")
    eval_out = args.eval_out or (project_root / "training" / "data" / "saulgpt_sft_eval.jsonl")

    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)

    print(f"Total examples: {len(examples)}")
    print(f"Train examples: {len(train_rows)} -> {train_out}")
    print(f"Eval examples: {len(eval_rows)} -> {eval_out}")


if __name__ == "__main__":
    main()
