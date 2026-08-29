"""
SETU Learn Service — High-level educational AI actions:
  - Explain Simply
  - Multi-mode Summarize (Quick, Key Points, Detailed)
  - Grounded RAG Q&A with Source Citations
  - Structured Accessible Quiz Generation (5 or 10 MCQs)
"""
from __future__ import annotations
import json
import logging
import re
import time
from typing import Dict, Any, List, Optional

import httpx

from server import config
from server.learn.document import ParsedDocument, extractor
from server.learn.retriever import registry, DocumentChunk

log = logging.getLogger("setu.learn.service")

# In-memory document storage for the active session
_DOCUMENTS: Dict[str, ParsedDocument] = {}


LEARN_SYSTEM_PROMPT = (
    "You are SETU Learn, an accessibility-focused educational assistant for blind and low-vision students.\n"
    "Use clear, natural language suitable for text-to-speech.\n"
    "Base every answer only on the supplied study material.\n"
    "Never invent or assume information.\n"
    "If sufficient information is unavailable in the material, clearly say:\n"
    "'This information was not found in the uploaded material.'\n"
    "Avoid relying on diagrams or visual positioning (do not say 'as shown above').\n"
    "If a concept depends on a visual representation, explain it sequentially in plain words.\n"
    "Be concise and direct."
)


async def _call_ollama(prompt: str, system: str = LEARN_SYSTEM_PROMPT, temperature: float = 0.2) -> str:
    """Calls local Ollama instance with fallback error handling."""
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": temperature,
            "num_predict": 700,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(f"{config.OLLAMA_URL}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
    except Exception as e:
        log.error("Ollama query failed: %s", e)
        raise RuntimeError(f"Ollama local model is currently unavailable: {e}")


class LearnService:
    """Core educational reasoning engine."""

    def save_document(self, doc: ParsedDocument) -> None:
        _DOCUMENTS[doc.document_id] = doc
        registry.add_document(doc)

    def get_document(self, doc_id: str) -> Optional[ParsedDocument]:
        return _DOCUMENTS.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        if doc_id in _DOCUMENTS:
            del _DOCUMENTS[doc_id]
            registry.remove_document(doc_id)
            return True
        return False

    async def explain_simply(self, doc_id: str, section_id: Optional[str] = None, text_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Explains the selected section or document in simple, jargon-free language.
        """
        doc = self.get_document(doc_id)
        if not doc and not text_override:
            raise ValueError("Document not found.")

        target_text = ""
        heading = "Document"
        if text_override:
            target_text = text_override
            heading = "Custom Selection"
        elif doc:
            if section_id:
                for s in doc.sections:
                    if s.section_id == section_id:
                        target_text = s.full_text
                        heading = s.heading
                        break
            if not target_text:
                target_text = doc.sections[0].full_text if doc.sections else doc.full_text
                heading = doc.sections[0].heading if doc.sections else doc.title

        # Truncate to reasonable context window if huge
        target_text = target_text[:3500]

        prompt = (
            f"Study Material Section: {heading}\n\n"
            f"{target_text}\n\n"
            f"Task: Explain this section in simple, crystal-clear language for a visually impaired student. "
            f"Avoid unnecessary technical jargon, keep sentences short and easy to understand when read aloud, "
            f"and preserve all key educational concepts accurately."
        )

        t0 = time.monotonic()
        explanation = await _call_ollama(prompt, system=LEARN_SYSTEM_PROMPT, temperature=0.2)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        return {
            "document_id": doc_id,
            "section_id": section_id,
            "heading": heading,
            "explanation": explanation,
            "latency_ms": latency_ms,
        }

    async def summarize(self, doc_id: str, section_id: Optional[str] = None, mode: str = "quick") -> Dict[str, Any]:
        """
        Generates grounded summaries: 'quick' (3-5 points), 'key_points', or 'detailed'.
        """
        doc = self.get_document(doc_id)
        if not doc:
            raise ValueError("Document not found.")

        target_text = ""
        heading = doc.title
        if section_id:
            for s in doc.sections:
                if s.section_id == section_id:
                    target_text = s.full_text
                    heading = s.heading
                    break

        if not target_text:
            target_text = doc.full_text[:4000]

        if mode == "quick":
            prompt_instruction = "Provide a QUICK SUMMARY of this material in exactly 3 to 5 clear, spoken bullet points."
        elif mode == "key_points":
            prompt_instruction = "Extract the essential KEY POINTS and definitions that a student must memorize from this material."
        else:  # detailed
            prompt_instruction = "Provide a structured, DETAILED SUMMARY covering all concepts and sub-topics in this section."

        prompt = (
            f"Material for {heading}:\n\n"
            f"{target_text[:3500]}\n\n"
            f"Task: {prompt_instruction} Stay 100% faithful to the text. Do not invent external facts."
        )

        t0 = time.monotonic()
        summary = await _call_ollama(prompt, system=LEARN_SYSTEM_PROMPT, temperature=0.15)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        return {
            "document_id": doc_id,
            "section_id": section_id,
            "mode": mode,
            "heading": heading,
            "summary": summary,
            "latency_ms": latency_ms,
        }

    async def ask_grounded(self, doc_id: str, question: str) -> Dict[str, Any]:
        """
        Answers questions strictly grounded in the document with source citations.
        """
        doc = self.get_document(doc_id)
        if not doc:
            raise ValueError("Document not found.")

        idx = registry.get_index(doc_id)
        if not idx:
            raise ValueError("Document index not ready.")

        # 1. Retrieve top relevant chunks
        results = idx.retrieve(question, top_k=3)
        if not results:
            return {
                "answer": "This information was not found in the uploaded material.",
                "found": False,
                "source": None,
            }

        top_chunk, top_score = results[0]

        # Assemble retrieved context
        context_blocks = []
        sources = []
        for chk, score in results:
            if score > 0.02:
                context_blocks.append(f"--- [Page {chk.page}, Section: {chk.heading}] ---\n{chk.text}")
                sources.append(f"Page {chk.page}, Section: {chk.heading}")

        if not context_blocks or top_score < 0.02:
            return {
                "answer": "This information was not found in the uploaded material.",
                "found": False,
                "source": None,
            }

        context_str = "\n\n".join(context_blocks)
        prompt = (
            f"Study Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer the question directly and concisely based ONLY on the context above. "
            f"If the answer is not stated in the context, respond strictly with: "
            f"'This information was not found in the uploaded material.' Do not guess."
        )

        t0 = time.monotonic()
        raw_answer = await _call_ollama(prompt, system=LEARN_SYSTEM_PROMPT, temperature=0.1)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        not_found = (
            "not found in the uploaded material" in raw_answer.lower()
            or "not found in the study material" in raw_answer.lower()
            or "not mentioned" in raw_answer.lower()
        )

        primary_source = f"Page {top_chunk.page}, Section: {top_chunk.heading}" if not not_found else None

        return {
            "answer": raw_answer,
            "found": not not_found,
            "source": primary_source,
            "latency_ms": latency_ms,
        }

    async def generate_quiz(self, doc_id: str, num_questions: int = 5) -> Dict[str, Any]:
        """
        Generates structured Multiple Choice Questions grounded strictly in the uploaded material.
        Validates JSON format with automatic retry.
        """
        doc = self.get_document(doc_id)
        if not doc:
            raise ValueError("Document not found.")

        # Take representative text samples from multiple sections
        sample_texts = []
        for s in doc.sections[:6]:
            sample_texts.append(f"Section {s.heading}:\n{s.full_text[:800]}")
        doc_context = "\n\n".join(sample_texts) if sample_texts else doc.full_text[:3500]

        prompt = (
            f"Study Material for {doc.title}:\n\n"
            f"{doc_context[:3000]}\n\n"
            f"Task: Generate a multiple-choice quiz with exactly {num_questions} educational questions strictly based on this material.\n"
            f"You MUST return ONLY valid JSON matching this exact schema:\n"
            f'{{\n'
            f'  "questions": [\n'
            f'    {{\n'
            f'      "question": "What is ...?",\n'
            f'      "options": ["Option A", "Option B", "Option C", "Option D"],\n'
            f'      "correct_index": 0,\n'
            f'      "explanation": "Clear 1-sentence reason why this option is correct.",\n'
            f'      "source_section": "Name of section"\n'
            f'    }}\n'
            f'  ]\n'
            f'}}\n'
            f"Ensure correct_index is an integer from 0 to 3. Return JSON only, no markdown formatting."
        )

        t0 = time.monotonic()
        for attempt in range(2):
            try:
                response = await _call_ollama(prompt, system="You are a strict JSON generator for educational quizzes.", temperature=0.2)
                # Clean code fences if present
                clean_json = re.sub(r'^```(?:json)?\s*', '', response.strip())
                clean_json = re.sub(r'\s*```$', '', clean_json)

                parsed = json.loads(clean_json)
                if "questions" in parsed and isinstance(parsed["questions"], list) and len(parsed["questions"]) > 0:
                    # Validate question elements
                    valid_qs = []
                    for q in parsed["questions"]:
                        if (
                            "question" in q
                            and "options" in q
                            and isinstance(q["options"], list)
                            and len(q["options"]) >= 2
                            and "correct_index" in q
                        ):
                            # Pad options to 4 if needed
                            while len(q["options"]) < 4:
                                q["options"].append("None of the above")
                            valid_qs.append({
                                "question": str(q["question"]),
                                "options": [str(opt) for opt in q["options"][:4]],
                                "correct_index": int(q["correct_index"]) % len(q["options"][:4]),
                                "explanation": str(q.get("explanation", "Grounded in study notes.")),
                                "source_section": str(q.get("source_section", doc.title)),
                            })

                    if valid_qs:
                        return {
                            "document_id": doc_id,
                            "title": doc.title,
                            "question_count": len(valid_qs),
                            "questions": valid_qs,
                            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                        }
            except Exception as json_err:
                log.warning("Quiz JSON attempt %d failed: %s", attempt + 1, json_err)
                prompt += "\n\nError: Output was not valid JSON. Please return valid raw JSON only."

        # Fallback question generation if LLM JSON was malformed
        fallback_qs = []
        for i, s in enumerate(doc.sections[:num_questions]):
            fallback_qs.append({
                "question": f"Which core concept is primarily discussed in '{s.heading}'?",
                "options": [
                    s.heading,
                    "General computer history",
                    "Unrelated networking protocol",
                    "None of the above",
                ],
                "correct_index": 0,
                "explanation": f"The section '{s.heading}' covers this topic in the uploaded material.",
                "source_section": s.heading,
            })

        return {
            "document_id": doc_id,
            "title": doc.title,
            "question_count": len(fallback_qs),
            "questions": fallback_qs,
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }


service = LearnService()
