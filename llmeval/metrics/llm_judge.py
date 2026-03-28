"""
LLM-as-judge — uses GPT-4o-mini or Claude to score predictions on a 1-5 rubric.
Opt-in only (llm_judge: true in config) since it costs extra API calls.
"""

from __future__ import annotations

import os
import re

JUDGE_PROMPT = """You are an expert evaluator. Score the following model answer against the reference answer.

Question: {question}
Reference answer: {reference}
Model answer: {prediction}

Score the model answer on a scale of 1-5:
5 - Perfect: correct, complete, concise
4 - Good: mostly correct with minor gaps
3 - Partial: some correct elements but missing key info
2 - Poor: mostly incorrect or very incomplete
1 - Wrong: factually incorrect or irrelevant

Respond with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""


async def compute_llm_judge(
    question: str,
    reference: str,
    prediction: str,
    judge_model: str = "gpt-4o-mini",
) -> dict[str, float]:
    """Returns llm_judge score normalised to 0.0–1.0 (raw 1-5 divided by 5)."""
    if not prediction.strip():
        return {"llm_judge": 0.0, "llm_judge_raw": 0.0}

    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference, prediction=prediction
    )

    try:
        if "claude" in judge_model:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            msg = await client.messages.create(
                model=judge_model, max_tokens=128, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
        else:
            import openai
            client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            resp = await client.chat.completions.create(
                model=judge_model, max_tokens=128, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""

        import json
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            raw = float(data.get("score", 3))
            return {"llm_judge": round(raw / 5.0, 4), "llm_judge_raw": raw}

    except Exception as e:
        return {"llm_judge": 0.0, "llm_judge_raw": 0.0, "llm_judge_error": str(e)}

    return {"llm_judge": 0.0, "llm_judge_raw": 0.0}
