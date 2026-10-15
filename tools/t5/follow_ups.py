"""Follow-up user turns that keep a conversation going once its script is exhausted.

A scripted conversation can end before it reaches the target length when the model answers briefly. The client then
sends follow-ups from these pools, which ask for more substance rather than a restatement: depth, examples, edge
cases, alternatives. The order is fixed per conversation (`for_conversation`), so a rerun asks the same questions.
"""

from __future__ import annotations

import hashlib

ENGLISH = {
    "conversation": [
        "Can you go deeper on the part you think matters most? I'd like the reasoning, not just the conclusion.",
        "Walk me through a concrete example of this, step by step.",
        "What would someone who disagrees with you say, and how would you answer them?",
        "What are the edge cases or situations where this advice breaks down?",
        "If I had to explain this to a colleague tomorrow, what details would they ask me about?",
        "What would you do differently if the constraints were tighter, say half the time or budget?",
        "Which part of this is most often misunderstood, and why?",
        "Give me a checklist I could actually follow, with a short note on why each step is there.",
    ],
    "coding": [
        "Now write tests for this, including the edge cases you think are most likely to break.",
        "What are the failure modes of this code under load or with bad input? Show how you'd handle them.",
        "Refactor it for readability and explain each change.",
        "How would you change the design if this had to scale to a hundred times the data?",
        "Add proper error handling and logging, and show the full updated code.",
        "Walk through the code line by line and point out anything you would flag in review.",
        "Show an alternative implementation with a different approach and compare the trade-offs.",
        "Write the documentation a new team member would need to use and maintain this.",
    ],
    "tool_use": [
        "Before we finish, double-check the details with the tools and tell me if anything changed.",
        "What else should I check before I commit to this? Use the tools if they help.",
        "Summarize the options you looked at in a table, with the trade-offs for each.",
        "Plan the next steps for me in order, and note anything that depends on what we found.",
        "Is there a cheaper or faster alternative? Check if you can.",
        "What could go wrong with this plan, and what would you do about each risk?",
    ],
}

MULTILINGUAL = {
    "es": [
        "¿Puedes profundizar en la parte que consideras más importante? Me interesa el razonamiento.",
        "Dame un ejemplo concreto, paso a paso.",
        "¿Qué diría alguien que no está de acuerdo contigo, y cómo le responderías?",
        "¿En qué situaciones este consejo deja de funcionar?",
        "¿Qué harías distinto si tuvieras la mitad del tiempo o del presupuesto?",
        "Hazme una lista de pasos que pueda seguir, con una breve explicación de cada uno.",
    ],
    "zh": [
        "你能再深入讲讲你认为最重要的那部分吗？我想了解背后的推理。",
        "请举一个具体的例子，一步一步说明。",
        "持不同意见的人会怎么说？你会如何回应？",
        "在哪些情况下这个建议就不适用了？",
        "如果时间或预算只有一半，你会怎么调整？",
        "请给我一个可以照着做的清单，并简单说明每一步的原因。",
    ],
    "ja": [
        "一番重要だと思う部分をもっと詳しく説明してもらえますか？結論だけでなく理由も知りたいです。",
        "具体的な例を挙げて、順を追って説明してください。",
        "反対の立場の人は何と言うでしょうか。それにどう答えますか？",
        "このアドバイスが通用しないのはどんな場合ですか？",
        "時間や予算が半分だったら、何を変えますか？",
        "実際に使えるチェックリストを作ってください。各項目の理由も短く添えてください。",
    ],
    "ar": [
        "هل يمكنك التعمق أكثر في الجزء الذي تراه الأهم؟ أريد أن أفهم طريقة التفكير وليس النتيجة فقط.",
        "أعطني مثالاً عملياً خطوة بخطوة.",
        "ماذا سيقول شخص يختلف معك، وكيف سترد عليه؟",
        "في أي الحالات لا تنجح هذه النصيحة؟",
        "ماذا ستغير لو كان لديك نصف الوقت أو نصف الميزانية؟",
        "اكتب لي قائمة خطوات يمكنني اتباعها، مع سبب مختصر لكل خطوة.",
    ],
    "hi": [
        "क्या आप उस हिस्से को और विस्तार से समझा सकते हैं जो आपको सबसे ज़रूरी लगता है? मुझे तर्क जानना है।",
        "इसका एक ठोस उदाहरण कदम-दर-कदम दीजिए।",
        "जो आपसे असहमत है वह क्या कहेगा, और आप उसे क्या जवाब देंगे?",
        "किन हालात में यह सलाह काम नहीं करती?",
        "अगर समय या बजट आधा होता तो आप क्या बदलते?",
        "मुझे एक चेकलिस्ट दीजिए जिसे मैं सच में अपना सकूँ, हर कदम के कारण के साथ।",
    ],
    "mixed": [
        "Okay but can you go deeper on that, like the reasoning behind it? Thoda detail mein batao.",
        "Give me a concrete example, step by step, porfa.",
        "What would someone who disagrees say? Aur aap unko kya jawab doge?",
        "In which cases does this not work? 具体一点。",
        "If I only had half the time, qué cambiarías?",
        "Make me a checklist I can actually follow, with a short why for each step.",
    ],
}


def pool(domain: str, language: str) -> list[str]:
    """The follow-ups for a conversation's domain and language (BCP-47 base or "mixed")."""
    if domain == "multilingual":
        return MULTILINGUAL.get(language.split("-")[0].lower(), MULTILINGUAL["mixed"])
    return ENGLISH.get(domain, ENGLISH["conversation"])


def for_conversation(conversation_id: str, domain: str, language: str) -> list[str]:
    """The pool rotated by a hash of the conversation id, so conversations do not all ask the same first question."""
    items = pool(domain, language)
    start = int.from_bytes(hashlib.sha256(conversation_id.encode()).digest()[:4], "big") % len(items)
    return items[start:] + items[:start]
