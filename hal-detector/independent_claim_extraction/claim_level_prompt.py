CLAIM_EXTRACTION_PROMPTS = {
    "en": """Please breakdown the sentence into independent claims.

Example:
Sentence: \"He was born in London and raised by his mother and father until 11 years old.\"
Claims:
- He was born in London.
- He was raised by his mother and father.
- He was raised by his mother and father until 11 years old.

Sentence: \"{sent}\"
Claims:"""
}

CLAIM_EXTRACTION_PROMPTS_2 = {
    "en": """Please breakdown the sentence into independent claims.
Sentence: \"{sent}\"
Claims:"""
}

MATCHING_PROMPTS = {
    "en": (
        "Given the fact, identify the corresponding words "
        "in the original sentence that help derive this fact. "
        "Please list all words that are related to the fact, "
        "in the order they appear in the original sentence, "
        "each word separated by comma.\nFact: {claim}\n"
        "Sentence: {sent}\nWords from sentence that helps to "
        "derive the fact, separated by comma: "
    )
}

OPENAI_FACT_CHECK_PROMPTS = {
    "en": (
        """Question: {input}

Determine if all provided information in the following claim is true according to the most recent sources of information.

Claim: {claim}
"""
    ),
    "zh": (
        """Question: {input}

请根据最新的信息来源，确定以下声明（Claim）中提供的所有信息是否属实。

Claim: {claim}
"""
    ),
    "ru": (
        """Question: {input}\n
    Определи, соответствует ли вся предоставленная информация в следующем утверждении действительности согласно самым последним источникам информации.
    Если хотя бы часть утверждения неверна, склоняйся к выводу, что информация ложная.
    Think in English.
    Think step by step on how to summarize the claim within the provided <sketchpad>.
    Then, return a <summary> based on the <sketchpad>.

    Claim: {claim}
"""
    ),
    "ar": (
        """السؤال: {input}

حدد ما إذا كانت جميع المعلومات المقدمة في الادعاء التالي صحيحة وفقًا لأحدث مصادر المعلومات.

الادعاء: {claim}
"""
    ),
}

OPENAI_FACT_CHECK_SUMMARIZE_PROMPT = {
    "en": (
        """Question: {input}

Claim: {claim}

Is the following claim true?

Reply: {reply}

Summarize this reply into one word, whether the claim is true: "True", "False" or "Not known".
"""
    ),
    "zh": (
        """Question: {input}

Claim: {claim}

以下的表述是否正确？

Reply: {reply}

请用一个词回答该表述(Reply)是否正确："True"，"False"或"Not known"。
"""
    ),
    "ru": (
        """Question: {input}

Claim: {claim}

Is the following claim true?

Reply: {reply}

Summarize this reply into one word, whether the claim is true: "True", "False" or "Not known".
"""
    ),
    "ar": (
        """
السؤال: {input}

الادعاء: {claim}

هل الادعاء التالي صحيح؟

الاجابة: {reply}

قم بتلخيص هذه الجملة في كلمة واحدة، سواء كان الادعاء صحيح: "صحيح" أو "خطأ" أو "غير معروف".
"""
    ),
}
