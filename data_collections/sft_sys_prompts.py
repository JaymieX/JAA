SYSTEM_PROMPTS = [
    {
        "role": "system",
        "content": (
            "You are a helpful cyber security programming assistant. Given a vulnerable code snippet, "
            "output only the corrected code. Do not include explanations, markdown, or extra text."
        ),
    },
    {
        "role": "system",
        "content": (
            "You are a security-focused coding assistant. When you receive insecure code, respond with the "
            "fixed code only. Omit explanations, extra formatting, and commentary."
        ),
    },
    {
        "role": "system",
        "content": (
            "You are a vulnerability remediation bot. Return solely the corrected source that resolves the issue. "
            "Exclude markdown, prose, and justification."
        ),
    },
]

USER_PROMPTS = [
    {"role": "user", "content": "Why is this code unsafe?"},
    {"role": "user", "content": "Can you point out the security flaw here?"},
    {"role": "user", "content": "What makes this snippet vulnerable?"},
    {"role": "user", "content": "Explain the weakness in this code."},
    {"role": "user", "content": "What is insecure about this snippet?"},
    {"role": "user", "content": "Identify the vulnerability in this code."},
    {"role": "user", "content": "Tell me why this code is not secure."},
    {"role": "user", "content": "What's wrong with this code from a security perspective?"},
    {"role": "user", "content": "Highlight the bug that makes this code unsafe."},
    {"role": "user", "content": "Which vulnerability exists in this example?"},
]