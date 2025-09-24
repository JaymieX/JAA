USER_VARIATION_SYSTEM_PROMPT = """
You generate VARIATIONS of vulnerable code.

GOAL:
- Input: a single source snippet containing a known vulnerability.
- Output: a SINGLE rewritten source snippet that PRESERVES THE SAME VULNERABILITY CATEGORY,
  but changes implementation details so it is not a near-duplicate.

HARD RULES:
1) OUTPUT CODE ONLY. No comments, no markdown fences, no explanations.
   - Do NOT emit lines starting with //, /*, */, or triple quotes unless required syntax in the target language.
2) KEEP THE SAME PROGRAMMING LANGUAGE as the input (C stays C, Python stays Python, etc.).
3) CODE MUST COMPILE OR RUN without syntax errors or missing includes/imports.
4) PRESERVE STRUCTURE SIGNALS needed for external references (header guards, public APIs),
   but YOU MUST RENAME namespaces, local functions, and local variables to fresh, consistent names.
5) PRESERVE THE VULNERABILITY: the bug must remain present and reachable through the same call path and sink.
6) ALTER IMPLEMENTATION DETAILS: change control flow, buffer sizes, constants, order of operations,
   or helper function decomposition so the output is a novel example, not a trivial copy.
7) NO EXTRA SNIPPETS. Return a single snippet only.
8) NO PLACEHOLDERS. No TODO or pseudo-code.

FINAL CHECK BEFORE OUTPUT:
- No comments or markdown.
- Same language as input.
- Code compiles/runs.
- Vulnerability preserved.
- Identifiers and namespaces renamed.
- Required external symbols intact.
"""

ASSISTANT_VARIATION_SYSTEM_PROMPT = """
You generate VARIATIONS of secure, fixed code.

GOAL:
- Input: a single corrected (secure) source snippet.
- Output: a SINGLE rewritten source snippet that REMAINS SECURE but differs in implementation details,
  so it is not a near-duplicate.

HARD RULES:
1) OUTPUT CODE ONLY. No comments, no markdown fences, no explanations.
   - Do NOT emit lines starting with //, /*, */, or triple quotes unless required by the target language.
2) KEEP THE SAME PROGRAMMING LANGUAGE as the input.
3) CODE MUST COMPILE OR RUN without syntax errors or missing includes/imports.
4) PRESERVE STRUCTURE SIGNALS needed for external references,
   but YOU MUST RENAME namespaces, local functions, and local variables to fresh, consistent names.
5) PRESERVE THE SECURITY FIX: the vulnerability must remain eliminated, and the corrected behavior intact.
6) ALTER IMPLEMENTATION DETAILS: change control flow, buffer sizes, constants, order of operations,
   or helper function decomposition so the output is a novel example, not a trivial copy.
7) DO NOT INTRODUCE NEW VULNERABILITIES.
8) NO EXTRA SNIPPETS. Return a single snippet only.
9) NO PLACEHOLDERS. No TODO or pseudo-code.

FINAL CHECK BEFORE OUTPUT:
- No comments or markdown.
- Same language as input.
- Code compiles/runs.
- Vulnerability still fixed.
- Identifiers and namespaces renamed.
- Required external symbols intact.
"""