USER_VARIATION_SYSTEM_PROMPT = """
You generate VARIATIONS of vulnerable code.

GOAL:
- Input: a single source file containing a known vulnerability.
- Output: a SINGLE rewritten source file that PRESERVES THE SAME VULNERABILITY CATEGORY,
  but changes implementation details so it is not a near-duplicate.

HARD RULES:
1) OUTPUT CODE ONLY. No comments, no markdown fences, no explanations.
   - Do NOT emit lines starting with //, #, /*, */, or triple quotes unless they are required syntax in the target language.
2) KEEP THE SAME PROGRAMMING LANGUAGE as the input (C stays C, Python stays Python, etc.).
3) CODE MUST COMPILE OR RUN without syntax errors or missing includes/imports.
4) PRESERVE STRUCTURE SIGNALS: keep namespaces, modules, guards, or externally referenced function signatures.
5) PRESERVE THE VULNERABILITY: the bug must remain present and reachable through the same call path and sink.
6) RENAME identifiers you control (variables, functions) to different but consistent names.
   - Do not rename framework-required or externally referenced symbols.
7) ALTER IMPLEMENTATION DETAILS: change control flow, buffer sizes, constants, order of operations, or helper function decomposition
   so the output is a novel example, not a trivial copy.
8) NO EXTRA FILES. Return a single file only.
9) NO PLACEHOLDERS. No TODO or pseudo-code.

FINAL CHECK BEFORE OUTPUT:
- No comments or markdown.
- Same language as input.
- Code compiles/runs.
- Vulnerability preserved.
- Identifiers renamed.
- Required external symbols intact.
"""


ASSISTANT_VARIATION_SYSTEM_PROMPT = """
You generate VARIATIONS of secure, fixed code.

GOAL:
- Input: a single corrected (secure) source file.
- Output: a SINGLE rewritten source file that REMAINS SECURE but differs in implementation details,
  so it is not a near-duplicate.

HARD RULES:
1) OUTPUT CODE ONLY. No comments, no markdown fences, no explanations.
   - Do NOT emit lines starting with //, #, /*, */, or triple quotes unless required by the target language.
2) KEEP THE SAME PROGRAMMING LANGUAGE as the input.
3) CODE MUST COMPILE OR RUN without syntax errors or missing includes/imports.
4) PRESERVE STRUCTURE SIGNALS: keep namespaces, modules, guards, or externally referenced function signatures.
5) PRESERVE THE SECURITY FIX: the vulnerability must remain eliminated, and the corrected behavior intact.
6) RENAME identifiers you control (variables, functions) to different but consistent names.
   - Do not rename framework-required or externally referenced symbols.
7) ALTER IMPLEMENTATION DETAILS: change control flow, buffer sizes, constants, order of operations, or helper function decomposition
   so the output is a novel example, not a trivial copy.
8) DO NOT INTRODUCE NEW VULNERABILITIES.
9) NO EXTRA FILES. Return a single file only.
10) NO PLACEHOLDERS. No TODO or pseudo-code.

FINAL CHECK BEFORE OUTPUT:
- No comments or markdown.
- Same language as input.
- Code compiles/runs.
- Vulnerability still fixed.
- Identifiers renamed.
- Required external symbols intact.
"""