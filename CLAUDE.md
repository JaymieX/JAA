# Claude Code Instructions

## File Path Requirements

**ALWAYS use relative paths when performing file operations (Read, Edit, Write, etc.)**

### Rule: Convert Absolute to Relative Paths

Even when the user provides an absolute path like:
- `'d:/PY/JAA/data_collections/taco_sft_data_gen.py'`
- `/home/user/JAA/src/main.py`
- `C:\Users\name\JAA\config.toml`

**You MUST convert it to a relative path from the JAA root directory before using it:**
- `data_collections/taco_sft_data_gen.py`
- `src/main.py`
- `config.toml`

### Examples

❌ **WRONG:**
```
Read file_path="D:/PY/JAA/data_collections/taco_sft_data_gen.py"
Edit file_path="/home/user/JAA/config.toml"
```

✅ **CORRECT:**
```
Read file_path="data_collections/taco_sft_data_gen.py"
Edit file_path="config.toml"
```

### Implementation

When the user provides an absolute path:
1. Identify the JAA root directory in the path
2. Extract everything after `/JAA/` or `\JAA\` as the relative path
3. Use only the relative path in file operations
4. Never use the full absolute path in Read/Edit/Write tools

This ensures compatibility across different environments and working directories while treating JAA as the project root.