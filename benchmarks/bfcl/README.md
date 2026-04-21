# BFCL Samples

This folder contains generated local tasks adapted from the Berkeley Function
Calling Leaderboard (BFCL), specifically `BFCL_v3_simple`.

Source:

- Dataset: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
- URL: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- License: Apache-2.0

The tasks convert BFCL user requests and available function schemas into the
repo's `structured_extract` protocol:

```json
{
  "tool_name": "function.name",
  "arguments": {
    "arg": "value"
  }
}
```

Regenerate with:

```bash
python3 scripts/import_bfcl_samples.py
```

If local Python TLS certificates block direct downloads, fetch the two raw BFCL
JSONL files separately and pass:

```bash
python3 scripts/import_bfcl_samples.py \
  --data-file /path/to/BFCL_v3_simple.json \
  --answers-file /path/to/BFCL_v3_simple_answers.json
```
