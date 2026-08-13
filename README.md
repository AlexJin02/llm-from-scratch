# LLM from scratch

Public learning repository for W1-W4 of the eleven-week study plan.

Planned modules:

- `src/micrograd`
- `src/makemore`
- `src/gpt`
- `src/tokenizer`
- `src/optim`

The learning-critical implementations are intentionally absent. Environment check:

```bash
uv sync
uv run ruff check .
uv run mypy .
uv run pytest
```

Before publishing, scan for secrets, model checkpoints, datasets, and prohibited coursework.
