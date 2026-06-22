# Local Retrieval Mode

## Default local options

- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`

## Guidance

- Prefer small, local embeddings for Mac-friendly demos.
- Keep any larger model downloads optional and user-initiated.
- Use the repo evaluation artifacts to document retrieval quality rather than inventing new numbers.

## Verified retrieval metrics

- Recall@5: 0.9917
- Recall@10: 1.0
- MRR: 0.9812
- nDCG@10: 0.9857

