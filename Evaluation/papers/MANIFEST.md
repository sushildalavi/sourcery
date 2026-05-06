# ScholarRAG Corpus (15 diverse scholarly papers)

This directory contains the evaluation corpus: a deliberately diverse set of
15 landmark papers spanning 1997–2023 across computer vision, generative
models, reinforcement learning, large language models, computational biology,
and information retrieval.

## Corpus

| # | File | Paper | Field | Venue / Year | Official link |
|---|---|---|---|---|---|
| 01 | `01_ResNet.pdf` | Deep Residual Learning for Image Recognition | Computer vision | CVPR 2016 | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| 02 | `02_GAN.pdf` | Generative Adversarial Networks | Generative models | NeurIPS 2014 | [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) |
| 03 | `03_Word2Vec.pdf` | Efficient Estimation of Word Representations in Vector Space | Representation learning | ICLR 2013 | [arXiv:1301.3781](https://arxiv.org/abs/1301.3781) |
| 04 | `04_LLaMA2.pdf` | LLaMA 2: Open Foundation and Fine-Tuned Chat Models | Open LLMs | Meta, 2023 | [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) |
| 05 | `05_Chinchilla.pdf` | Training Compute-Optimal Large Language Models | Scaling laws | NeurIPS 2022 | [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) |
| 06 | `06_ConstitutionalAI.pdf` | Constitutional AI: Harmlessness from AI Feedback | Alignment | Anthropic, 2022 | [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) |
| 07 | `07_AlphaGo.pdf` | Mastering the Game of Go with Deep Neural Networks and Tree Search | Reinforcement learning | *Nature* 529 (2016) | [doi:10.1038/nature16961](https://doi.org/10.1038/nature16961) |
| 08 | `08_CLIP.pdf` | Learning Transferable Visual Models From Natural Language Supervision | Multimodal | ICML 2021 | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| 09 | `09_AlphaFold.pdf` | Highly Accurate Protein Structure Prediction with AlphaFold | Computational biology | *Nature* 596 (2021) OA | [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2) |
| 10 | `10_StableDiffusion.pdf` | High-Resolution Image Synthesis with Latent Diffusion Models | Generative models | CVPR 2022 | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) |
| 11 | `11_LSTM.pdf` | Long Short-Term Memory | Foundational ML | *Neural Computation* 9(8), 1997 | [doi:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735) |
| 12 | `12_DQN.pdf` | Human-level Control Through Deep Reinforcement Learning | Reinforcement learning | *Nature* 518 (2015) | [doi:10.1038/nature14236](https://doi.org/10.1038/nature14236) |
| 13 | `13_VAE.pdf` | Auto-Encoding Variational Bayes | Generative models | ICLR 2014 | [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) |
| 14 | `14_SwinTransformer.pdf` | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | Computer vision | ICCV 2021 | [arXiv:2103.14030](https://arxiv.org/abs/2103.14030) |
| 15 | `15_PageRank.pdf` | The Anatomy of a Large-Scale Hypertextual Web Search Engine | Information retrieval | WWW 1998 | [Stanford InfoLab](http://infolab.stanford.edu/pub/papers/google.pdf) |

## Regenerating the corpus

From this directory:

```bash
bash download_corpus.sh
```

The script hits multiple candidate URLs per paper (arXiv, publisher OA copy,
author-hosted) and verifies that each download is a real PDF via the `%PDF`
magic bytes + `file` content sniff, rejecting HTML error pages that some
publishers return for paywalled URLs.

## Ingesting into the live system

After the PDFs are present:

1. Upload each PDF via the Documents UI, OR
2. Run `python -m backend.scripts.ingest_corpus` (purges any stale docs + indexes each).

The calibrated MSA grounding weights are stored in the `confidence_calibration`
table under `label='unified'` and loaded by `_load_latest_calibration_weights()`
when `CONFIDENCE_USE_FITTED_WEIGHTS=true` is set.
