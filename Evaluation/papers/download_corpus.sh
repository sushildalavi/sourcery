#!/usr/bin/env bash
# Downloads the 15-paper corpus (diverse scholarly papers across 7 subfields).
# Primary sources prefer open PDFs (arXiv PDF, publisher OA copy, author-hosted).
# Usage:  bash download_corpus.sh
# Output: 01_ResNet.pdf ... 15_PageRank.pdf in the current directory.

set -u
PAPERS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PAPERS_DIR"

UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"

ok=0
fail=0
failures=()

fetch() {
  local outfile="$1"; shift
  local name="$1"; shift
  # Remaining args are candidate URLs (in order of preference)
  for url in "$@"; do
    echo ""
    echo "[$outfile] $name"
    echo "  trying: $url"
    if curl -fsSL --max-time 45 -A "$UA" -o "$outfile.tmp" "$url"; then
      # Reject HTML error pages masquerading as PDFs by reading the magic bytes
      # and the `file -b` content sniff (which outputs just the type, without
      # the filename — earlier check grepped the filename and matched "pdf").
      local magic
      magic=$(head -c 4 "$outfile.tmp" 2>/dev/null)
      local sniff
      sniff=$(file -b "$outfile.tmp" 2>/dev/null)
      if [[ "$magic" == "%PDF" && "$sniff" =~ ^PDF ]]; then
        mv "$outfile.tmp" "$outfile"
        local sz
        sz=$(wc -c <"$outfile" | tr -d ' ')
        echo "  OK  ($sz bytes)"
        ok=$((ok + 1))
        return 0
      else
        echo "  got non-PDF ($sniff), discarding"
        rm -f "$outfile.tmp"
      fi
    else
      echo "  curl failed"
      rm -f "$outfile.tmp" 2>/dev/null || true
    fi
  done
  fail=$((fail + 1))
  failures+=("$outfile | $name")
  return 1
}

fetch "01_ResNet.pdf" "Deep Residual Learning for Image Recognition (He et al., 2016)" \
  "https://arxiv.org/pdf/1512.03385.pdf" \
  "https://arxiv.org/pdf/1512.03385v1.pdf"

fetch "02_GAN.pdf" "Generative Adversarial Networks (Goodfellow et al., 2014)" \
  "https://arxiv.org/pdf/1406.2661.pdf" \
  "https://arxiv.org/pdf/1406.2661v1.pdf"

fetch "03_Word2Vec.pdf" "Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)" \
  "https://arxiv.org/pdf/1301.3781.pdf" \
  "https://arxiv.org/pdf/1301.3781v3.pdf"

fetch "04_LLaMA2.pdf" "LLaMA 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al., 2023)" \
  "https://arxiv.org/pdf/2307.09288.pdf" \
  "https://arxiv.org/pdf/2307.09288v2.pdf"

fetch "05_Chinchilla.pdf" "Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)" \
  "https://arxiv.org/pdf/2203.15556.pdf" \
  "https://arxiv.org/pdf/2203.15556v1.pdf"

fetch "06_ConstitutionalAI.pdf" "Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)" \
  "https://arxiv.org/pdf/2212.08073.pdf" \
  "https://arxiv.org/pdf/2212.08073v1.pdf"

fetch "07_AlphaGo.pdf" "Mastering the Game of Go with Deep Neural Networks and Tree Search (Silver et al., 2016, Nature)" \
  "https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf" \
  "https://www.nature.com/articles/nature16961.pdf"

fetch "08_CLIP.pdf" "Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)" \
  "https://arxiv.org/pdf/2103.00020.pdf" \
  "https://arxiv.org/pdf/2103.00020v1.pdf"

fetch "09_AlphaFold.pdf" "Highly Accurate Protein Structure Prediction with AlphaFold (Jumper et al., 2021, Nature OA)" \
  "https://www.nature.com/articles/s41586-021-03819-2.pdf" \
  "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf"

fetch "10_StableDiffusion.pdf" "High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)" \
  "https://arxiv.org/pdf/2112.10752.pdf" \
  "https://arxiv.org/pdf/2112.10752v2.pdf"

fetch "11_LSTM.pdf" "Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)" \
  "https://www.bioinf.jku.at/publications/older/2604.pdf" \
  "https://direct.mit.edu/neco/article-pdf/9/8/1735/813796/neco.1997.9.8.1735.pdf"

fetch "12_DQN.pdf" "Human-level Control Through Deep Reinforcement Learning (Mnih et al., 2015, Nature)" \
  "https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf" \
  "https://www.nature.com/articles/nature14236.pdf" \
  "https://arxiv.org/pdf/1312.5602.pdf"

fetch "13_VAE.pdf" "Auto-Encoding Variational Bayes (Kingma & Welling, 2014)" \
  "https://arxiv.org/pdf/1312.6114.pdf" \
  "https://arxiv.org/pdf/1312.6114v11.pdf"

fetch "14_SwinTransformer.pdf" "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Liu et al., 2021)" \
  "https://arxiv.org/pdf/2103.14030.pdf" \
  "https://arxiv.org/pdf/2103.14030v2.pdf"

fetch "15_PageRank.pdf" "The Anatomy of a Large-Scale Hypertextual Web Search Engine (Brin & Page, 1998)" \
  "http://ilpubs.stanford.edu:8090/361/1/1998-8.pdf" \
  "https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf"

echo ""
echo "===================================="
echo "  OK:   $ok"
echo "  FAIL: $fail"
if (( fail > 0 )); then
  echo "  Missing:"
  for f in "${failures[@]}"; do echo "    - $f"; done
  exit 1
fi
