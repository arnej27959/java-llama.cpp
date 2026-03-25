#!/bin/sh

MODEL_URL=https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q2_K.gguf
MODEL_NAME=codellama-7b.Q2_K.gguf

RERANKING_MODEL_URL=https://huggingface.co/gpustack/jina-reranker-v1-tiny-en-GGUF/resolve/main/jina-reranker-v1-tiny-en-Q4_0.gguf
RERANKING_MODEL_NAME=jina-reranker-v1-tiny-en-Q4_0.gguf

if ! [ -d models ]; then cd ..; fi
echo "Fetching models to `pwd`/models directory..."
set -x

curl -L ${MODEL_URL} -o models/${MODEL_NAME}
curl -L ${RERANKING_MODEL_URL} -o models/${RERANKING_MODEL_NAME}
