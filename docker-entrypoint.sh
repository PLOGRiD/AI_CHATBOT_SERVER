#!/bin/sh
set -e

if [ ! -d "rag/chroma_db" ] || [ -z "$(ls -A rag/chroma_db 2>/dev/null)" ]; then
    echo "rag/chroma_db가 없어 벡터 스토어를 생성합니다..."
    python -m rag.ingest
fi

exec "$@"
