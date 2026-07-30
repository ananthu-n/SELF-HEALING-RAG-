#!/usr/bin/env bash

###############################################################################
# Self-Healing RAG Project Snapshot
#
# Generates:
#   project_snapshot.md
#
# This file contains:
#   - Project structure
#   - Selected configuration files
#   - Source code for important modules
#
###############################################################################

OUTPUT="project_snapshot.md"

rm -f "$OUTPUT"

###############################################################################
# Helper
###############################################################################

section() {
    echo "" >> "$OUTPUT"
    echo "# $1" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
}

file() {
    local FILE="$1"

    if [ -f "$FILE" ]; then
        echo "" >> "$OUTPUT"
        echo "## $FILE" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
        echo '```python' >> "$OUTPUT"
        cat "$FILE" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
        echo '```' >> "$OUTPUT"
    else
        echo "" >> "$OUTPUT"
        echo "## $FILE (NOT FOUND)" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
    fi
}

###############################################################################
section "Project Information"

date >> "$OUTPUT"

echo "" >> "$OUTPUT"

pwd >> "$OUTPUT"

###############################################################################
section "Project Structure"

tree -L 3 app >> "$OUTPUT" 2>/dev/null || find app -maxdepth 3 >> "$OUTPUT"

###############################################################################
section "Retrieval Structure"

tree app/retrieval >> "$OUTPUT" 2>/dev/null || find app/retrieval >> "$OUTPUT"

###############################################################################
section "Self Healing Structure"

tree app/self_healing >> "$OUTPUT" 2>/dev/null || find app/self_healing >> "$OUTPUT"

###############################################################################
section "LLM Structure"

tree app/llm >> "$OUTPUT" 2>/dev/null || find app/llm >> "$OUTPUT"

###############################################################################
section "Configs"

file configs/config.yaml
file app/core/config.py

###############################################################################
section "Retrieval"

file app/retrieval/models.py
file app/retrieval/service.py
file app/retrieval/dense.py
file app/retrieval/validation.py
file app/retrieval/base.py

###############################################################################
section "Self Healing"

file app/self_healing/state.py
file app/self_healing/controller.py
file app/self_healing/retry.py
file app/self_healing/query_optimizer.py

###############################################################################
section "LLM"

file app/llm/client.py
file app/llm/generator.py
file app/llm/service.py

###############################################################################
section "Prompt"

file app/prompt/builder.py
file app/prompt/models.py
file app/prompt/validator.py

###############################################################################
section "Evaluation"

file app/evaluation/grounding.py
file app/evaluation/decision.py

###############################################################################
section "Vector Store"

file app/vectorstore/qdrant.py

###############################################################################
section "Embedding"

file app/embeddings/model.py
file app/embeddings/storage.py

###############################################################################
section "Package List"

pip freeze >> "$OUTPUT"

###############################################################################

echo ""
echo "======================================="
echo " Snapshot generated:"
echo "   $OUTPUT"
echo "======================================="