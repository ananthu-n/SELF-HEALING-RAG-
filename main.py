from __future__ import annotations

import os
import sys
import uvicorn
from app.core.logger import logger
from app.self_healing.controller import SelfHealingController


def run_server(host: str = None, port: int = None, reload: bool = True):
    """Starts the FastAPI Web Server with Uvicorn Auto-Reloading."""
    server_host = host or os.getenv("HOST", "0.0.0.0")
    server_port = int(port or os.getenv("PORT", "8000"))
    
    logger.info(f"Starting Self-Healing RAG Server with Auto-Reload on http://{server_host}:{server_port}...")
    
    uvicorn.run(
        "app.api.server:app",
        host=server_host,
        port=server_port,
        reload=reload,
        reload_dirs=["app", "static"],
    )


def run_cli():
    """Runs the Interactive CLI Session."""
    print("=" * 80)
    print("        SELF-HEALING RESEARCH ASSISTANT (Interactive CLI)")
    print("=" * 80)
    print("Type your questions below. Type 'exit' or 'quit' to end the session.\n")

    controller = SelfHealingController()

    while True:
        try:
            query = input("\n[User Query] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session. Goodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Exiting session. Goodbye!")
            break

        print("\nProcessing request through Self-Healing RAG pipeline...")
        state = controller.answer(query)

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        if state.last_generation:
            print(state.last_generation.response.answer)
        else:
            print("No answer produced.")

        print("\n" + "=" * 80)
        print("CITATIONS & SOURCES")
        print("=" * 80)
        if state.last_generation and state.last_generation.citations:
            for idx, citation in enumerate(state.last_generation.citations, start=1):
                print(f"{idx}. Paper ID: {citation.paper_id} | Page: {citation.page_number} | Chunk: {citation.chunk_id}")
        else:
            print("No citations recorded.")

        print("\n" + "=" * 80)
        print("HEALING HISTORY")
        print("=" * 80)
        print(f"Total Retries Executed: {state.retry_count}")
        for idx, plan in enumerate(state.healing_history, start=1):
            print(f"Attempt {idx}: Query='{plan.query}' | Rewrite={plan.rewrite_query} | Reason='{plan.reason}'")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        port_env = int(os.getenv("PORT", 8000))
        is_dev = os.getenv("ENV", "production").lower() == "development"
        run_server(host="0.0.0.0", port=port_env, reload=is_dev)