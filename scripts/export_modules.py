from __future__ import annotations

from pathlib import Path
import shutil

PROJECT_ROOT = Path(".").resolve()
OUTPUT_DIR = PROJECT_ROOT / "project_export"

# --------------------------------------------------------
# Modules to export
# --------------------------------------------------------

MODULES = {
    "core": [
        "app/core",
    ],
    "preprocessing": [
        "app/preprocessing",
    ],
    "embeddings": [
        "app/embeddings",
    ],
    "vectorstore": [
        "app/vectorstore",
    ],
    "retrieval": [
        "app/retrieval",
    ],
    "reranker": [
        "app/reranker",
    ],
    "context": [
        "app/context",
    ],
    "prompt": [
        "app/prompt",
    ],
    "llm": [
        "app/llm",
    ],
    "pipeline": [
        "app/pipeline",
    ],
    "evaluation": [
        "app/evaluation",
    ],
    "self_healing": [
        "app/self_healing",
    ],
    "configs": [
        "configs",
    ],
    "scripts": [
        "scripts",
    ],
    "tests": [
        "tests",
    ],
    "root": [
        "main.py",
        "requirements.txt",
        "README.md",
    ],
}

# --------------------------------------------------------
# Ignore
# --------------------------------------------------------

IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    ".idea",
    ".vscode",
    "data",
    "notebooks",
    "project_export",
}

IGNORE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".npy",
    ".npz",
    ".pkl",
    ".sqlite",
    ".db",
    ".log",
    ".zip",
    ".tar",
    ".gz",
}

IGNORE_FILES = {
    "project_codes.md",
    "project_source.md",
}


# --------------------------------------------------------


def should_skip(path: Path) -> bool:

    if path.name in IGNORE_FILES:
        return True

    if path.suffix.lower() in IGNORE_SUFFIXES:
        return True

    for part in path.parts:
        if part in IGNORE_DIRS:
            return True

    return False


# --------------------------------------------------------


def iter_source_files(base: Path):

    if base.is_file():
        if not should_skip(base):
            yield base
        return

    for file in sorted(base.rglob("*")):

        if not file.is_file():
            continue

        if should_skip(file):
            continue

        yield file


# --------------------------------------------------------


def export_module(module_name: str, paths: list[str]):

    outfile = OUTPUT_DIR / f"{module_name}.md"

    with outfile.open("w", encoding="utf-8") as out:

        out.write(f"# {module_name}\n\n")

        for item in paths:

            target = PROJECT_ROOT / item

            if not target.exists():
                continue

            for file in iter_source_files(target):

                rel = file.relative_to(PROJECT_ROOT)

                out.write("\n---\n\n")
                out.write(f"## FILE: {rel}\n\n")

                out.write(f"Path: {rel}\n")
                out.write(f"Size: {file.stat().st_size} bytes\n")
                out.write(
                    f"Modified: {file.stat().st_mtime}\n\n"
                )

                language = ""

                if file.suffix == ".py":
                    language = "python"

                elif file.suffix in {".yaml", ".yml"}:
                    language = "yaml"

                elif file.suffix == ".md":
                    language = "markdown"

                elif file.suffix == ".txt":
                    language = "text"

                out.write(f"```{language}\n")

                out.write(
                    file.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                )

                out.write("\n```\n")


# --------------------------------------------------------


def main():

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True)

    for module_name, paths in MODULES.items():
        print(f"Exporting {module_name}...")
        export_module(module_name, paths)

    print()
    print("Done.")
    print(f"Output -> {OUTPUT_DIR}")


# --------------------------------------------------------

if __name__ == "__main__":
    main()