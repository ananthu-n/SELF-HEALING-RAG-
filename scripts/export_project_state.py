from pathlib import Path
import os
from datetime import datetime

ROOT = Path.cwd()
OUTPUT_DIR = ROOT / "project_state"

OUTPUT_DIR.mkdir(exist_ok=True)

SOURCE_FILE = OUTPUT_DIR / "PROJECT_SOURCE.txt"
TREE_FILE = OUTPUT_DIR / "PROJECT_TREE.txt"
INFO_FILE = OUTPUT_DIR / "PROJECT_INFO.txt"

# ------------------------------------------
# Ignore folders
# ------------------------------------------

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "logs",
    "data",
    "models",
    "checkpoints",
    "wandb",
    ".ipynb_checkpoints"
}

# ------------------------------------------
# Source extensions
# ------------------------------------------

SOURCE_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".txt",
    ".md",
    ".ini",
    ".cfg",
    ".sh",
    ".sql"
}


def ignored(path: Path):
    return any(part in IGNORE_DIRS for part in path.parts)


# ============================================================
# TREE
# ============================================================

tree_lines = []


def build_tree(path: Path, prefix=""):
    items = sorted(
        [x for x in path.iterdir() if not ignored(x)],
        key=lambda x: (x.is_file(), x.name.lower())
    )

    for i, item in enumerate(items):

        connector = "└── " if i == len(items)-1 else "├── "

        tree_lines.append(prefix + connector + item.name)

        if item.is_dir():
            extension = "    " if i == len(items)-1 else "│   "
            build_tree(item, prefix + extension)


tree_lines.append(ROOT.name)
build_tree(ROOT)

TREE_FILE.write_text("\n".join(tree_lines), encoding="utf-8")


# ============================================================
# SOURCE EXPORT
# ============================================================

total_files = 0
total_lines = 0

with SOURCE_FILE.open("w", encoding="utf-8") as out:

    out.write("=" * 120 + "\n")
    out.write("SELF-HEALING RAG SOURCE SNAPSHOT\n")
    out.write("=" * 120 + "\n\n")

    for file in sorted(ROOT.rglob("*")):

        if ignored(file):
            continue

        if not file.is_file():
            continue

        if file.suffix.lower() not in SOURCE_EXTENSIONS:
            continue

        relative = file.relative_to(ROOT)

        out.write("=" * 120 + "\n")
        out.write(f"FILE : {relative}\n")
        out.write("=" * 120 + "\n\n")

        try:
            content = file.read_text(encoding="utf-8")

            out.write(content)

            if not content.endswith("\n"):
                out.write("\n")

            total_files += 1
            total_lines += len(content.splitlines())

        except UnicodeDecodeError:
            out.write("Binary file skipped.\n")

        except Exception as e:
            out.write(str(e) + "\n")

        out.write("\n\n")


# ============================================================
# INFO
# ============================================================

info = []

info.append("PROJECT INFORMATION")
info.append("=" * 80)
info.append("")
info.append(f"Project Name : {ROOT.name}")
info.append(f"Generated    : {datetime.now()}")
info.append(f"Root         : {ROOT}")
info.append("")
info.append(f"Total Source Files : {total_files}")
info.append(f"Total Lines        : {total_lines}")
info.append("")

py = 0
yaml = 0
json = 0
md = 0

for f in ROOT.rglob("*"):

    if ignored(f):
        continue

    if not f.is_file():
        continue

    s = f.suffix.lower()

    if s == ".py":
        py += 1
    elif s in [".yaml", ".yml"]:
        yaml += 1
    elif s == ".json":
        json += 1
    elif s == ".md":
        md += 1

info.append("FILE TYPES")
info.append("-" * 40)
info.append(f"Python      : {py}")
info.append(f"YAML        : {yaml}")
info.append(f"JSON        : {json}")
info.append(f"Markdown    : {md}")

INFO_FILE.write_text("\n".join(info), encoding="utf-8")

print("\nExport Complete")
print(f"Folder : {OUTPUT_DIR}")
print(f"Tree   : {TREE_FILE.name}")
print(f"Source : {SOURCE_FILE.name}")
print(f"Info   : {INFO_FILE.name}")