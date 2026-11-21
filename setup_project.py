from pathlib import Path

folders = [
    "data/raw",
    "data/interim",
    "data/processed",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/billing",
    "src/explain",
    "app",
    "models",
    "reports/figures",
    "tests",
]

for f in folders:
    Path(f).mkdir(parents=True, exist_ok=True)

    if not f.startswith("data/"):
        (Path(f) / ".gitkeep").touch()

for pkg in [
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/billing",
    "src/explain",
]:
    (Path(pkg) / "__init__.py").touch()

Path(".gitignore").write_text(
    ".venv/\n"
    "__pycache__/\n"
    ".ipynb_checkpoints/\n"
    "data/raw/\n"
    "data/interim/\n"
    "data/processed/\n"
    "models/*.joblib\n"
    ".env\n"
    "kaggle.json\n"
)

Path("README.md").write_text(
    "# unitwise\n\n"
    "Household Energy Intelligence System (work in progress).\n"
)

print("Project structure created.")