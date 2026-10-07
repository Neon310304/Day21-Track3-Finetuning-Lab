import json
import pathlib
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
PREFIX = "lab21_2A202602522"


def main():
    audit = json.loads((ROOT / "results/submission_audit.json").read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise SystemExit("Submission audit must pass before packaging")
    files = set()
    for directory in ("src", "scripts", "tests", "notebooks", "colab", "data", "docs", "submission", "results"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".zip" and not path.name.startswith(".env"):
                files.add(path)
    for filename in ("pyproject.toml", "Makefile", ".gitignore", ".gitattributes", ".dockerignore", ".env.example", "Dockerfile.local"):
        files.add(ROOT / filename)
    files.update(ROOT.glob("*.md"))
    files.update(ROOT.glob("requirements*.txt"))
    files.update(ROOT / "adapters/correct" / filename for filename in ("adapter_model.safetensors", "adapter_config.json", "README.md"))
    archive = ROOT / "submission" / (PREFIX + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as destination:
        for path in sorted(files):
            destination.write(path, PREFIX + "/" + path.relative_to(ROOT).as_posix())
    print(f"{archive.relative_to(ROOT)}: {archive.stat().st_size / 1024**2:.2f} MiB, {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
