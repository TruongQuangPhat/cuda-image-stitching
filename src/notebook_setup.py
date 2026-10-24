"""Thiết lập notebook từ thư mục nguồn hoặc gói ZIP trên Colab."""

import importlib
import importlib.util
from pathlib import Path
import stat
import subprocess
import sys
import zipfile


def start(root):
    root = Path(root).resolve()
    if not (root / "src/notebook_support.py").is_file():
        archive = root / "Group09_Midterm.zip"
        destination = root / "cuda-image-stitching"
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("ZIP bị lỗi CRC")
            for entry in bundle.infolist():
                target = (destination / entry.filename).resolve()
                if not target.is_relative_to(destination) or stat.S_ISLNK(
                    entry.external_attr >> 16
                ):
                    raise ValueError("Đường dẫn ZIP không an toàn")
                if not (
                    entry.filename.startswith("src/")
                    or entry.filename in ("Group09_Midterm.ipynb", "examples/campus.png")
                ):
                    raise ValueError("ZIP không đúng cấu trúc dự án")
            bundle.extractall(destination)
        root = destination

    packages = {"PIL": "Pillow", "pandas": "pandas"}
    missing = [
        package for module, package in packages.items() if importlib.util.find_spec(module) is None
    ]
    if missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *missing],
            check=True,
            capture_output=True,
        )
    sys.path.insert(0, str(root / "src"))
    report = importlib.import_module("notebook_support")
    report.configure(root)
    return report
