import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile


spec = importlib.util.spec_from_file_location(
    "notebook_setup", Path(__file__).resolve().parents[1] / "src/notebook_setup.py"
)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class NotebookSetupTests(unittest.TestCase):
    def start(self, root):
        report = SimpleNamespace(configure=Mock())
        with patch.object(setup.importlib, "import_module", return_value=report), patch.object(
            setup.importlib.util, "find_spec", return_value=object()
        ), patch.object(sys, "path", list(sys.path)):
            self.assertIs(setup.start(root), report)
        return report

    def test_existing_source_needs_no_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "src").mkdir()
            (root / "src/notebook_support.py").write_text("")
            self.start(root).configure.assert_called_once_with(root.resolve())

    def test_colab_zip_extracts_sources_and_notebook(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with zipfile.ZipFile(root / "Group09_Midterm.zip", "w") as bundle:
                bundle.writestr("src/notebook_support.py", "# source")
                bundle.writestr("Group09_Midterm.ipynb", "{}")
                bundle.writestr("examples/campus.png", b"demo image")
            destination = root.resolve() / "cuda-image-stitching"
            self.start(root).configure.assert_called_once_with(destination)
            self.assertEqual((destination / "src/notebook_support.py").read_text(), "# source")
            self.assertTrue((destination / "Group09_Midterm.ipynb").is_file())
            self.assertEqual((destination / "examples/campus.png").read_bytes(), b"demo image")

    def test_unsafe_zip_is_rejected_before_extracting(self):
        for name in ("../escaped.txt", "src/../../escaped.txt", "unexpected.txt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                with zipfile.ZipFile(root / "Group09_Midterm.zip", "w") as bundle:
                    bundle.writestr("src/notebook_support.py", "# source")
                    bundle.writestr(name, "invalid")
                with self.assertRaises(ValueError):
                    self.start(root)
                self.assertFalse((root / "cuda-image-stitching/src/notebook_support.py").exists())
                self.assertFalse((root / "escaped.txt").exists())


if __name__ == "__main__":
    unittest.main()
