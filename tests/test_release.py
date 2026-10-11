"""Ensure private runtime artifacts can never enter the source release."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import package_release


class ReleaseTests(unittest.TestCase):
    def test_release_excludes_local_data_and_secrets(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "release.zip"
            names = package_release.build_release(archive)
            with zipfile.ZipFile(archive) as source:
                self.assertIsNone(source.testzip())
                self.assertEqual(names, source.namelist())
        for name in names:
            self.assertFalse(name.startswith(("data/", "results/", "checkpoints/", ".gradio/", ".cache/")))
            self.assertNotEqual(name, ".env")
            self.assertNotIn("__pycache__", name)

    def test_credentials_are_rejected_without_printing_them(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "README.md").write_text("MSS_AUTH=" + "private:password\n")
            with patch.multiple(package_release, ROOT=root, ROOT_FILES=["README.md"], SOURCE_FILES=[]):
                with self.assertRaises(ValueError) as caught:
                    package_release.release_files()
        self.assertNotIn("private:password", str(caught.exception))

    def test_personal_absolute_path_is_rejected(self):
        self.assertTrue(package_release.PERSONAL_PATH.search("/" + "home/example/private/file"))


if __name__ == "__main__":
    unittest.main()
