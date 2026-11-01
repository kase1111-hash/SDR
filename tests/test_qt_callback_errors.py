"""An exception in a Qt slot or event handler fails the test it happens in
instead of aborting the whole run (see tests/conftest.py)."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestQtCallbackErrors(unittest.TestCase):
    def test_a_raising_slot_fails_its_test_and_the_run_goes_on(self):
        tests = textwrap.dedent("""
            from PyQt6.QtCore import QObject, pyqtSignal
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance() or QApplication([])


            class Source(QObject):
                ping = pyqtSignal()


            def test_raising_slot():
                source = Source()

                def slot():
                    raise ValueError("raised in a slot")

                source.ping.connect(slot)
                source.ping.emit()


            def test_after():
                assert True
            """)
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "test_raising.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(tests)
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        run = subprocess.run(
            # The suite's conftest, loaded as a plugin for the file outside it.
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            + ["-p", "tests.conftest", "--rootdir", folder, path],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        out = run.stdout + run.stderr
        self.assertNotEqual(run.returncode, -6, out)  # not aborted
        self.assertIn("1 failed, 1 passed", out)
        self.assertRegex(out, r"FAILED \S*test_raising\.py::test_raising_slot")
        self.assertIn("ValueError: raised in a slot", out)
