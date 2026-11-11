from pathlib import Path
import subprocess
import sys
from setuptools.command.build_py import build_py


class BuildWithVec1(build_py):
    """Compile Vec1 into the installed package instead of copying old binaries."""

    def run(self):
        if not sys.platform.startswith("linux"):
            raise RuntimeError("Vec1 automatic builds currently support Linux only.")
        super().run()
        output = Path(self.build_lib).resolve() / "dotvectordb" / "compiled_vec1"
        if self.editable_mode:
            output = Path(__file__).resolve().parent
        subprocess.run(
            ["make", f"OUTPUT_DIR={output}", "vec1"],
            cwd=Path(__file__).resolve().parent,
            check=True,
        )

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        outputs.append(str(Path(self.build_lib) / "dotvectordb/compiled_vec1/vec1.so"))
        return outputs


