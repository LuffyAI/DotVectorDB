from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel


class PlatformWheel(bdist_wheel):
    """Tag wheels as platform-specific because they bundle ``vec1.so``."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag

root = Path(__file__).parent

setup(
    # The name users type: pip install my-package
    name="dotvectordb",

    # Increase this for every PyPI release
    version="0.1.0",

    description="DotVectorDB is a simple, transparent vector and graph database that makes building GraphRAG pipelines easier without hiding the underlying process.",

    # Displayed on the PyPI project page
    long_description=(root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",

    author="Larnell Moore",
    author_email="larnell@umich.edu",

    # Finds folders containing __init__.py
    packages=find_packages(),

    # Minimum supported Python version
    python_requires=">=3.9",
    include_package_data=True,
    cmdclass={"bdist_wheel": PlatformWheel},

    package_data={
        "dotvectordb": [
            "compiled_vec1/vec1.so",
            "compiled_vec1/vec1_gl.so",
            "schema/*.sql",
            "web_interface/*.html",
            "sql_browser/*.html",
            "sql_browser/*.svg",
        ],
    },

    # Packages automatically installed with yours
    install_requires=[
        "torch==2.10.0",
        "aiosqlite==0.22.1",
        "sentence-transformers==5.5.1",
        "python-dotenv==1.2.2",
        "openai==2.44.0",
        "google-genai==2.12.0",
        "aiohttp==3.13.3",
        "tiktoken==0.13.0",
        "igraph==1.0.0",
        "pdfplumber==0.11.6",
        "python-docx==1.2.0",
        "python-pptx==1.0.2",
        "openpyxl==3.1.5",
        "pydantic>=2.0,<3.0",
    ],

    classifiers=[
    "Programming Language :: Python :: 3",
    "Operating System :: POSIX :: Linux",
    ]
)
