from pathlib import Path
from setuptools import find_packages, setup

root = Path(__file__).parent

setup(
    # The name users type: pip install my-package
    name="dotvectordb",

    # Increase this for every PyPI release
    version="0.1.0",

    description="An asynchronous vector and graph database built for the research paper DotRAG: Retrieval-Time Reasoning Along Paths",

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

    package_data={
        "dot_vectordb": [
            "compiled_vec1/vec1.so",
            "schema/*.sql",
        ],
    },

    # Packages automatically installed with yours
    install_requires=[
        "pytest==9.0.3",
        "torch==2.10.0",
        "pytest-asyncio==1.4.0",
        "aiosqlite==0.22.1",
        "sentence-transformers==5.5.1",
        "python-dotenv==1.2.2",
        "openai==2.44.0",
        "google-genai==2.12.0",
        "aiohttp==3.13.3",
        "tiktoken==0.13.0",
        "igraph==1.0.0",
        "ipykernel==7.2.0",
        "pdfplumber==0.11.6",
        "python-docx==1.2.0",
        "python-pptx==1.0.2",
        "openpyxl==3.1.5",
    ],

    classifiers=[
    "Programming Language :: Python :: 3",
    "Operating System :: POSIX :: Linux",
    ]
)