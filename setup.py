from setuptools import setup, find_packages

setup(
    name="openai-json",
    version="1.0.1",
    description="A Python library for processing and structuring JSON responses from the OpenAI API.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Shane Vigil",
    author_email="savigil@gmail.com",
    url="https://github.com/shanevigil/openai-json",
    license="MIT",
    packages=find_packages(where="openai_json"),
    package_dir={"": "openai_json"},
    python_requires=">=3.8",
    install_requires=[
        "jsonschema==4.23.0",
        "openai==1.58.1",
        "word2number==1.1",
        "rapidfuzz==3.11.0",
        "requests==2.32.3",
        "joblib==1.4.2",
        "scikit-learn==1.6.0",
        "numpy==2.0.2",
        "torch==2.0.1",
        "transformers==4.47.1",
        "safetensors==0.4.5",
    ],
    extras_require={
        "testing": [
            "pytest==8.3.4",
            "pytest-asyncio==0.25.0",
        ],
        "linting": ["flake8==7.1.1"],
        "documentation": [
            "Sphinx==7.4.7",
            "sphinx-rtd-theme==3.0.2",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="openai json schema machine-learning api",
    project_urls={
        "Bug Tracker": "https://github.com/shanevigil/openai-json/issues",
        "Source Code": "https://github.com/shanevigil/openai-json",
    },
)
