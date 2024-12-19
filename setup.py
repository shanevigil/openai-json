from setuptools import setup, find_packages

setup(
    name="openai-json",
    version="0.1.0",
    description="A Python wrapper for processing and structuring JSON responses from the OpenAI API.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Shane Vigil",
    author_email="savigil@gmail.com",
    url="https://github.com/shanevigil/openai-json", # TODO Add my username
    license="MIT",
    packages=find_packages(where="openai_json"),
    package_dir={"": "openai_json"},
    python_requires=">=3.8",
    install_requires=[
        "openai",
        "joblib",
        "scikit-learn",
        "jsonschema",
    ],
    extras_require={
        "testing": ["pytest", "pytest-cov"],
        "linting": ["flake8", "mypy"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="openai json wrapper schema machine-learning api",
    project_urls={
        "Bug Tracker": "https://github.com/shanevigil/openai-json/issues", # TODO Add my username
        "Source Code": "https://github.com/shanevigil/openai-json", # TODO Add my username
    },
)
