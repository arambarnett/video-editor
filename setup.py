from setuptools import setup, find_packages

setup(
    name="ai-editor",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'openai',
        'pytest',
        'pytest-cov',
    ],
    python_requires=">=3.8",
)
