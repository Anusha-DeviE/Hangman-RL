from setuptools import setup, find_packages

setup(
    name="hangman-rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "torch",
        "numpy",
        "matplotlib"
    ],
)
