from setuptools import setup

with open("VERSION", "r") as f:
    __version__ = f.read()

setup(
    name="recipeqa",
    version=__version__,
    description="Recipe QA with LLM",
    author="Cuong Nguyen",
    author_email="cuong.nguyen1004@gmail.com",
    license="MIT",
    packages=["recipeqa"],
    install_requires=["langchain>=0.0.80", "openai", "faiss-cpu"],
)
1