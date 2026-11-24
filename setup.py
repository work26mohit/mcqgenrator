from setuptools import find_packages, setup

setup(
    name='mcqgenrator',
    version='0.0.1',
    author='mohit marwal',
    author_email='work26mohit@gmail.com',
    install_requires=[
        "openai",
        "langchain",
        "langchain-openai",
        "langchain-google-genai",
        "langchain-community",
        "pypdf",
        "streamlit",
        "python-dotenv",
        "pandas",
    ],
    packages=find_packages()
)