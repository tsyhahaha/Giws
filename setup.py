from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if not line.startswith("#")]

# 读取README作为长描述
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="giws",  # 包名（pip install 用的名称）
    version="0.1.0",          # 初始版本
    author="SiyuanTao",
    author_email="taosiyuan24s@ict.ac.cn",
    description="Give It a Whirl: Best practices for pytorch training.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsyhahaha/Giws",
    
    packages=find_packages(exclude=["tests*", "docs*", "data*"]),
    python_requires=">=3.8",
    install_requires=parse_requirements("requirements.txt"),
    
    include_package_data=True,
    package_data={
        "configs": ["*.yaml"],
    },
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)