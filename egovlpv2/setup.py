#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "EgoVLPv2: Egocentric Video-Language Pre-training Framework"

setup(
    name="egovlpv2",
    version="1.0.0",
    description="EgoVLPv2: Egocentric Video-Language Pre-training Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="EgoVLPv2 Team",
    author_email="",
    url="https://github.com/facebookresearch/EgoVLPv2",
    
    # 自动发现所有包 - 只包含egovlpv2包及其子包
    packages=find_packages(include=['egovlpv2', 'egovlpv2.*']),
    
    # Python版本要求
    python_requires=">=3.7",
    
    # 依赖包
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "pandas>=1.0.0",
        "numpy==1.26.4",
        "opencv-python>=4.0.0",
        "Pillow>=8.0.0",
        "dominate",
        "ffmpeg",
        "av",
        "humanize",
        "timm",
        "easydict",
        "tensorboardX",
        "webdataset",
        "scikit-learn",
        "PyYAML>=5.4.0",
        "decord>=0.6.0",
        "transformers>=4.0.0",
        "tensorboard>=2.0.0",
    ],
    
    # 可选依赖
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "isort>=5.0.0",
        ],
    },
    
    # 分类
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # 包含非Python文件
    include_package_data=True,
    
    # 包数据
    package_data={
        "egovlpv2": ["configs/*.yml", "configs/*.yaml", "*.md"],
    },
) 