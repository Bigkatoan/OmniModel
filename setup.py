from setuptools import setup, find_packages

setup(
    name="omni_model",
    version="0.1.0",
    author="Bigkatoan",
    description="A lightweight Dual-Encoder model for Vision-Language tasks",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Bigkatoan/OmniModel",
    packages=find_packages(),
    include_package_data=True, # Quan trọng để lấy file config
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "opencv-python",
        "albumentations",
        "pyyaml",
        "huggingface_hub", # Bắt buộc phải có
        "timm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
