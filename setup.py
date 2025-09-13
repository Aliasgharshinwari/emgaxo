from setuptools import setup, find_packages

setup(
    name="emgaxo",
    version="0.1.1",
    author="Ali Asghar, Shahzad Bangash, Suleman Shah",
    author_email="aaliasghar8@example.com",
    description="A library containing tools for manipulating and incorporating approximate operators in ONNX Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your-library",
    packages=find_packages(where="src"),  # Look for packages inside the "src" directory
    package_dir={"": "src"},             # Tell setuptools that "src" is the root for your packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[],
)