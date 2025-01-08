from setuptools import setup, find_packages

setup(
    name="ray-launcher",
    version="1.1.0",
    description="An out-of-the-box ray cluster launcher",
    author="0-1CxH",
    author_email="h0.1c@foxmail.com",
    url="https://github.com/0-1CxH/ray-launcher",
    packages=find_packages(include=[
        "ray_launcher"
    ], exclude=[
        'setup.py', 'test.py'
    ]),
    include_package_data=False,
    install_requires=[
        "ray",
        "loguru"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    license = "MIT Licence",
)