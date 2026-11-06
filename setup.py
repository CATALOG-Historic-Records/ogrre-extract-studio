from setuptools import setup, find_packages

setup(
    name="ogrre-extract-studio",
    version="0.1.0",
    description="Graphical desktop interface and studio for OGRRE Extract pipeline",
    packages=find_packages(),
    install_requires=[
        "dash",
        "dash-bootstrap-components",
        "dash-mantine-components",
        "plotly",
        "diskcache",
        "ogrre-extract",
    ],
    entry_points={
        "console_scripts": [
            "ogrre-extract-studio=ogrre_extract_studio.app:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)