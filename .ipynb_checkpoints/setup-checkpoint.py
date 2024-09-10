from setuptools import setup, find_packages

# Helper function to read the contents of requirements.txt
def parse_requirements(filename):
    """Read and return dependencies from the specified requirements file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
# Parse dependencies from requirements.txt
requirements = parse_requirements('requirements.txt')
dev_requirements = parse_requirements('dev-requirements.txt')

setup(
    name="complete_collision_recorder",
    version="0.1.0",
    author="Daniel Zielinski",
    author_email="hip_dog_fur@runbox.com",
    description="Creates narratives of collisions from unstructured sources",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/beardedDan/complete_collision_recorder",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": dev_requirements,},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "collision-recorder=src.main:main",
        ],
    },
)
