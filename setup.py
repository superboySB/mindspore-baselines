import os

from setuptools import find_packages, setup

with open(os.path.join("mindspore_baselines", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

setup(
    name="mindspore_baselines",
    packages=[package for package in find_packages() if package.startswith("mindspore_baselines")],
    package_data={"mindspore_baselines": ["py.typed", "version.txt"]},
    install_requires=[
        "gym==0.21",
        "numpy",
        # For saving models
        "cloudpickle",
        # For reading logs
        "pandas",
        # Plotting learning curves
        "matplotlib",
        # gym and flake8 not compatible with importlib-metadata>5.0
        "importlib-metadata~=4.13",
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Find likely bugs
            "flake8-bugbear",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
            # For toy text Gym envs
            "scipy>=1.4.1",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
            # Copy button for code snippets
            "sphinx_copybutton",
        ],
        "extra": [
            # For render
            "opencv-python",
            "pyglet==1.5",
            # For atari games,
            "ale-py==0.7.4",
            "autorom[accept-rom-license]~=0.4.2",
            "pillow",
            # Tensorboard support
            "tensorboard>=2.9.1",
            "tensorboardX",
            # Checking memory taken by replay buffer
            "psutil",
            # For progress bar callback
            "tqdm",
            "rich",
        ],
    },
    description="MindSpore version of Stable Baselines3, implementations of reinforcement learning algorithms.",
    author="Zipeng Dai",
    url="https://github.com/superboySB/mindspore-baselines",
    author_email="604896160@qq.com",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
             "gym openai stable baselines toolbox python data-science",
    license="MIT",
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.7",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
