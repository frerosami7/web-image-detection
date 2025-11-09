from setuptools import setup, find_packages

setup(
    name='image-anomaly-detection',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for detecting anomalies in images using deep learning techniques.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/enrico310786/Image_Anomaly_Detection',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'opencv-python',
        'Pillow',
        'tensorflow',  # or 'torch' if using PyTorch
        'scikit-learn',
        'flask',  # or 'fastapi' if using FastAPI for the API
        'gradio',
        'streamlit'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)