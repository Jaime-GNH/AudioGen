from setuptools import setup

setup(
    name='audio_gen',
    version='1.0.0.0',
    install_requires=[
        "streamlit==1.4.0",
        "diffusers==0.31.0",
        "torch==2.4.1+cu118",
        "scipy==1.14.1",
        "librosa==0.10.2.post1",
        "ffmpeg-python==0.2.0",
        "samplerate==0.2.1",
        "python-dotenv==1.0.1"
    ],
    packages=['audio_gen'],
    description="Audio and Music short track AI generation"
)
