# Project Echo

Project echo is a bioaccouostic classification tool.


## Setup

We used Python 3.9.9 and [PyTorch](https://pytorch.org/) 1.10.1, but the codebase is expected to be compatible with Python 3.7 or later and recent PyTorch versions. The codebase also depends on a few Python packages like [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) for reading audio files. The following command will pull and install the latest commit from this repository, along with its Python dependencies 

    pip install git+https://github.com/stephankokkas/project-echo/pip-package.git

To update the package to the latest version of this repository, please run:

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/stephankokkas/project-echo/pip-package.git

It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

## Command-line usage

Not yet implemented


## Python usage

Transcription can also be performed within Python: 

```python
import project_echo

model = echo.load_model("base")
result = model.predict("audio.mp3")
print(result["text"])
```

Internally, the `transcribe()` method reads the entire file and processes the audio with a sliding 30-second window, performing autoregressive sequence-to-sequence predictions on each window.
