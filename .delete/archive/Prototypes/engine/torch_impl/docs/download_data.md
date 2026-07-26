# Downloading Background Noise Dataset

The project uses a subset of the ESC-50 dataset to add background noise to the main dataset for augmentation purposes.

## Download Instructions

The dataset is provided as a single ZIP file. The download link is: `https://github.com/karoldvl/ESC-50/archive/master.zip`

### Linux / macOS

You can download the dataset using `wget` or `curl`.

#### Using `wget`

```bash
wget https://github.com/karoldvl/ESC-50/archive/master.zip -O ESC-50-master.zip
```

#### Using `curl`

```bash
curl -L https://github.com/karoldvl/ESC-50/archive/master.zip -o ESC-50-master.zip
```

After downloading, unzip the file:

```bash
unzip ESC-50-master.zip
```

This will create a directory named `ESC-50-master`. The audio files are located in the `ESC-50-master/audio` subdirectory.

### Windows

You can use PowerShell to download the dataset.

#### Using PowerShell

Open PowerShell and run the following command:

```powershell
Invoke-WebRequest -Uri https://github.com/karoldvl/ESC-50/archive/master.zip -OutFile ESC-50-master.zip
```

After downloading, you can extract the ZIP file using the File Explorer or by running the following command in PowerShell:

```powershell
Expand-Archive -Path "ESC-50-master.zip" -DestinationPath "."
```

This will create a directory named `ESC-50-master`. The audio files are located in the `ESC-50-master/audio` subdirectory.
