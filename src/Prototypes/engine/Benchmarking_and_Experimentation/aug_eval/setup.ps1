# init.ps1 - Fast small dataset setup with real speech clips

$ErrorActionPreference = "Stop"

# Output dirs
$dirs = @(
  "project_data/id/class_a",
  "project_data/id/class_b",
  "project_data/ood/class_a",
  "project_data/ood/class_b",
  "assets/noise",
  "assets/rir"
)
$dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

# Dataset source
$zipUrl  = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
$zipPath = "$PSScriptRoot\mini_speech_commands.zip"
$dataDir = "$PSScriptRoot\mini_speech_commands"

# Labels mapping
$labels = @("yes","no")
$labelToClass = @{
  "yes" = @{ ID = "project_data/id/class_a"; OOD = "project_data/ood/class_a" }
  "no"  = @{ ID = "project_data/id/class_b"; OOD = "project_data/ood/class_b" }
}

# Download dataset
if (-not (Test-Path -LiteralPath $dataDir)) {
    Write-Host "Downloading mini_speech_commands..."
    Start-BitsTransfer -Source $zipUrl -Destination $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $PSScriptRoot -Force
    Remove-Item $zipPath -Force
}

# Function to extract speaker ID from filename
function Get-SpeakerId($fileName) {
  $base = [IO.Path]::GetFileNameWithoutExtension($fileName)
  if ($base.Contains("_")) { return $base.Split("_")[0] }
  return $base
}

# Copy samples into ID/OOD by unseen speakers
function Copy-SplitBySpeaker {
  param($srcDir, $dstId, $dstOod, $maxId, $maxOod)
  $files = Get-ChildItem -Path $srcDir -Filter *.wav
  $bySpeaker = $files | Group-Object { Get-SpeakerId $_.Name }
  $speakers = $bySpeaker | ForEach-Object { $_.Name } | Sort-Object
  $nOod = [Math]::Max(1, [Math]::Floor($speakers.Count * 0.2))
  $oodSpeakers = $speakers[0..($nOod-1)]
  $idSpeakers  = $speakers[$nOod..($speakers.Count-1)]

  $idFiles  = $bySpeaker | Where-Object { $idSpeakers  -contains $_.Name } | ForEach-Object { $_.Group } | Select-Object -First $maxId
  $oodFiles = $bySpeaker | Where-Object { $oodSpeakers -contains $_.Name } | ForEach-Object { $_.Group } | Select-Object -First $maxOod

  $i = 0
  foreach ($f in $idFiles)  { Copy-Item $f.FullName "$dstId\$($f.BaseName)_id_$i.wav";  $i++ }
  $j = 0
  foreach ($f in $oodFiles) { Copy-Item $f.FullName "$dstOod\$($f.BaseName)_ood_$j.wav"; $j++ }
}

# Split yes/no into ID/OOD
foreach ($lab in $labels) {
  $src    = Join-Path $dataDir $lab
  $dstId  = $labelToClass[$lab].ID
  $dstOod = $labelToClass[$lab].OOD
  Copy-SplitBySpeaker $src $dstId $dstOod 120 60
}

# Noise + RIR generators (simple placeholders)
function New-SineWav($path, $freq, $sec, $sr) {
  $n = [int]($sr * $sec)
  $samples = 0..($n-1) | ForEach-Object { [math]::Sin(2*[math]::PI*$freq*$_/$sr) }
  $int16 = $samples | ForEach-Object { [int16]($_ * 32767) }
  $bytes = New-Object byte[] ($int16.Count*2)
  [System.Buffer]::BlockCopy($int16, 0, $bytes, 0, $bytes.Length)
  $fs = [IO.File]::Create($path)
  $bw = New-Object IO.BinaryWriter($fs)
  $bw.Write([Text.Encoding]::ASCII.GetBytes('RIFF'))
  $bw.Write([BitConverter]::GetBytes(36 + $bytes.Length))
  $bw.Write([Text.Encoding]::ASCII.GetBytes('WAVEfmt '))
  $bw.Write([BitConverter]::GetBytes(16))
  $bw.Write([BitConverter]::GetBytes(1))
  $bw.Write([BitConverter]::GetBytes(1))
  $bw.Write([BitConverter]::GetBytes($sr))
  $bw.Write([BitConverter]::GetBytes($sr*2))
  $bw.Write([BitConverter]::GetBytes(2))
  $bw.Write([BitConverter]::GetBytes(16))
  $bw.Write([Text.Encoding]::ASCII.GetBytes('data'))
  $bw.Write([BitConverter]::GetBytes($bytes.Length))
  $bw.Write($bytes)
  $bw.Close(); $fs.Close()
}

function New-NoiseWav($path, $sec, $sr) {
  $rand = New-Object Random 42
  $n = [int]($sr * $sec)
  $samples = 0..($n-1) | ForEach-Object { ($rand.NextDouble()*2) - 1 }
  $int16 = $samples | ForEach-Object { [int16]($_ * 32767) }
  $bytes = New-Object byte[] ($int16.Count*2)
  [System.Buffer]::BlockCopy($int16, 0, $bytes, 0, $bytes.Length)
  $fs = [IO.File]::Create($path)
  $bw = New-Object IO.BinaryWriter($fs)
  $bw.Write([Text.Encoding]::ASCII.GetBytes('RIFF'))
  $bw.Write([BitConverter]::GetBytes(36 + $bytes.Length))
  $bw.Write([Text.Encoding]::ASCII.GetBytes('WAVEfmt '))
  $bw.Write([BitConverter]::GetBytes(16))
  $bw.Write([BitConverter]::GetBytes(1))
  $bw.Write([BitConverter]::GetBytes(1))
  $bw.Write([BitConverter]::GetBytes($sr))
  $bw.Write([BitConverter]::GetBytes($sr*2))
  $bw.Write([BitConverter]::GetBytes(2))
  $bw.Write([BitConverter]::GetBytes(16))
  $bw.Write([Text.Encoding]::ASCII.GetBytes('data'))
  $bw.Write([BitConverter]::GetBytes($bytes.Length))
  $bw.Write($bytes)
  $bw.Close(); $fs.Close()
}

# Make 3 noise wavs
New-NoiseWav "assets/noise/noise1.wav" 2 16000
New-NoiseWav "assets/noise/noise2.wav" 2 16000
New-NoiseWav "assets/noise/noise3.wav" 2 16000

# Make 3 RIR placeholders
New-SineWav "assets/rir/rir_t60_0.3s_near.wav" 2000 0.15 16000
New-SineWav "assets/rir/rir_t60_0.6s_mid.wav" 1500 0.30 16000
New-SineWav "assets/rir/rir_t60_1.0s_far.wav" 1200 0.50 16000

Write-Host "Dataset ready: ID/OOD split with noise and RIR assets."
