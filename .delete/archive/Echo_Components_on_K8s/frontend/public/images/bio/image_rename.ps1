Get-ChildItem -Filter *.png | ForEach-Object {
    $oldName = $_.Name
    $newName = $oldName.ToLower() -replace '\.png$', '-bio.png'
    Rename-Item $_.FullName $newName
}