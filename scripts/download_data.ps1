# Download datasets via Kaggle CLI (PowerShell).
# Prereq:
#   pip install kaggle
#   Put kaggle.json in  $env:USERPROFILE\.kaggle\kaggle.json

$ErrorActionPreference = "Stop"

function Expand-Zips($path) {
  Get-ChildItem -Path $path -Filter *.zip -File | ForEach-Object {
    Write-Host "[info] Unzipping $($_.Name)"
    Expand-Archive -Path $_.FullName -DestinationPath $path -Force
    Remove-Item $_.FullName -Force
  }
}

$datasets = @(
  @{ name = "cifar10";      slug = "petitbonney/cifar10-image-recognition" },
  @{ name = "fashionmnist"; slug = "zalando-research/fashionmnist" },
  @{ name = "imagenet100";  slug = "ambityga/imagenet100" },
  @{ name = "tiny-imagenet";slug = "akash2sharma/tiny-imagenet" },
  @{ name = "imagenetv2";   slug = "derrickdaniel/imagenet-v2-test" }
)

foreach ($d in $datasets) {
  $out = "data/$($d.name)"
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  Write-Host "[info] Downloading $($d.slug) -> $out"
  # use python module form so PATH to kaggle.exe isn't an issue
  py -m kaggle datasets download -d $d.slug -p $out -q
  Expand-Zips $out
  Write-Host "✓ $($d.name) ready at $out`n"
}

Write-Host "All done. Top-level contents of data/:"
Get-ChildItem data