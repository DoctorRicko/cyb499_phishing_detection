# Load the dataset
$data = Import-Csv "data\extracted\phishing\enron.csv"

# Split into phishing (1) and legitimate (0)
$phishing = $data | Where-Object { $_.label -eq 1 }
$legit = $data | Where-Object { $_.label -eq 0 }

# Set sample sizes (adjust counts to match your needs)
$trainCount = [math]::Floor($phishing.Count * 0.7)
$valTestCount = $phishing.Count - $trainCount
$valCount = [math]::Floor($valTestCount / 2)
$testCount = $valTestCount - $valCount

# Randomly shuffle and split PHISHING emails
$phishingShuffled = $phishing | Get-Random -Count $phishing.Count
$phishingTrain = $phishingShuffled[0..($trainCount-1)]
$phishingVal = $phishingShuffled[$trainCount..($trainCount+$valCount-1)]
$phishingTest = $phishingShuffled[($trainCount+$valCount)..($phishing.Count-1)]

# Randomly shuffle and split LEGITIMATE emails
$legitShuffled = $legit | Get-Random -Count $legit.Count
$legitTrain = $legitShuffled[0..($trainCount-1)]
$legitVal = $legitShuffled[$trainCount..($trainCount+$valCount-1)]
$legitTest = $legitShuffled[($trainCount+$valCount)..($legit.Count-1)]

# Combine and export
@(
    ($phishingTrain + $legitTrain | Export-Csv "data/train.csv" -NoTypeInformation),
    ($phishingVal + $legitVal | Export-Csv "data/val.csv" -NoTypeInformation),
    ($phishingTest + $legitTest | Export-Csv "data/test.csv" -NoTypeInformation)
)

Write-Host "Splitting complete! Files saved to:"
Write-Host "- Training set: data/train.csv ($($phishingTrain.Count + $legitTrain.Count) emails"
Write-Host "- Validation set: data/val.csv ($($phishingVal.Count + $legitVal.Count) emails"
Write-Host "- Test set: data/test.csv ($($phishingTest.Count + $legitTest.Count) emails"