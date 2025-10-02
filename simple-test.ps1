# Simple Test Speech Service
$speechServiceUrl = "http://localhost:3005"
$audioFile = "D:\TLCN\vocabu-rex-speech-service\test.mp3"
$referenceText = "Hello world"

Write-Host "Testing Speech Service with audio file" -ForegroundColor Green
Write-Host "Audio File: $audioFile" -ForegroundColor Cyan

# Check if file exists
if (!(Test-Path $audioFile)) {
    Write-Host "Audio file not found!" -ForegroundColor Red
    exit 1
}

Write-Host "File exists" -ForegroundColor Green

# Test Health Check
Write-Host "Step 0: Health Check" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$speechServiceUrl/health" -Method GET
    Write-Host "Health OK: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "Health failed: $_" -ForegroundColor Red
}

# Test Step 1: Phonemization
Write-Host "Step 1: Phonemization" -ForegroundColor Yellow
try {
    $step1Result = & curl -X POST -F "text=$referenceText" "$speechServiceUrl/api/v1/speech/step1/phonemize" 2>$null
    Write-Host "Step 1 Result:" -ForegroundColor Green
    Write-Host $step1Result -ForegroundColor White
} catch {
    Write-Host "Step 1 failed: $_" -ForegroundColor Red
}

# Test Step 2: Forced Alignment
Write-Host "Step 2: Forced Alignment" -ForegroundColor Yellow
try {
    $step2Result = & curl -X POST -F "audio_file=@`"$audioFile`"" -F "text=$referenceText" "$speechServiceUrl/api/v1/speech/step2/align" 2>$null
    Write-Host "Step 2 Result:" -ForegroundColor Green
    Write-Host $step2Result -ForegroundColor White
} catch {
    Write-Host "Step 2 failed: $_" -ForegroundColor Red
}

# Test Step 3: Enhanced ASR
Write-Host "Step 3: Enhanced ASR" -ForegroundColor Yellow
try {
    $step3Result = & curl -X POST -F "audio_file=@`"$audioFile`"" -F "reference_text=$referenceText" "$speechServiceUrl/api/v1/speech/step3/asr" 2>$null
    Write-Host "Step 3 Result:" -ForegroundColor Green
    Write-Host $step3Result -ForegroundColor White
} catch {
    Write-Host "Step 3 failed: $_" -ForegroundColor Red
}

# Test Step 4: Comprehensive Scoring
Write-Host "Step 4: Comprehensive Scoring" -ForegroundColor Yellow
try {
    $step4Result = & curl -X POST -F "audio_file=@`"$audioFile`"" -F "reference_text=$referenceText" "$speechServiceUrl/api/v1/speech/step4/score" 2>$null
    Write-Host "Step 4 Result:" -ForegroundColor Green
    Write-Host $step4Result -ForegroundColor White
} catch {
    Write-Host "Step 4 failed: $_" -ForegroundColor Red
}

# Test Full Pipeline
Write-Host "Full Pipeline Assessment" -ForegroundColor Yellow
try {
    $fullResult = & curl -X POST -F "audio_file=@`"$audioFile`"" -F "reference_text=$referenceText" "$speechServiceUrl/api/v1/speech/full-assessment" 2>$null
    Write-Host "Full Pipeline Result:" -ForegroundColor Green
    Write-Host $fullResult -ForegroundColor White
} catch {
    Write-Host "Full Pipeline failed: $_" -ForegroundColor Red
}

Write-Host "Test completed!" -ForegroundColor Green