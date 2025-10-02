# Test Speech Service với file audio thật
# File: test-speech-api.ps1

$speechServiceUrl = "http://localhost:3007"
$audioFile = "D:\TLCN\vocabu-rex-speech-service\test.mp3"
$referenceText = "Hello world, this is a pronunciation test."

Write-Host "🎤 Testing VocabuRex Speech Service với file audio thật" -ForegroundColor Green
Write-Host "Audio File: $audioFile" -ForegroundColor Cyan
Write-Host "Reference Text: $referenceText" -ForegroundColor Cyan
Write-Host ""

# Check if audio file exists
if (!(Test-Path $audioFile)) {
    Write-Host "❌ Không tìm thấy file audio: $audioFile" -ForegroundColor Red
    exit 1
}

Write-Host "✅ File audio tồn tại: $(Get-Item $audioFile | Select-Object Name, Length)" -ForegroundColor Green
Write-Host ""

# Test 1: Health Check
Write-Host "📋 Step 0: Health Check" -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Uri "$speechServiceUrl/health" -Method GET
    Write-Host "✅ Health Check thành công:" -ForegroundColor Green
    Write-Host ($healthResponse | ConvertTo-Json -Depth 3) -ForegroundColor White
} catch {
    Write-Host "❌ Health Check thất bại: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Test 2: Step 1 - Phonemization
Write-Host "🔤 Step 1: Phonemization (Text → Phonemes)" -ForegroundColor Yellow
try {
    $phonemizationBody = @{
        text = $referenceText
        language = "english"
    } | ConvertTo-Json

    $step1Response = Invoke-RestMethod -Uri "$speechServiceUrl/api/v1/phonemize" -Method POST `
        -Body $phonemizationBody -ContentType "application/json"
    
    Write-Host "✅ Step 1 thành công:" -ForegroundColor Green
    Write-Host "Phonemes: $($step1Response.phonemes -join ' ')" -ForegroundColor White
    Write-Host "Total phonemes: $($step1Response.phonemes.Count)" -ForegroundColor White
} catch {
    Write-Host "❌ Step 1 thất bại: $_" -ForegroundColor Red
}
Write-Host ""

# Test 3: Step 2 - Forced Alignment (cần file audio)
Write-Host "⏱️  Step 2: Forced Alignment (Audio + Text → Timing)" -ForegroundColor Yellow
try {
    # Create multipart form data for file upload
    $boundary = [System.Guid]::NewGuid().ToString()
    $LF = "`r`n"
    
    $bodyLines = (
        "--$boundary",
        "Content-Disposition: form-data; name=`"audio_file`"; filename=`"test.mp3`"",
        "Content-Type: audio/mpeg$LF",
        [System.IO.File]::ReadAllBytes($audioFile),
        "--$boundary",
        "Content-Disposition: form-data; name=`"text`"$LF",
        $referenceText,
        "--$boundary",
        "Content-Disposition: form-data; name=`"language`"$LF", 
        "english",
        "--$boundary--$LF"
    ) -join $LF

    $step2Response = Invoke-RestMethod -Uri "$speechServiceUrl/api/v1/align" -Method POST `
        -Body $bodyLines -ContentType "multipart/form-data; boundary=$boundary"
    
    Write-Host "✅ Step 2 thành công:" -ForegroundColor Green
    Write-Host "Words aligned: $($step2Response.words.Count)" -ForegroundColor White
    Write-Host "Total duration: $($step2Response.total_duration)s" -ForegroundColor White
} catch {
    Write-Host "❌ Step 2 thất bại: $_" -ForegroundColor Red
    Write-Host "Error details: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 4: Step 3 - Enhanced ASR
Write-Host "🎯 Step 3: Enhanced ASR (Audio → Transcription + Analysis)" -ForegroundColor Yellow
try {
    # Use curl for multipart file upload (PowerShell Invoke-RestMethod has issues with files)
    $curlArgs = @(
        "-X", "POST"
        "-F", "audio_file=@`"$audioFile`""
        "-F", "reference_text=$referenceText"
        "-F", "language=english"
        "-F", "model_size=base"
        "-F", "include_phonemes=true"
        "-F", "compare_pronunciation=true"
        "$speechServiceUrl/api/v1/asr/transcribe"
    )
    
    $step3Result = & curl @curlArgs 2>$null
    $step3Response = $step3Result | ConvertFrom-Json
    
    Write-Host "✅ Step 3 thành công:" -ForegroundColor Green
    Write-Host "Transcribed: '$($step3Response.actual_utterance.transcribed_text)'" -ForegroundColor White
    Write-Host "Pronunciation Score: $($step3Response.overall_pronunciation_score)" -ForegroundColor White
    Write-Host "Fluency Score: $($step3Response.fluency_score)" -ForegroundColor White
    Write-Host "Total Score: $($step3Response.total_score)" -ForegroundColor White
    Write-Host "Grade: $($step3Response.pronunciation_grade)" -ForegroundColor White
} catch {
    Write-Host "❌ Step 3 thất bại: $_" -ForegroundColor Red
    Write-Host "Error details: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

# Test 5: Step 4 - Full Pipeline Assessment
Write-Host "🏆 Step 4: Full Pipeline Assessment (All Steps Combined)" -ForegroundColor Yellow
try {
    # Use curl for Step 4 as well
    $curlArgs4 = @(
        "-X", "POST"
        "-F", "audio_file=@`"$audioFile`""
        "-F", "text=$referenceText"
        "-F", "language=english"
        "-F", "whisper_model_size=base"
        "-F", "include_phonemization=true"
        "-F", "include_alignment=true"
        "-F", "include_asr_analysis=true"
        "$speechServiceUrl/api/v1/pronunciation/analyze-full"
    )
    
    $step4Result = & curl @curlArgs4 2>$null
    $step4Response = $step4Result | ConvertFrom-Json
    
    Write-Host "✅ Step 4 (Full Pipeline) thành công:" -ForegroundColor Green
    Write-Host "Pipeline Success: $($step4Response.success)" -ForegroundColor White
    Write-Host "Phonemization: $($step4Response.phonemization_success)" -ForegroundColor White
    Write-Host "Alignment: $($step4Response.alignment_success)" -ForegroundColor White
    Write-Host "ASR Analysis: $($step4Response.asr_success)" -ForegroundColor White
    
    if ($step4Response.asr_result) {
        Write-Host "Final Assessment:" -ForegroundColor Cyan
        Write-Host "  - Overall Score: $($step4Response.asr_result.total_score)" -ForegroundColor White
        Write-Host "  - Grade: $($step4Response.asr_result.pronunciation_grade)" -ForegroundColor White
        Write-Host "  - Feedback: $($step4Response.asr_result.feedback.overall_feedback)" -ForegroundColor White
    }
    
    Write-Host "Learning Recommendations:" -ForegroundColor Cyan
    foreach ($recommendation in $step4Response.learning_recommendations) {
        Write-Host "  - $recommendation" -ForegroundColor White
    }
} catch {
    Write-Host "❌ Step 4 thất bại: $_" -ForegroundColor Red
    Write-Host "Error details: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host ""

Write-Host "🎉 Test hoan tat! Tat ca 4 buoc da duoc test voi file audio that." -ForegroundColor Green