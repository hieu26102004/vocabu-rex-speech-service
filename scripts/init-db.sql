-- Initialize Speech Service Database

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables for speech service domain

-- Audio Sessions table - tracks audio processing sessions
CREATE TABLE IF NOT EXISTS audio_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    session_type VARCHAR(50) NOT NULL, -- 'pronunciation', 'conversation', 'dictation'
    status VARCHAR(20) NOT NULL DEFAULT 'active', -- 'active', 'completed', 'failed'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

-- Speech Recognition Records table
CREATE TABLE IF NOT EXISTS speech_recognitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES audio_sessions(id) ON DELETE CASCADE,
    audio_file_path VARCHAR(500),
    original_text TEXT,
    recognized_text TEXT,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    processing_duration_ms INTEGER,
    recognition_engine VARCHAR(50), -- 'whisper', 'google', 'azure'
    language_code VARCHAR(10) DEFAULT 'en-US',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Pronunciation Assessments table - Enhanced for Forced Alignment
CREATE TABLE IF NOT EXISTS pronunciation_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES audio_sessions(id) ON DELETE CASCADE,
    recognition_id UUID REFERENCES speech_recognitions(id) ON DELETE CASCADE,
    target_text TEXT NOT NULL,
    spoken_text TEXT,
    
    -- Overall Scores
    overall_score FLOAT CHECK (overall_score >= 0 AND overall_score <= 100),
    accuracy_score FLOAT CHECK (accuracy_score >= 0 AND accuracy_score <= 100),
    fluency_score FLOAT CHECK (fluency_score >= 0 AND fluency_score <= 100),
    pronunciation_score FLOAT CHECK (pronunciation_score >= 0 AND pronunciation_score <= 100),
    
    -- Step 1: Phonemization Results
    target_phonemes JSONB DEFAULT '[]', -- [{"word": "hello", "phonemes": ["h", "ə", "l", "oʊ"], "ipa": "/hɛloʊ/"}]
    
    -- Step 2: Forced Alignment Results  
    alignment_data JSONB DEFAULT '[]', -- [{"phoneme": "h", "start": 0.1, "end": 0.2, "confidence": 0.95}]
    
    -- Step 3: ASR Transcription
    asr_phonemes JSONB DEFAULT '[]', -- Actual phonemes from Whisper transcription
    
    -- Step 4: Detailed Error Analysis
    phoneme_errors JSONB DEFAULT '[]', -- [{"phoneme": "l", "expected": "l", "actual": "r", "error_type": "substitution", "start": 0.3, "end": 0.4, "severity": "major"}]
    word_level_scores JSONB DEFAULT '[]', -- [{"word": "hello", "score": 75, "errors": [...]}]
    
    -- Processing Metadata
    phonemizer_engine VARCHAR(50) DEFAULT 'espeak', -- 'espeak', 'festival'
    alignment_engine VARCHAR(50) DEFAULT 'mfa', -- 'mfa', 'gentle', 'wav2vec2'
    assessment_engine VARCHAR(50) DEFAULT 'custom',
    processing_duration_ms INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Phoneme Analysis table - Detailed phoneme-level analysis
CREATE TABLE IF NOT EXISTS phoneme_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    assessment_id UUID REFERENCES pronunciation_assessments(id) ON DELETE CASCADE,
    word_index INTEGER NOT NULL, -- Position of word in sentence
    phoneme_index INTEGER NOT NULL, -- Position of phoneme in word
    
    -- Target phoneme info
    target_phoneme VARCHAR(10) NOT NULL,
    target_ipa VARCHAR(20),
    
    -- Actual phoneme info from alignment
    actual_phoneme VARCHAR(10),
    actual_ipa VARCHAR(20),
    
    -- Timing from Forced Alignment
    start_time FLOAT NOT NULL, -- Start time in seconds
    end_time FLOAT NOT NULL, -- End time in seconds
    duration FLOAT GENERATED ALWAYS AS (end_time - start_time) STORED,
    
    -- Quality metrics
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    acoustic_score FLOAT,
    pronunciation_accuracy FLOAT CHECK (pronunciation_accuracy >= 0 AND pronunciation_accuracy <= 100),
    
    -- Error classification
    error_type VARCHAR(20), -- 'substitution', 'omission', 'insertion', 'distortion', 'correct'
    error_severity VARCHAR(10), -- 'minor', 'major', 'critical'
    
    -- Audio features
    formant_data JSONB DEFAULT '{}', -- F1, F2, F3 values
    pitch_data JSONB DEFAULT '{}', -- Fundamental frequency info
    energy_level FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Word Analysis table - Word-level pronunciation analysis
CREATE TABLE IF NOT EXISTS word_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    assessment_id UUID REFERENCES pronunciation_assessments(id) ON DELETE CASCADE,
    word_index INTEGER NOT NULL,
    
    -- Word info
    target_word VARCHAR(100) NOT NULL,
    spoken_word VARCHAR(100),
    
    -- Timing
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    duration FLOAT GENERATED ALWAYS AS (end_time - start_time) STORED,
    
    -- Scores
    word_score FLOAT CHECK (word_score >= 0 AND word_score <= 100),
    stress_accuracy FLOAT CHECK (stress_accuracy >= 0 AND stress_accuracy <= 100),
    rhythm_score FLOAT CHECK (rhythm_score >= 0 AND rhythm_score <= 100),
    
    -- Error summary
    total_phoneme_errors INTEGER DEFAULT 0,
    substitution_errors INTEGER DEFAULT 0,
    omission_errors INTEGER DEFAULT 0,
    insertion_errors INTEGER DEFAULT 0,
    
    -- Phoneme breakdown
    phoneme_count INTEGER NOT NULL,
    correct_phonemes INTEGER DEFAULT 0,
    accuracy_percentage FLOAT GENERATED ALWAYS AS (
        CASE WHEN phoneme_count > 0 
        THEN (correct_phonemes::FLOAT / phoneme_count * 100)
        ELSE 0 END
    ) STORED,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Voice Profiles table - for voice similarity and user identification
CREATE TABLE IF NOT EXISTS voice_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL UNIQUE,
    voice_embedding FLOAT[],
    voice_characteristics JSONB DEFAULT '{}',
    sample_count INTEGER DEFAULT 0,
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 100),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- Conversation Sessions table - for AI chatbot interactions
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES audio_sessions(id) ON DELETE CASCADE,
    conversation_topic VARCHAR(200),
    conversation_level VARCHAR(20), -- 'beginner', 'intermediate', 'advanced'
    total_exchanges INTEGER DEFAULT 0,
    user_satisfaction_score INTEGER CHECK (user_satisfaction_score >= 1 AND user_satisfaction_score <= 5),
    conversation_data JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

-- Audio Files table - metadata for uploaded/processed audio files
CREATE TABLE IF NOT EXISTS audio_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES audio_sessions(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT,
    duration_seconds FLOAT,
    format VARCHAR(10), -- 'wav', 'mp3', 'm4a', etc.
    sample_rate INTEGER,
    channels INTEGER,
    bitrate INTEGER,
    checksum VARCHAR(64),
    upload_status VARCHAR(20) DEFAULT 'processing', -- 'processing', 'completed', 'failed'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_audio_sessions_user_id ON audio_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_audio_sessions_created_at ON audio_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_speech_recognitions_session_id ON speech_recognitions(session_id);
CREATE INDEX IF NOT EXISTS idx_pronunciation_assessments_session_id ON pronunciation_assessments(session_id);
CREATE INDEX IF NOT EXISTS idx_voice_profiles_user_id ON voice_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_sessions_session_id ON conversation_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_audio_files_session_id ON audio_files(session_id);

-- Additional indexes for pronunciation analysis
CREATE INDEX IF NOT EXISTS idx_phoneme_analyses_assessment_id ON phoneme_analyses(assessment_id);
CREATE INDEX IF NOT EXISTS idx_phoneme_analyses_word_phoneme ON phoneme_analyses(word_index, phoneme_index);
CREATE INDEX IF NOT EXISTS idx_phoneme_analyses_error_type ON phoneme_analyses(error_type);
CREATE INDEX IF NOT EXISTS idx_word_analyses_assessment_id ON word_analyses(assessment_id);
CREATE INDEX IF NOT EXISTS idx_word_analyses_word_index ON word_analyses(word_index);
CREATE INDEX IF NOT EXISTS idx_word_analyses_accuracy ON word_analyses(accuracy_percentage);

-- Update triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_audio_sessions_updated_at 
    BEFORE UPDATE ON audio_sessions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial data (optional)
INSERT INTO audio_sessions (user_id, session_type, status) 
VALUES 
    ('00000000-0000-0000-0000-000000000001', 'pronunciation', 'completed'),
    ('00000000-0000-0000-0000-000000000002', 'conversation', 'active')
ON CONFLICT DO NOTHING;