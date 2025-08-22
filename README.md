# TTS Repository - Audio Transcription with Speaker Assignment

This repository provides audio transcription with speaker diarization or approximate speaker assignment.

## Features

- **Audio Transcription**: Uses Whisper for high-quality speech-to-text conversion
- **Speaker Diarization**: Optional precise speaker identification using pyannote.audio (requires HuggingFace token)
- **Approximate Speaker Assignment**: Simple heuristic-based speaker assignment (no token required)
- **GitHub Actions Integration**: Automated transcription workflows

## Usage

### Local Usage

```bash
# Basic transcription with approximate speakers (no HF token required)
python scripts/transcribe_dialogue.py \
  --audio audio_file.m4a \
  --language zh \
  --speaker-names "甲,乙" \
  --timestamps \
  --approx-speakers

# With diarization (requires HF token)
python scripts/transcribe_dialogue.py \
  --audio audio_file.m4a \
  --language zh \
  --speaker-names "甲,乙" \
  --timestamps \
  --hf-token YOUR_HF_TOKEN
```

### Options

- `--audio`: Path to audio file (required)
- `--language`: Language code (e.g., 'zh', 'en', 'auto')
- `--speaker-names`: Comma-separated speaker names (e.g., '甲,乙')
- `--timestamps`: Include timestamps in output
- `--approx-speakers`: Use approximate speaker assignment (no HF token needed)
- `--model-size`: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
- `--device`: Device to use (auto, cpu, cuda)
- `--output`: Output file path
- `--hf-token`: HuggingFace token for diarization

### GitHub Actions

#### Transcribe (simple) Workflow

The "Transcribe (simple)" workflow provides transcription without requiring a HuggingFace token:

1. Go to Actions tab
2. Select "Transcribe (simple)" workflow
3. Click "Run workflow"
4. Optionally specify:
   - Audio file name (default: 20250822_144051.m4a)
   - Language (default: zh)
   - Speaker names (default: 甲,乙)
5. Download the generated dialogue artifact

This workflow uses approximate speaker assignment based on timing heuristics.

## Speaker Assignment Methods

### Approximate Mode (`--approx-speakers`)

Uses simple heuristics for speaker assignment:
- Starts with the first speaker
- Switches speakers if:
  - Inter-segment pause > 0.8 seconds, OR
  - Current speaker's continuous turn > 15 seconds
- Merges consecutive segments from the same speaker (max gap: 0.3s)

### Diarization Mode (default)

Uses pyannote.audio for precise speaker identification:
- Requires HuggingFace token
- More accurate speaker assignment
- Handles overlapping speech better

## Dependencies

### Minimal (for approximate mode):
- torch (CPU)
- faster-whisper
- soundfile
- numpy
- tqdm

### Full (for diarization mode):
- All minimal dependencies
- pyannote.audio
- HuggingFace account and token

## Output Format

```
[00:15-00:18] 甲: Hello there, how are you doing today?
[00:19-00:22] 乙: I'm doing well, thank you for asking.
[00:25-00:28] 甲: That's great to hear.
```

## Testing

Run the core functionality tests:

```bash
python test_transcribe.py
```

This tests the speaker assignment and segment merging logic without requiring model downloads.