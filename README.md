# Speech Transcription Pipeline

Automatically transcribe audio files with speaker diarization using Whisper and pyannote.audio.

## Features

- **Speech Recognition**: Uses faster-whisper for high-quality ASR with Voice Activity Detection (VAD)
- **Speaker Diarization**: Uses pyannote.audio 3.x for identifying different speakers
- **Smart Alignment**: Aligns transcription segments with speaker labels using overlap voting
- **Segment Merging**: Merges consecutive segments from the same speaker within 0.3s gaps
- **Flexible Output**: Plain text dialogue with optional timestamps
- **Chinese Support**: Default speaker names 甲 (Speaker A) and 乙 (Speaker B) for Chinese content

## Quick Start

### 1. Set up Hugging Face Token

The pipeline requires a Hugging Face access token for speaker diarization:

1. Get a token from [Hugging Face](https://huggingface.co/settings/tokens) (read permissions)
2. Add it to GitHub repository secrets:
   - Go to **Settings** → **Secrets and variables** → **Actions**
   - Create new secret: `HUGGINGFACE_TOKEN`

### 2. Run the Pipeline

#### Via GitHub Actions (Recommended)
1. Go to the **Actions** tab
2. Select "Speech Transcription Pipeline"
3. Click "Run workflow"
4. Download the transcript from workflow artifacts

#### Via Command Line
```bash
# Install dependencies
pip install -r requirements.txt

# Run transcription
python scripts/transcribe_dialogue.py \
  --audio 20250822_144051.m4a \
  --language zh \
  --speaker-names "甲,乙" \
  --timestamps \
  --hf-token YOUR_HF_TOKEN
```

## Command Line Options

```
--audio         Audio file path (required)
--num-speakers  Number of speakers (default: 2)
--language      Language code (e.g., 'zh', 'en')
--model         Whisper model (default: 'large-v3')
--device        Device: auto/cpu/cuda (default: auto)
--compute-type  Compute type for faster-whisper
--beam-size     Beam size for decoding (default: 5)
--speaker-names Speaker names, comma-separated (default: '甲,乙')
--timestamps    Include timestamps in output
--output        Output file path
--hf-token      Hugging Face access token
```

## Output Format

### With Timestamps
```
[00:15-00:18] 甲: 你好，今天天气怎么样？
[00:19-00:22] 乙: 天气很好，阳光明媚。
[00:23-00:26] 甲: 那我们出去走走吧。
```

### Without Timestamps
```
甲: 你好，今天天气怎么样？
乙: 天气很好，阳光明媚。
甲: 那我们出去走走吧。
```

## Dependencies

- **faster-whisper**: Efficient Whisper implementation
- **pyannote.audio**: Speaker diarization toolkit
- **torch**: Deep learning framework
- **transformers**: Hugging Face transformers library

## Workflow Triggers

- **Manual**: Run on-demand from Actions tab
- **Automatic**: Triggers on pull requests to main/master branches

## Architecture

1. **ASR**: faster-whisper transcribes audio with VAD filtering
2. **Diarization**: pyannote.audio identifies speaker segments
3. **Alignment**: Overlap voting assigns speakers to transcription segments
4. **Merging**: Consecutive segments from same speaker are merged
5. **Formatting**: Output as readable dialogue with optional timestamps

## License

See repository license for terms of use.