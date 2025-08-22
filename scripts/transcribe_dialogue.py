#!/usr/bin/env python3
"""
Transcribe audio with speaker diarization or approximate speaker assignment.

This script supports two modes:
1. Diarization mode: Uses pyannote.audio for precise speaker diarization (requires HuggingFace token)
2. Approximate mode: Uses simple heuristics for speaker assignment (no token required)
"""

import argparse
import os
import sys
from pathlib import Path
import warnings

# Core dependencies (always available)
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from tqdm import tqdm

# Optional diarization dependencies
DIARIZATION_AVAILABLE = False
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    import torch
    DIARIZATION_AVAILABLE = True
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with speaker diarization or approximate speaker assignment"
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Language code for transcription (e.g., 'zh', 'en', 'auto')"
    )
    parser.add_argument(
        "--speaker-names",
        default="Speaker A,Speaker B",
        help="Comma-separated list of speaker names (e.g., '甲,乙')"
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in output"
    )
    parser.add_argument(
        "--approx-speakers",
        action="store_true",
        help="Use approximate speaker assignment instead of diarization (no HuggingFace token required)"
    )
    parser.add_argument(
        "--model-size",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: input_filename_dialogue.txt)"
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token for diarization (can also be set via HUGGINGFACE_TOKEN env var)"
    )
    
    return parser.parse_args()


def get_device(device_arg):
    """Determine the best device to use."""
    if device_arg == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    return device_arg


def transcribe_audio(audio_path, model_size="medium", language="auto", device="auto"):
    """Transcribe audio using Whisper."""
    print(f"Loading Whisper model: {model_size}")
    
    # Try to load model, first allowing downloads, then local-only if that fails
    try:
        model = WhisperModel(model_size, device=get_device(device))
    except Exception as e:
        print(f"Failed to load model with downloads enabled: {e}")
        print("Trying to load from local cache only...")
        try:
            model = WhisperModel(model_size, device=get_device(device), local_files_only=True)
        except Exception as e2:
            print(f"Failed to load model from local cache: {e2}")
            print("Model not available locally. In GitHub Actions, this will be downloaded automatically.")
            raise e2
    
    print(f"Transcribing audio: {audio_path}")
    if language == "auto":
        segments, info = model.transcribe(audio_path, beam_size=5)
        language = info.language
        print(f"Detected language: {language}")
    else:
        segments, info = model.transcribe(audio_path, language=language, beam_size=5)
    
    # Convert to list for easier processing
    segments = list(segments)
    print(f"Found {len(segments)} segments")
    
    return segments, language


def approximate_speaker_assignment(segments, speaker_names):
    """
    Assign speakers using simple heuristics:
    - Start with first speaker
    - Switch speakers if inter-segment pause > 0.8s OR current speaker's turn > 15s
    - Otherwise keep same speaker
    """
    if not segments:
        return []
    
    assigned_segments = []
    current_speaker_idx = 0
    current_speaker_start_time = segments[0].start
    
    for i, segment in enumerate(segments):
        # Check if we should switch speakers
        should_switch = False
        
        if i > 0:
            # Check inter-segment pause
            prev_segment = segments[i-1]
            pause_duration = segment.start - prev_segment.end
            
            # Check current speaker's running turn length
            turn_duration = segment.end - current_speaker_start_time
            
            if pause_duration > 0.8 or turn_duration > 15.0:
                should_switch = True
        
        if should_switch:
            current_speaker_idx = (current_speaker_idx + 1) % len(speaker_names)
            current_speaker_start_time = segment.start
        
        # Create segment with speaker assignment
        segment_data = {
            'start': segment.start,
            'end': segment.end,
            'text': segment.text.strip(),
            'speaker': speaker_names[current_speaker_idx]
        }
        assigned_segments.append(segment_data)
    
    return assigned_segments


def diarize_audio(audio_path, hf_token):
    """Perform speaker diarization using pyannote.audio."""
    if not DIARIZATION_AVAILABLE:
        raise ImportError("pyannote.audio not available. Use --approx-speakers instead.")
    
    if not hf_token:
        raise ValueError("HuggingFace token required for diarization")
    
    print("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    print("Performing speaker diarization...")
    diarization = pipeline(audio_path)
    
    return diarization


def assign_speakers_with_diarization(segments, diarization, speaker_names):
    """Assign speakers based on diarization results."""
    assigned_segments = []
    
    # Create speaker mapping
    unique_speakers = list(set(turn.speaker for turn in diarization.itertracks(yield_label=True)))
    speaker_mapping = {spk: speaker_names[i % len(speaker_names)] for i, spk in enumerate(unique_speakers)}
    
    for segment in segments:
        # Find the dominant speaker for this segment
        segment_interval = Segment(segment.start, segment.end)
        overlapping_speakers = {}
        
        for turn in diarization.itertracks(yield_label=True):
            turn_segment, _, speaker = turn
            overlap = segment_interval.intersect(turn_segment)
            if overlap:
                overlap_duration = overlap.duration
                if speaker not in overlapping_speakers:
                    overlapping_speakers[speaker] = 0
                overlapping_speakers[speaker] += overlap_duration
        
        # Assign to speaker with most overlap
        if overlapping_speakers:
            dominant_speaker = max(overlapping_speakers.items(), key=lambda x: x[1])[0]
            assigned_speaker = speaker_mapping[dominant_speaker]
        else:
            assigned_speaker = speaker_names[0]  # Fallback
        
        segment_data = {
            'start': segment.start,
            'end': segment.end,
            'text': segment.text.strip(),
            'speaker': assigned_speaker
        }
        assigned_segments.append(segment_data)
    
    return assigned_segments


def merge_consecutive_segments(segments, max_gap=0.3):
    """Merge consecutive segments from the same speaker."""
    if not segments:
        return []
    
    merged = []
    current = segments[0].copy()
    
    for segment in segments[1:]:
        # Check if same speaker and small gap
        if (segment['speaker'] == current['speaker'] and 
            segment['start'] - current['end'] <= max_gap):
            # Merge segments
            current['end'] = segment['end']
            current['text'] += ' ' + segment['text']
        else:
            # Start new segment
            merged.append(current)
            current = segment.copy()
    
    merged.append(current)
    return merged


def format_timestamp(seconds):
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def format_output(segments, include_timestamps=False):
    """Format segments for output."""
    lines = []
    
    for segment in segments:
        if include_timestamps:
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            line = f"[{start_time}-{end_time}] {segment['speaker']}: {segment['text']}"
        else:
            line = f"{segment['speaker']}: {segment['text']}"
        lines.append(line)
    
    return '\n'.join(lines)


def main():
    args = parse_args()
    
    # Validate audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Parse speaker names
    speaker_names = [name.strip() for name in args.speaker_names.split(',')]
    if len(speaker_names) < 2:
        speaker_names = ["Speaker A", "Speaker B"]
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = audio_path.parent / f"{audio_path.stem}_dialogue.txt"
    
    try:
        # Step 1: Transcribe audio
        segments, detected_language = transcribe_audio(
            str(audio_path), 
            args.model_size, 
            args.language, 
            args.device
        )
        
        if not segments:
            print("No speech detected in audio file")
            sys.exit(1)
        
        # Step 2: Assign speakers
        if args.approx_speakers:
            print("Using approximate speaker assignment...")
            assigned_segments = approximate_speaker_assignment(segments, speaker_names)
        else:
            # Try diarization if available and token provided
            hf_token = args.hf_token or os.getenv('HUGGINGFACE_TOKEN')
            
            if DIARIZATION_AVAILABLE and hf_token:
                print("Using diarization for speaker assignment...")
                try:
                    diarization = diarize_audio(str(audio_path), hf_token)
                    assigned_segments = assign_speakers_with_diarization(segments, diarization, speaker_names)
                except Exception as e:
                    print(f"Diarization failed: {e}")
                    print("Falling back to approximate speaker assignment...")
                    assigned_segments = approximate_speaker_assignment(segments, speaker_names)
            else:
                print("Diarization not available (missing token or pyannote.audio), using approximate assignment...")
                assigned_segments = approximate_speaker_assignment(segments, speaker_names)
        
        # Step 3: Merge consecutive segments from same speaker
        print("Merging consecutive segments...")
        merged_segments = merge_consecutive_segments(assigned_segments)
        
        # Step 4: Format and save output
        output_text = format_output(merged_segments, args.timestamps)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        print(f"Transcription saved to: {output_path}")
        print(f"Total segments: {len(merged_segments)}")
        print(f"Speakers: {', '.join(speaker_names[:2])}")
        
        # Preview first few lines
        lines = output_text.split('\n')
        print("\nPreview:")
        for line in lines[:5]:
            print(f"  {line}")
        if len(lines) > 5:
            print(f"  ... and {len(lines) - 5} more lines")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()