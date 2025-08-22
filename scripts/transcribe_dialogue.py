#!/usr/bin/env python3
"""
Speech Transcription Pipeline with Speaker Diarization

This script converts audio files into speaker-diarized transcripts using:
- faster-whisper for automatic speech recognition (ASR) with Voice Activity Detection (VAD)
- pyannote.audio for speaker diarization

Features:
- Aligns diarization segments to ASR segments by overlap voting
- Merges consecutive segments from the same speaker within a configurable gap
- Outputs plain text dialogue with optional timestamps
- Configurable speaker names (default: 甲, 乙 for Chinese speakers)

Requirements:
- Hugging Face access token for pyannote.audio models
- Set HUGGINGFACE_TOKEN secret in GitHub repository settings under:
  Settings -> Secrets and variables -> Actions
  
Usage:
  python scripts/transcribe_dialogue.py --audio audio_file.m4a --language zh --timestamps
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings

def setup_environment():
    """Set up environment and suppress warnings"""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import faster_whisper
    except ImportError:
        missing_deps.append("faster-whisper")
    
    try:
        import pyannote.audio
    except ImportError:
        missing_deps.append("pyannote.audio")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    if missing_deps:
        print(f"Error: Missing required dependencies: {', '.join(missing_deps)}")
        print("Please install them using: pip install -r requirements.txt")
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert audio to speaker-diarized transcript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription with Chinese language
  python scripts/transcribe_dialogue.py --audio audio.m4a --language zh
  
  # With timestamps and custom speaker names
  python scripts/transcribe_dialogue.py --audio audio.m4a --language zh --timestamps --speaker-names "Alice,Bob"
  
  # Specify Hugging Face token explicitly
  python scripts/transcribe_dialogue.py --audio audio.m4a --hf-token your_token_here

Note: Set HUGGINGFACE_TOKEN in GitHub repository secrets for automated workflows.
        """
    )
    
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--num-speakers", type=int, default=2, help="Number of speakers (default: 2)")
    parser.add_argument("--language", help="Language code (e.g., 'zh', 'en')")
    parser.add_argument("--model", default="large-v3", help="Whisper model size (default: large-v3)")
    parser.add_argument("--device", default="auto", help="Device: auto/cpu/cuda (default: auto)")
    parser.add_argument("--compute-type", help="Compute type for faster-whisper")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding (default: 5)")
    parser.add_argument("--speaker-names", default="甲,乙", help="Speaker names separated by comma (default: 甲,乙)")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps in output")
    parser.add_argument("--output", help="Output file path (default: next to audio file)")
    parser.add_argument("--hf-token", help="Hugging Face access token (or set HUGGINGFACE_TOKEN env var)")
    
    return parser.parse_args()

class AudioTranscriber:
    """Main class for audio transcription and diarization"""
    
    def __init__(self, args):
        self.args = args
        self.setup_device()
        self.setup_hf_token()
        
    def setup_device(self):
        """Set up compute device"""
        if self.args.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = self.args.device
        
        print(f"Using device: {self.device}")
    
    def setup_hf_token(self):
        """Set up Hugging Face token"""
        self.hf_token = self.args.hf_token or os.getenv('HUGGINGFACE_TOKEN')
        
        if not self.hf_token:
            print("Error: Hugging Face access token is required for speaker diarization.")
            print("Please either:")
            print("1. Pass --hf-token argument")
            print("2. Set HUGGINGFACE_TOKEN environment variable")
            print("3. For GitHub Actions: Set HUGGINGFACE_TOKEN secret in repository settings")
            sys.exit(1)
    
    def transcribe_audio(self):
        """Transcribe audio using faster-whisper"""
        print(f"Transcribing audio: {self.args.audio}")
        
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("Error: faster-whisper not available. Please install: pip install faster-whisper")
            sys.exit(1)
        
        # Initialize Whisper model
        model_kwargs = {
            "model_size_or_path": self.args.model,
            "device": self.device,
        }
        
        if self.args.compute_type:
            model_kwargs["compute_type"] = self.args.compute_type
        
        model = WhisperModel(**model_kwargs)
        
        # Transcribe with VAD
        transcribe_kwargs = {
            "beam_size": self.args.beam_size,
            "vad_filter": True,
            "vad_parameters": dict(min_silence_duration_ms=500)
        }
        
        if self.args.language:
            transcribe_kwargs["language"] = self.args.language
        
        segments, info = model.transcribe(self.args.audio, **transcribe_kwargs)
        
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Convert to list for processing
        transcription_segments = []
        for segment in segments:
            transcription_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
        
        print(f"Found {len(transcription_segments)} transcription segments")
        return transcription_segments
    
    def diarize_audio(self):
        """Perform speaker diarization using pyannote.audio"""
        print("Performing speaker diarization...")
        
        try:
            from pyannote.audio import Pipeline
            import torch
        except ImportError:
            print("Error: pyannote.audio not available. Please install: pip install pyannote.audio")
            sys.exit(1)
        
        # Initialize diarization pipeline
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            
            # Set device
            if hasattr(pipeline, 'to'):
                pipeline.to(torch.device(self.device))
        except Exception as e:
            print(f"Error loading diarization model: {e}")
            print("Make sure your Hugging Face token has access to pyannote.audio models")
            sys.exit(1)
        
        # Perform diarization
        try:
            diarization = pipeline(self.args.audio, num_speakers=self.args.num_speakers)
        except Exception as e:
            print(f"Error during diarization: {e}")
            sys.exit(1)
        
        # Convert to segments list
        diarization_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        print(f"Found {len(diarization_segments)} speaker segments")
        return diarization_segments
    
    def align_segments(self, transcription_segments: List[Dict], diarization_segments: List[Dict]):
        """Align transcription segments with speaker labels using overlap voting"""
        print("Aligning transcription with speaker labels...")
        
        aligned_segments = []
        
        for trans_seg in transcription_segments:
            trans_start, trans_end = trans_seg['start'], trans_seg['end']
            trans_duration = trans_end - trans_start
            
            # Find overlapping diarization segments
            speaker_votes = {}
            
            for diar_seg in diarization_segments:
                diar_start, diar_end = diar_seg['start'], diar_seg['end']
                
                # Calculate overlap
                overlap_start = max(trans_start, diar_start)
                overlap_end = min(trans_end, diar_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > 0:
                    speaker = diar_seg['speaker']
                    if speaker not in speaker_votes:
                        speaker_votes[speaker] = 0
                    speaker_votes[speaker] += overlap_duration
            
            # Assign speaker with most overlap
            if speaker_votes:
                assigned_speaker = max(speaker_votes, key=speaker_votes.get)
            else:
                assigned_speaker = "SPEAKER_00"  # Default fallback
            
            aligned_segments.append({
                'start': trans_start,
                'end': trans_end,
                'text': trans_seg['text'],
                'speaker': assigned_speaker
            })
        
        print(f"Aligned {len(aligned_segments)} segments")
        return aligned_segments
    
    def merge_consecutive_segments(self, segments: List[Dict], gap_threshold: float = 0.3):
        """Merge consecutive segments from the same speaker within gap threshold"""
        print(f"Merging consecutive segments (gap threshold: {gap_threshold}s)...")
        
        if not segments:
            return segments
        
        merged_segments = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            gap = next_segment['start'] - current_segment['end']
            same_speaker = current_segment['speaker'] == next_segment['speaker']
            
            if same_speaker and gap <= gap_threshold:
                # Merge segments
                current_segment['end'] = next_segment['end']
                current_segment['text'] += ' ' + next_segment['text']
            else:
                # Start new segment
                merged_segments.append(current_segment)
                current_segment = next_segment.copy()
        
        # Add the last segment
        merged_segments.append(current_segment)
        
        print(f"Merged into {len(merged_segments)} segments (from {len(segments)})")
        return merged_segments
    
    def format_output(self, segments: List[Dict]):
        """Format segments into dialogue output"""
        print("Formatting output...")
        
        speaker_names = self.args.speaker_names.split(',')
        speaker_mapping = {}
        
        # Map speaker IDs to names
        unique_speakers = sorted(set(seg['speaker'] for seg in segments))
        for i, speaker_id in enumerate(unique_speakers):
            if i < len(speaker_names):
                speaker_mapping[speaker_id] = speaker_names[i].strip()
            else:
                speaker_mapping[speaker_id] = f"Speaker_{i+1}"
        
        # Format dialogue
        lines = []
        for segment in segments:
            speaker_name = speaker_mapping[segment['speaker']]
            text = segment['text'].strip()
            
            if self.args.timestamps:
                start_time = self.format_timestamp(segment['start'])
                end_time = self.format_timestamp(segment['end'])
                line = f"[{start_time}-{end_time}] {speaker_name}: {text}"
            else:
                line = f"{speaker_name}: {text}"
            
            lines.append(line)
        
        return '\n'.join(lines)
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def save_output(self, content: str):
        """Save formatted dialogue to file"""
        if self.args.output:
            output_path = Path(self.args.output)
        else:
            audio_path = Path(self.args.audio)
            output_path = audio_path.parent / f"{audio_path.stem}_dialogue.txt"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Dialogue saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"Error saving output: {e}")
            sys.exit(1)
    
    def run(self):
        """Run the complete transcription pipeline"""
        print("Starting speech transcription pipeline...")
        
        # Check if audio file exists
        if not os.path.exists(self.args.audio):
            print(f"Error: Audio file not found: {self.args.audio}")
            sys.exit(1)
        
        try:
            # Step 1: Transcribe audio
            transcription_segments = self.transcribe_audio()
            
            # Step 2: Diarize audio
            diarization_segments = self.diarize_audio()
            
            # Step 3: Align segments
            aligned_segments = self.align_segments(transcription_segments, diarization_segments)
            
            # Step 4: Merge consecutive segments
            merged_segments = self.merge_consecutive_segments(aligned_segments)
            
            # Step 5: Format output
            dialogue_text = self.format_output(merged_segments)
            
            # Step 6: Save output
            output_file = self.save_output(dialogue_text)
            
            print("\nTranscription completed successfully!")
            print(f"Output saved to: {output_file}")
            
            # Also print to stdout for verification
            print("\n" + "="*50)
            print("DIALOGUE PREVIEW:")
            print("="*50)
            print(dialogue_text)
            
        except KeyboardInterrupt:
            print("\nTranscription interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"Error during transcription: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def main():
    """Main entry point"""
    setup_environment()
    check_dependencies()
    
    args = parse_arguments()
    transcriber = AudioTranscriber(args)
    transcriber.run()

if __name__ == "__main__":
    main()