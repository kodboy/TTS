#!/usr/bin/env python3
"""
Test script to validate the script behavior when diarization is not available.
This simulates the fallback behavior.
"""

import sys
import os
import tempfile
from pathlib import Path

# Mock the transcribe_dialogue imports to test fallback behavior
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

def test_fallback_behavior():
    """Test that the script falls back gracefully when diarization is not available."""
    print("Testing fallback behavior when diarization is unavailable...")
    
    # Import the script components
    from transcribe_dialogue import approximate_speaker_assignment, merge_consecutive_segments
    
    # Create mock segments as if they came from Whisper
    class MockSegment:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text
    
    segments = [
        MockSegment(0.0, 3.0, "Hello, how are you today?"),
        MockSegment(3.5, 6.0, "I'm doing well, thank you."),
        MockSegment(7.0, 10.0, "That's great to hear."),
        MockSegment(10.5, 14.0, "Do you have any plans for the weekend?"),
        MockSegment(15.0, 18.0, "Yes, I'm planning to visit some friends."),
    ]
    
    speaker_names = ["甲", "乙"]
    
    # Test approximate assignment
    assigned = approximate_speaker_assignment(segments, speaker_names)
    print(f"Assigned {len(assigned)} segments to speakers")
    
    # Test merging
    merged = merge_consecutive_segments(assigned)
    print(f"Merged into {len(merged)} final segments")
    
    # Display results
    for segment in merged:
        start_min = int(segment['start'] // 60)
        start_sec = int(segment['start'] % 60)
        end_min = int(segment['end'] // 60)
        end_sec = int(segment['end'] % 60)
        print(f"  [{start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d}] {segment['speaker']}: {segment['text']}")
    
    # Verify we have a reasonable distribution of speakers
    speaker_counts = {}
    for segment in merged:
        speaker = segment['speaker']
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    
    print(f"Speaker distribution: {speaker_counts}")
    
    # Should have both speakers represented
    if len(speaker_counts) >= 2:
        print("✓ Fallback behavior test passed - multiple speakers assigned")
        return True
    else:
        print("✗ Fallback behavior test failed - only one speaker assigned")
        return False

def test_import_behavior():
    """Test that pyannote imports are handled gracefully."""
    print("\nTesting optional import behavior...")
    
    # Import the main script to check DIARIZATION_AVAILABLE flag
    import transcribe_dialogue
    
    print(f"Diarization available: {transcribe_dialogue.DIARIZATION_AVAILABLE}")
    
    if transcribe_dialogue.DIARIZATION_AVAILABLE:
        print("✓ pyannote.audio is available")
    else:
        print("✓ pyannote.audio is not available - fallback mode will be used")
    
    return True

def main():
    """Run fallback behavior tests."""
    print("Testing transcribe_dialogue.py fallback behavior\n")
    
    tests = [
        test_fallback_behavior,
        test_import_behavior,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
    
    print(f"\nFallback Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All fallback tests passed! The script will handle missing dependencies gracefully.")
        return 0
    else:
        print("✗ Some fallback tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())