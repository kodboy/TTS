#!/usr/bin/env python3
"""
Simple test for the approximate speaker assignment functionality.
This tests the core logic without requiring model downloads.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from transcribe_dialogue import approximate_speaker_assignment, merge_consecutive_segments

# Mock segment class to simulate Whisper output
class MockSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text

def test_approximate_speaker_assignment():
    """Test the approximate speaker assignment logic."""
    print("Testing approximate speaker assignment...")
    
    # Create mock segments with different timing patterns
    segments = [
        MockSegment(0.0, 2.0, "Hello there"),           # SPK_A (start)
        MockSegment(2.1, 4.0, "How are you"),           # SPK_A (short pause)
        MockSegment(5.5, 7.0, "I'm fine thanks"),       # SPK_B (pause > 0.8s)
        MockSegment(7.2, 9.0, "That's good to hear"),   # SPK_B (short pause)
        MockSegment(10.0, 12.0, "What about you"),      # SPK_A (pause > 0.8s)
        MockSegment(12.1, 14.0, "I'm doing well"),      # SPK_A (short pause)
    ]
    
    speaker_names = ["甲", "乙"]
    assigned = approximate_speaker_assignment(segments, speaker_names)
    
    print("Results:")
    for i, segment in enumerate(assigned):
        print(f"  {segment['start']:.1f}-{segment['end']:.1f}s: {segment['speaker']}: {segment['text']}")
    
    # Verify speaker assignment logic
    expected_speakers = ["甲", "甲", "乙", "乙", "甲", "甲"]
    actual_speakers = [s['speaker'] for s in assigned]
    
    if actual_speakers == expected_speakers:
        print("✓ Speaker assignment test passed")
        return True
    else:
        print(f"✗ Speaker assignment test failed. Expected: {expected_speakers}, Got: {actual_speakers}")
        return False

def test_merge_consecutive_segments():
    """Test the segment merging logic."""
    print("\nTesting segment merging...")
    
    # Create segments that should be merged
    segments = [
        {'start': 0.0, 'end': 2.0, 'text': 'Hello', 'speaker': '甲'},
        {'start': 2.1, 'end': 4.0, 'text': 'there', 'speaker': '甲'},  # Should merge (same speaker, small gap)
        {'start': 5.0, 'end': 7.0, 'text': 'How are', 'speaker': '乙'},  # New speaker
        {'start': 7.2, 'end': 9.0, 'text': 'you doing', 'speaker': '乙'},  # Should merge
    ]
    
    merged = merge_consecutive_segments(segments, max_gap=0.3)
    
    print("Results:")
    for segment in merged:
        print(f"  {segment['start']:.1f}-{segment['end']:.1f}s: {segment['speaker']}: {segment['text']}")
    
    # Should have 2 merged segments
    if len(merged) == 2:
        print("✓ Segment merging test passed")
        return True
    else:
        print(f"✗ Segment merging test failed. Expected 2 segments, got {len(merged)}")
        return False

def test_long_speaker_turn():
    """Test speaker switching after long turns."""
    print("\nTesting long speaker turn switching...")
    
    # Create segments where one speaker talks for > 15 seconds
    segments = [
        MockSegment(0.0, 5.0, "First part"),
        MockSegment(5.1, 10.0, "Second part"),
        MockSegment(10.1, 16.0, "Third part"),  # This should trigger speaker switch (>15s total)
    ]
    
    speaker_names = ["甲", "乙"]
    assigned = approximate_speaker_assignment(segments, speaker_names)
    
    print("Results:")
    for segment in assigned:
        print(f"  {segment['start']:.1f}-{segment['end']:.1f}s: {segment['speaker']}: {segment['text']}")
    
    # The third segment should be assigned to a different speaker
    if assigned[0]['speaker'] == assigned[1]['speaker'] and assigned[2]['speaker'] != assigned[0]['speaker']:
        print("✓ Long speaker turn test passed")
        return True
    else:
        print("✗ Long speaker turn test failed")
        return False

def main():
    """Run all tests."""
    print("Running transcribe_dialogue.py functionality tests\n")
    
    tests = [
        test_approximate_speaker_assignment,
        test_merge_consecutive_segments,
        test_long_speaker_turn,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All tests passed! The core functionality is working correctly.")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())