"""
test_with_audio.py - Test profanity detection with REAL audio
Supports: WAV files, MP3 files, or live microphone input
"""

import numpy as np
import pyaudio
import wave
import time
import os

# Audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠ pydub not available - install: pip install pydub")

# Speech recognition
try:
    import speech_recognition as sr
    SPEECH_REC_AVAILABLE = True
except ImportError:
    SPEECH_REC_AVAILABLE = False
    print("⚠ speech_recognition not available - install: pip install SpeechRecognition")

# Import our advanced detector
from audio_processor import AdvancedProfanityDetector


class AudioTester:
    """
    Test profanity detection with real audio
    """
    
    def __init__(self, detector):
        self.detector = detector
        self.recognizer = sr.Recognizer() if SPEECH_REC_AVAILABLE else None
        self.pyaudio = pyaudio.PyAudio()
        
        # Beep sound for playback
        self.beep_sound = self._generate_beep()
    
    def _generate_beep(self, duration_ms=500, frequency=1000, sample_rate=44100):
        """Generate beep tone"""
        duration_s = duration_ms / 1000.0
        t = np.linspace(0, duration_s, int(sample_rate * duration_s))
        beep = np.sin(2 * np.pi * frequency * t)
        
        # Fade in/out
        fade = int(0.01 * sample_rate)
        beep[:fade] *= np.linspace(0, 1, fade)
        beep[-fade:] *= np.linspace(1, 0, fade)
        
        return (beep * 32767).astype(np.int16)
    
    def play_beep(self):
        """Play beep sound"""
        stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            output=True
        )
        try:
            stream.write(self.beep_sound.tobytes())
        finally:
            stream.stop_stream()
            stream.close()
    
    # ══════════════════════════════════════════════════════════════════════
    # METHOD 1: Test with WAV file
    # ══════════════════════════════════════════════════════════════════════
    
    def test_wav_file(self, filepath):
        """Test with a WAV audio file"""
        if not SPEECH_REC_AVAILABLE:
            print("❌ speech_recognition not installed!")
            return
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return
        
        print(f"\n{'='*70}")
        print(f"Testing WAV file: {filepath}")
        print(f"{'='*70}")
        
        try:
            # Load audio file
            with sr.AudioFile(filepath) as source:
                print("🎵 Loading audio...")
                audio_data = self.recognizer.record(source)
            
            # Transcribe
            print("🔄 Transcribing (this may take a few seconds)...")
            text = self.recognizer.recognize_google(audio_data)
            
            print(f"📝 Transcribed text: \"{text}\"")
            print()
            
            # Detect profanity
            is_profane, confidence, matches = self.detector.detect(text)
            
            if is_profane:
                print(f"🚨 PROFANITY DETECTED!")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Matched terms: {[m['word'] for m in matches[:3]]}")
                print("🔊 Playing BEEP...")
                self.play_beep()
            else:
                print("✓ Clean - no profanity detected")
            
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # ══════════════════════════════════════════════════════════════════════
    # METHOD 2: Test with MP3 file (convert to WAV first)
    # ══════════════════════════════════════════════════════════════════════
    
    def test_mp3_file(self, filepath):
        """Test with an MP3 file (converts to WAV first)"""
        if not PYDUB_AVAILABLE:
            print("❌ pydub not installed! Install: pip install pydub")
            return
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return
        
        print(f"\n{'='*70}")
        print(f"Testing MP3 file: {filepath}")
        print(f"{'='*70}")
        
        try:
            # Convert MP3 to WAV
            print("🔄 Converting MP3 to WAV...")
            audio = AudioSegment.from_mp3(filepath)
            
            # Export to temporary WAV
            temp_wav = "temp_audio.wav"
            audio.export(temp_wav, format="wav")
            
            # Test the WAV
            self.test_wav_file(temp_wav)
            
            # Cleanup
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            
        except Exception as e:
            print(f"❌ Error processing MP3: {e}")
    
    # ══════════════════════════════════════════════════════════════════════
    # METHOD 3: Test with live microphone
    # ══════════════════════════════════════════════════════════════════════
    
    def test_microphone(self, duration=5):
        """Record from microphone and test"""
        if not SPEECH_REC_AVAILABLE:
            print("❌ speech_recognition not installed!")
            return
        
        print(f"\n{'='*70}")
        print(f"LIVE MICROPHONE TEST")
        print(f"{'='*70}")
        print(f"Recording for {duration} seconds...")
        print("🎤 Start speaking NOW!\n")
        
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                print("🔇 Adjusting for ambient noise... (be quiet)")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                print(f"🎤 Recording {duration} seconds... SPEAK NOW!")
                audio_data = self.recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
            
            print("✓ Recording complete!")
            print("🔄 Transcribing...")
            
            # Transcribe
            text = self.recognizer.recognize_google(audio_data)
            print(f"\n📝 You said: \"{text}\"")
            print()
            
            # Detect profanity
            is_profane, confidence, matches = self.detector.detect(text)
            
            if is_profane:
                print(f"🚨 PROFANITY DETECTED!")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Matched terms: {[m['word'] for m in matches[:3]]}")
                print("🔊 Playing BEEP...")
                self.play_beep()
            else:
                print("✓ Clean - no profanity detected")
            
            return text, is_profane, confidence
            
        except sr.WaitTimeoutError:
            print("❌ No speech detected in time limit")
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        return None, False, 0.0
    
    # ══════════════════════════════════════════════════════════════════════
    # METHOD 4: Continuous microphone monitoring (real-time streaming test)
    # ══════════════════════════════════════════════════════════════════════
    
    def test_continuous_microphone(self):
        """
        Continuous monitoring - closest to real streaming
        Press Ctrl+C to stop
        """
        if not SPEECH_REC_AVAILABLE:
            print("❌ speech_recognition not installed!")
            return
        
        print(f"\n{'='*70}")
        print(f"CONTINUOUS MONITORING MODE (Real-time streaming simulation)")
        print(f"{'='*70}")
        print("This will continuously listen and detect profanity")
        print("Press Ctrl+C to stop\n")
        
        time.sleep(2)
        
        with sr.Microphone() as source:
            print("🔇 Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("\n🎤 Monitoring started! Speak naturally...\n")
            
            try:
                while True:
                    try:
                        # Listen for speech (2 second phrases)
                        audio_data = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
                        
                        # Transcribe in background
                        try:
                            text = self.recognizer.recognize_google(audio_data)
                            print(f"[{time.strftime('%H:%M:%S')}] Heard: \"{text}\"")
                            
                            # Detect
                            is_profane, confidence, matches = self.detector.detect(text)
                            
                            if is_profane:
                                print(f"  🚨 PROFANE (conf: {confidence:.2f}) - {[m['word'] for m in matches[:2]]}")
                                self.play_beep()
                            else:
                                print(f"  ✓ Clean")
                            print()
                            
                        except sr.UnknownValueError:
                            pass  # Couldn't understand, skip
                        except sr.RequestError as e:
                            print(f"  ❌ API error: {e}")
                    
                    except sr.WaitTimeoutError:
                        pass  # No speech, continue listening
                    
            except KeyboardInterrupt:
                print("\n\n✓ Monitoring stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        self.pyaudio.terminate()


# ══════════════════════════════════════════════════════════════════════════
# INTERACTIVE MENU
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("AUDIO PROFANITY TESTING")
    print("="*70)
    
    if not SPEECH_REC_AVAILABLE:
        print("\n❌ speech_recognition not installed!")
        print("Install it: pip install SpeechRecognition")
        return
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = AdvancedProfanityDetector(recall_mode='high', allow_mild_profanity=False)
    tester = AudioTester(detector)
    
    while True:
        print("\n" + "="*70)
        print("TEST OPTIONS")
        print("="*70)
        print("1. Test with WAV file")
        print("2. Test with MP3 file") 
        print("3. Record from microphone (5 seconds)")
        print("4. Continuous monitoring (real-time simulation)")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            filepath = input("Enter WAV file path: ").strip()
            tester.test_wav_file(filepath)
        
        elif choice == '2':
            if not PYDUB_AVAILABLE:
                print("❌ pydub not installed! Install: pip install pydub")
            else:
                filepath = input("Enter MP3 file path: ").strip()
                tester.test_mp3_file(filepath)
        
        elif choice == '3':
            duration = input("Recording duration in seconds (default=5): ").strip()
            duration = int(duration) if duration else 5
            tester.test_microphone(duration)
        
        elif choice == '4':
            print("\nStarting continuous monitoring...")
            print("This simulates real streaming - speak naturally")
            input("Press Enter when ready...")
            tester.test_continuous_microphone()
        
        elif choice == '5':
            print("\nExiting...")
            break
        
        else:
            print("Invalid option")
    
    tester.cleanup()
    print("\n✓ Tests complete!")


# ══════════════════════════════════════════════════════════════════════════
# QUICK TEST SCRIPT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Check dependencies
    if not SPEECH_REC_AVAILABLE:
        print("\n" + "="*70)
        print("MISSING DEPENDENCY: speech_recognition")
        print("="*70)
        print("\nInstall it:")
        print("  pip install SpeechRecognition")
        print("\nOptional (for MP3 support):")
        print("  pip install pydub")
        print("\nThen run this script again!")
        print("="*70)
    else:
        main()