"""
audio_processor_semantic.py - Semantic profanity detection using word embeddings
Detects profanity based on MEANING, not just exact word matches
Works across languages, slang, and variants!
"""

import numpy as np
import pyaudio
import time
from collections import defaultdict
import pickle
import os

# Try to import sentence transformers (best option)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✓ sentence-transformers available (best option)")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠ sentence-transformers not available")
    print("  Install: pip install sentence-transformers")


class SemanticProfanityDetector:
    """
    Detects profanity using semantic similarity with word embeddings
    
    How it works:
    1. Pre-compute embeddings for known profanity words/phrases
    2. When text comes in, compute its embedding
    3. Compare similarity - if close to profanity embedding, it's profane!
    
    Advantages:
    - Catches variations: "f*ck", "fck", "fuk", etc.
    - Multilingual: works in any language the model supports
    - Context-aware: "damn" (profane) vs "dam" (structure)
    - Slang-aware: catches new slang terms similar to known profanity
    """
    
    def __init__(self, threshold=0.7, model_name='all-MiniLM-L6-v2'):
        """
        threshold: Similarity threshold (0-1). Lower = more strict
                   0.7 = good balance
                   0.6 = very strict (may have false positives)
                   0.8 = lenient (may miss variants)
        model_name: Sentence transformer model to use
        """
        print("\n🔧 Initializing Semantic Profanity Detector...")
        
        self.threshold = threshold
        print(f"✓ Similarity threshold: {threshold}")
        
        # Load embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"Loading model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            print("✓ Model loaded")
            self.model_available = True
        else:
            print("❌ sentence-transformers not available")
            print("Falling back to keyword matching")
            self.model_available = False
        
        # Profanity database - organized by category and language
        self.profanity_db = {
            'sexual': {
                'en': ['fuck', 'fucking', 'fucked', 'sex', 'sexual content', 'dick', 
                       'cock', 'pussy', 'bitch', 'slut', 'whore', 'porn', 'xxx'],
                'es': ['puta', 'verga', 'chingar', 'coño', 'joder'],
                'fr': ['putain', 'merde', 'con', 'salope'],
                'de': ['scheiße', 'fick', 'arsch', 'fotze'],
                'hi': ['chutiya', 'madarchod', 'bhenchod'],
                'ar': ['كس', 'زب', 'خرا'],  # Arabic
            },
            'religious': {
                'en': ['damn', 'hell', 'god damn', 'jesus christ', 'christ'],
                'es': ['maldito', 'infierno', 'carajo'],
                'fr': ['merde', 'putain de dieu'],
            },
            'aggressive': {
                'en': ['shit', 'crap', 'piss', 'ass', 'asshole', 'bastard', 'kill yourself'],
                'es': ['mierda', 'cabrón', 'idiota'],
                'fr': ['connard', 'salaud'],
                'de': ['scheißer', 'arschloch'],
            },
            'slurs': {
                'en': ['nigger', 'nigga', 'faggot', 'retard', 'retarded'],
                # Add other language slurs carefully
            }
        }
        
        # Flatten all profanity words
        self.all_profanity = []
        for category in self.profanity_db.values():
            for lang_words in category.values():
                self.all_profanity.extend(lang_words)
        
        print(f"✓ Loaded {len(self.all_profanity)} profanity terms across languages")
        
        # Pre-compute embeddings for all profanity
        self.profanity_embeddings = None
        if self.model_available:
            print("Computing embeddings for profanity database...")
            self.profanity_embeddings = self.model.encode(
                self.all_profanity, 
                convert_to_tensor=False,
                show_progress_bar=False
            )
            print("✓ Profanity embeddings computed")
        
        # Stats
        self.stats = {
            'total_checks': 0,
            'profanity_detected': 0,
            'detections_by_category': defaultdict(int),
            'detections_by_similarity': [],
        }
        
        print("✓ Initialization complete\n")
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def detect_profanity_semantic(self, text):
        """
        Detect profanity using semantic similarity
        Returns (is_profane, similarity_score, matched_words)
        """
        if not self.model_available or not text:
            return False, 0.0, []
        
        self.stats['total_checks'] += 1
        
        # Compute embedding for input text
        text_embedding = self.model.encode([text], convert_to_tensor=False)[0]
        
        # Compare with all profanity embeddings
        max_similarity = 0.0
        matched_words = []
        
        for i, prof_word in enumerate(self.all_profanity):
            prof_embedding = self.profanity_embeddings[i]
            similarity = self.cosine_similarity(text_embedding, prof_embedding)
            
            if similarity > self.threshold:
                matched_words.append({
                    'word': prof_word,
                    'similarity': similarity
                })
                max_similarity = max(max_similarity, similarity)
        
        is_profane = len(matched_words) > 0
        
        if is_profane:
            self.stats['profanity_detected'] += 1
            self.stats['detections_by_similarity'].append(max_similarity)
        
        # Sort matches by similarity
        matched_words.sort(key=lambda x: x['similarity'], reverse=True)
        
        return is_profane, max_similarity, matched_words
    
    def detect_profanity_keyword(self, text):
        """
        Fallback keyword matching (if embeddings not available)
        """
        if not text:
            return False, 0.0, []
        
        text_lower = text.lower()
        matched = []
        
        for word in self.all_profanity:
            if word.lower() in text_lower:
                matched.append({'word': word, 'similarity': 1.0})
        
        return len(matched) > 0, 1.0 if matched else 0.0, matched
    
    def detect(self, text):
        """
        Main detection method - uses semantic if available, else keyword
        """
        if self.model_available:
            return self.detect_profanity_semantic(text)
        else:
            return self.detect_profanity_keyword(text)
    
    def add_custom_profanity(self, words, category='custom', language='custom'):
        """
        Add custom profanity words to the database
        words: list of words/phrases to add
        """
        if category not in self.profanity_db:
            self.profanity_db[category] = {}
        
        if language not in self.profanity_db[category]:
            self.profanity_db[category][language] = []
        
        self.profanity_db[category][language].extend(words)
        self.all_profanity.extend(words)
        
        # Recompute embeddings if model available
        if self.model_available:
            print(f"Recomputing embeddings with {len(words)} new words...")
            self.profanity_embeddings = self.model.encode(
                self.all_profanity,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            print("✓ Embeddings updated")
    
    def save_database(self, filepath='profanity_db.pkl'):
        """Save profanity database and embeddings to file"""
        data = {
            'profanity_db': self.profanity_db,
            'all_profanity': self.all_profanity,
            'profanity_embeddings': self.profanity_embeddings,
            'threshold': self.threshold
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Database saved to {filepath}")
    
    def load_database(self, filepath='profanity_db.pkl'):
        """Load profanity database from file"""
        if not os.path.exists(filepath):
            print(f"⚠ Database file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.profanity_db = data['profanity_db']
        self.all_profanity = data['all_profanity']
        self.profanity_embeddings = data['profanity_embeddings']
        self.threshold = data['threshold']
        
        print(f"✓ Database loaded from {filepath}")
        return True


class AudioProcessorSemantic:
    """
    Audio processor with semantic profanity detection
    """
    
    def __init__(self, sample_rate=44100, threshold=0.7):
        print("\n🔧 Initializing Semantic Audio Processor...")
        
        self.sample_rate = sample_rate
        self.chunk_size = 1024
        
        # Semantic profanity detector
        self.detector = SemanticProfanityDetector(threshold=threshold)
        
        # Generate beep sound
        self.beep_sound = self._generate_beep(duration_ms=500, frequency=1000)
        print("✓ Beep sound generated")
        
        # PyAudio
        self.pyaudio = pyaudio.PyAudio()
        
        print("✓ Audio processor ready\n")
    
    def _generate_beep(self, duration_ms=500, frequency=1000):
        """Generate beep tone"""
        duration_s = duration_ms / 1000.0
        t = np.linspace(0, duration_s, int(self.sample_rate * duration_s))
        beep = np.sin(2 * np.pi * frequency * t)
        
        # Fade in/out
        fade = int(0.01 * self.sample_rate)
        beep[:fade] *= np.linspace(0, 1, fade)
        beep[-fade:] *= np.linspace(1, 0, fade)
        
        return (beep * 32767).astype(np.int16)
    
    def play_beep(self):
        """Play beep"""
        stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True
        )
        try:
            stream.write(self.beep_sound.tobytes())
        finally:
            stream.stop_stream()
            stream.close()
    
    def process_text(self, text):
        """Process text for profanity"""
        is_profane, similarity, matches = self.detector.detect(text)
        
        return {
            'text': text,
            'is_profane': is_profane,
            'similarity': similarity,
            'matches': matches,
            'should_beep': is_profane
        }
    
    def test_phrase(self, text, play_sound=True):
        """Test a phrase"""
        print(f"\n{'='*70}")
        print(f"Testing: \"{text}\"")
        print(f"{'='*70}")
        
        result = self.process_text(text)
        
        if result['is_profane']:
            print(f"🚨 PROFANITY DETECTED!")
            print(f"   Similarity score: {result['similarity']:.3f}")
            print(f"   Matched terms:")
            for match in result['matches'][:3]:  # Show top 3
                print(f"     - {match['word']} (similarity: {match['similarity']:.3f})")
            
            if play_sound:
                print("🔊 Playing BEEP...")
                self.play_beep()
        else:
            print("✓ Clean - no profanity detected")
        
        return result
    
    def cleanup(self):
        """Cleanup"""
        self.pyaudio.terminate()


# ══════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("SEMANTIC PROFANITY DETECTION TEST")
    print("="*70)
    print("\nThis uses word embeddings to detect profanity by MEANING")
    print("Works across languages, slang, and variants!\n")
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("⚠️  WARNING: sentence-transformers not installed!")
        print("Install it for best results:")
        print("  pip install sentence-transformers")
        print("\nFalling back to keyword matching...\n")
        time.sleep(2)
    
    # Choose threshold
    print("Similarity threshold:")
    print("  0.6 = Very strict (more false positives)")
    print("  0.7 = Balanced (recommended)")
    print("  0.8 = Lenient (may miss some)")
    
    threshold_input = input("\nEnter threshold (default=0.7): ").strip()
    threshold = float(threshold_input) if threshold_input else 0.7
    
    processor = AudioProcessorSemantic(threshold=threshold)
    
    # Test phrases in multiple languages and variants
    test_cases = [
        # Clean
        ("Hello, how are you today?", False),
        ("I love this game!", False),
        ("The weather is nice", False),
        
        # English profanity
        ("What the fuck is this?", True),
        ("This is fucking awesome", True),
        ("Oh shit, I forgot", True),
        ("Damn it!", True),
        
        # Variants/slang (embeddings should catch these!)
        ("What the fck", True),
        ("This is fkin great", True),
        ("Oh sht", True),
        
        # Other languages
        ("Qué puta mierda", True),  # Spanish
        ("C'est de la merde", True),  # French
        ("Was für eine Scheiße", True),  # German
        
        # Context-dependent (hard cases)
        ("This is a damn good result", True),  # "damn" used as intensifier
        ("The dam broke", False),  # "dam" (structure) - should NOT detect
    ]
    
    print(f"\n{'='*70}")
    print(f"Running {len(test_cases)} test cases...")
    print(f"{'='*70}\n")
    
    time.sleep(1)
    
    correct = 0
    for text, expected_profane in test_cases:
        result = processor.test_phrase(text, play_sound=True)
        
        is_correct = result['is_profane'] == expected_profane
        if is_correct:
            correct += 1
        else:
            print(f"  ❌ MISMATCH: Expected {expected_profane}, got {result['is_profane']}")
        
        time.sleep(0.8)
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Accuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.1f}%)")
    print(f"Total checks: {processor.detector.stats['total_checks']}")
    print(f"Profanity detected: {processor.detector.stats['profanity_detected']}")
    
    if processor.detector.stats['detections_by_similarity']:
        avg_sim = np.mean(processor.detector.stats['detections_by_similarity'])
        print(f"Average similarity score: {avg_sim:.3f}")
    
    print(f"{'='*70}")
    
    # Interactive mode
    print("\nEnter interactive mode? (y/n)")
    if input().strip().lower() == 'y':
        print("\nType phrases to test (or 'quit' to exit):\n")
        while True:
            try:
                text = input("Test: ").strip()
                if text.lower() == 'quit':
                    break
                if text:
                    processor.test_phrase(text, play_sound=True)
            except KeyboardInterrupt:
                break
    
    processor.cleanup()
    print("\n✓ Test complete!")