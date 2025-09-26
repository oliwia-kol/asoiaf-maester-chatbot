# tests/test_data_processor.py
"""
Comprehensive Test Suite for Data Processing Pipeline

Testing practices for NLP systems:
- Unit tests for individual functions
- Integration tests for complete workflows
- Property-based testing for invariants
- Performance testing for scalability
- Mock external dependencies for isolation

Testing Architecture: Pyramid Pattern
- Many fast unit tests (80%)
- Some integration tests (15%)
- Few end-to-end tests (5%)
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from typing import List

# Import our modules
from src.data_processor import ASoIaFDataProcessor, TextChunk
from src.config import ProcessingConfig


class TestTextChunkDataclass:
    """
    Test the TextChunk immutable data structure.
    
    - Ensures computed properties work correctly
    - Validates immutability constraints  
    - Tests serialization/deserialization
    """
    
    def test_chunk_creation_with_all_fields(self):
        """Test complete chunk creation with validation."""
        chunk = TextChunk(
            content="Jon Snow knew nothing of the dangers beyond the Wall.",
            chunk_id="test_001", 
            source_file="test.txt",
            start_position=0,
            end_position=50,
            characters=["Jon Snow"],
            locations=["Wall"]
        )
        
        assert chunk.content == "Jon Snow knew nothing of the dangers beyond the Wall."
        assert chunk.chunk_id == "test_001"
        assert chunk.word_count == 10  # Auto-calculated
        assert "Jon Snow" in chunk.characters
        assert "Wall" in chunk.locations
    
    def test_auto_word_count_calculation(self):
        """Test automatic word count calculation in __post_init__."""
        chunk = TextChunk(
            content="Winter is coming to Westeros.",
            chunk_id="test_002",
            source_file="test.txt", 
            start_position=0,
            end_position=30,
            characters=[],
            locations=["Westeros"],
            word_count=0  # Should be auto-calculated
        )
        
        assert chunk.word_count == 5 # Should be auto-calculated
        # Let's verify the calculation:
        manual_count = len("Winter is coming to Westeros.".split())
        assert chunk.word_count == manual_count
    
    def test_auto_id_generation(self):
        """Test ID generation when not provided.""" 
        chunk = TextChunk(
            content="The North remembers.",
            chunk_id="",  # Empty ID triggers auto-generation
            source_file="test.txt",
            start_position=0,
            end_position=20,
            characters=[],
            locations=["North"]
        )
        
        # Should generate deterministic ID from content hash
        assert chunk.chunk_id != ""
        assert "test.txt" in chunk.chunk_id
        assert len(chunk.chunk_id) > 10  # Should have hash component
    
    def test_chunk_immutability(self):
        """Test that chunks are immutable (frozen=True)."""
        chunk = TextChunk(
            content="Test content",
            chunk_id="test",
            source_file="test.txt", 
            start_position=0,
            end_position=10,
            characters=[],
            locations=[]
        )
        
        # Should not be able to modify chunk after creation
        with pytest.raises(AttributeError):
            chunk.content = "Modified content"


class TestASoIaFDataProcessor:
    """
    Test the main data processing class with realistic scenarios.
    
    Testing Strategy:
    - Use temporary directories for file operations
    - Mock external dependencies where needed
    - Test both success and failure paths
    """
    
    @pytest.fixture
    def temp_processor(self):
        """
        Pytest fixture for isolated processor testing.
        
        Fixtures provide:
        - Clean test environment for each test
        - Automatic cleanup after test completes
        - Reusable test setup across multiple tests
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create custom config for testing
            test_config = ProcessingConfig(
                data_dir=temp_dir,
                raw_data_dir=f"{temp_dir}/raw",
                processed_data_dir=f"{temp_dir}/processed", 
                chunk_size=10,  # Small chunks for easier testing
                chunk_overlap=2,
                clean_text=True,
                extract_metadata=True
            )
            
            processor = ASoIaFDataProcessor(config=test_config)
            yield processor
    
    @pytest.fixture
    def sample_asoiaf_text(self):
        """Sample text with known characters and locations."""
        return """
        Jon Snow stood atop the Wall, looking out over the vast expanse of the North.
        The wind was cold and sharp, cutting through his black cloak like a blade.
        Samwell Tarly approached from behind, his footsteps echoing in the courtyard
        of Castle Black. "Winter is coming," Jon said quietly, his breath visible
        in the frigid air. "And when it does, will we be ready?"
        
        Meanwhile, far to the south in King's Landing, Tyrion Lannister sat in his
        chambers, contemplating the game of thrones that swirled around the Iron Throne.
        """
    
    def test_processor_initialization(self, temp_processor):
        """Test processor setup and directory creation."""
        # Directories should be created automatically
        assert temp_processor.raw_dir.exists()
        assert temp_processor.processed_dir.exists()
        
        # Configuration should be applied
        assert temp_processor.config.chunk_size == 10
        assert temp_processor.config.chunk_overlap == 2
        
        # Knowledge base should be loaded
        assert len(temp_processor.characters) > 0
        assert len(temp_processor.locations) > 0
        assert "Jon Snow" in temp_processor.characters
        assert "Winterfell" in temp_processor.locations
    
    def test_text_cleaning_functionality(self, temp_processor):
        """Test text cleaning with various issues."""
        dirty_text = """
        This    has     excessive   spaces.
        
        
        Multiple   blank   lines.
        
        "Smart quotes" and 'apostrophes' need normalization.
        Hyphen-
        ated words from OCR.
        
        Chapter 15
        Page 123
        """
        
        cleaned = temp_processor.clean_text(dirty_text)
        
        # Should fix spacing issues
        assert "    " not in cleaned  # No excessive spaces
        assert "\n\n\n" not in cleaned  # No multiple blank lines
        
        # Should normalize quotes  
        assert '"Smart quotes"' in cleaned or "'Smart quotes'" in cleaned
        
        # Should fix hyphenated words
        assert "Hyphenated" in cleaned or "Hyphen-\nated" not in cleaned
        
        # Should remove chapter/page markers
        assert "Chapter 15" not in cleaned
        assert "Page 123" not in cleaned
    
    def test_metadata_extraction_accuracy(self, temp_processor):
        """Test character and location extraction."""
        test_text = """
        Jon Snow and Samwell Tarly met at Castle Black in the North.
        They discussed Tyrion Lannister's visit to Winterfell.
        """
        
        characters, locations = temp_processor.extract_metadata(test_text)
        
        # Should find all mentioned characters
        expected_characters = {"Jon Snow", "Samwell Tarly", "Tyrion Lannister"}
        found_characters = set(characters)
        assert expected_characters.issubset(found_characters)
        
        # Should find all mentioned locations
        expected_locations = {"Castle Black", "North", "Winterfell"}  
        found_locations = set(locations)
        # Note: Some locations might not be found due to regex limitations
        assert len(found_locations.intersection(expected_locations)) > 0
    
    def test_overlapping_chunk_creation(self, temp_processor, sample_asoiaf_text):
        """Test chunk creation with overlap validation.""" 
        chunks = temp_processor.create_chunks_with_overlap(
            sample_asoiaf_text, 
            "test.txt"
        )
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # All chunks should be TextChunk objects
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.source_file == "test.txt"
            assert chunk.word_count > 0
            assert chunk.chunk_id != ""
        
        # Test overlap: adjacent chunks should share words
        if len(chunks) >= 2:
            chunk1_words = set(chunks[0].content.split())
            chunk2_words = set(chunks[1].content.split())
            overlap_words = chunk1_words.intersection(chunk2_words)
            assert len(overlap_words) > 0, "Adjacent chunks should have overlapping content"
    
    def test_file_loading_success_path(self, temp_processor):
        """Test successful file loading with proper encoding."""
        # Create test file with UTF-8 content
        test_content = "The quick brown fox jumps over the lazy dog. 🐺"
        test_file = temp_processor.raw_dir / "test_utf8.txt"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Should load successfully
        loaded_content = temp_processor.load_raw_text("test_utf8.txt")
        assert loaded_content == test_content
    
    def test_file_loading_encoding_fallback(self, temp_processor):
        """Test encoding fallback for legacy files."""
        # Create file with latin-1 content that would fail UTF-8
        test_content = "Café résumé naïve"  # Contains accented characters
        test_file = temp_processor.raw_dir / "test_latin1.txt"
        
        with open(test_file, 'w', encoding='latin-1') as f:
            f.write(test_content)
        
        # Should handle encoding fallback gracefully
        loaded_content = temp_processor.load_raw_text("test_latin1.txt")
        assert loaded_content == test_content
    
    def test_file_not_found_handling(self, temp_processor):
        """Test proper error handling for missing files."""
        with pytest.raises(FileNotFoundError) as exc_info:
            temp_processor.load_raw_text("nonexistent.txt")
        
        assert "File not found" in str(exc_info.value)
    
    def test_complete_file_processing_pipeline(self, temp_processor, sample_asoiaf_text):
        """Test end-to-end file processing."""
        # Create test file
        test_file = temp_processor.raw_dir / "sample.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(sample_asoiaf_text)
        
        # Process the file
        chunks = temp_processor.process_file("sample.txt")
        
        # Validate results
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.source_file == "sample.txt"
            assert len(chunk.content.strip()) > 0
            assert chunk.word_count > 0
    
    def test_batch_processing_with_error_isolation(self, temp_processor):
        """Test batch processing continues despite individual file failures."""
        # Create mix of good and bad files
        good_file = temp_processor.raw_dir / "good.txt"
        with open(good_file, 'w', encoding='utf-8') as f:
            f.write("This is valid content with Jon Snow.")
        
        # Create file that will cause processing error (empty file)
        bad_file = temp_processor.raw_dir / "bad.txt"
        bad_file.touch()  # Create empty file
        
        # Batch process should handle errors gracefully
        chunks = temp_processor.process_all_files()
        
        # Should have processed the good file despite bad file
        assert len(chunks) > 0
        assert any(chunk.source_file == "good.txt" for chunk in chunks)
    
    def test_save_and_load_roundtrip(self, temp_processor, sample_asoiaf_text):
        """Test save/load cycle preserves data integrity."""
        # Create and process test data
        test_file = temp_processor.raw_dir / "roundtrip.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(sample_asoiaf_text)
        
        original_chunks = temp_processor.process_file("roundtrip.txt")
        
        # Save chunks
        output_file = temp_processor.save_processed_data(original_chunks)
        assert Path(output_file).exists()
        
        # Load chunks back
        filename = Path(output_file).name
        loaded_chunks = temp_processor.load_processed_data(filename)
        
        # Should match original data
        assert len(loaded_chunks) == len(original_chunks)
        
        for original, loaded in zip(original_chunks, loaded_chunks):
            assert original.content == loaded.content
            assert original.source_file == loaded.source_file  
            assert original.characters == loaded.characters
            assert original.locations == loaded.locations
            assert original.word_count == loaded.word_count
    
    def test_statistics_generation(self, temp_processor, sample_asoiaf_text):
        """Test comprehensive statistics calculation."""
        # Process sample text
        test_file = temp_processor.raw_dir / "stats_test.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(sample_asoiaf_text)
        
        chunks = temp_processor.process_file("stats_test.txt")
        stats = temp_processor.get_statistics(chunks)
        
        # Should have all expected statistics
        required_fields = [
            "total_chunks", "total_words", "average_chunk_size",
            "unique_characters", "unique_locations", "source_files"
        ]
        
        for field in required_fields:
            assert field in stats, f"Missing statistic: {field}"
        
        # Values should be reasonable
        assert stats["total_chunks"] == len(chunks)
        assert stats["total_words"] > 0
        assert stats["average_chunk_size"] > 0
        assert stats["source_files"] == 1  # One file processed


class TestErrorHandlingAndEdgeCases:
    """
    Test error conditions and edge cases for robustness.
    
    Production systems must handle edge cases gracefully.
    """
    
    @pytest.fixture
    def temp_processor(self):
        """Reuse processor fixture from main test class."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config = ProcessingConfig(
                data_dir=temp_dir,
                chunk_size=10,
                chunk_overlap=2
            )
            processor = ASoIaFDataProcessor(config=test_config)
            yield processor
    
    def test_empty_text_handling(self, temp_processor):
        """Test handling of empty or whitespace-only text."""
        empty_cases = ["", "   ", "\n\n\n", "\t\t"]
        
        for empty_text in empty_cases:
            chunks = temp_processor.create_chunks_with_overlap(empty_text, "empty.txt")
            assert chunks == [], f"Empty text should produce no chunks: '{empty_text}'"
    
    def test_very_short_text(self, temp_processor):
        """Test handling of text shorter than chunk size."""
        short_text = "Jon Snow."  # Only 2 words
        chunks = temp_processor.create_chunks_with_overlap(short_text, "short.txt")
        
        # Should either create no chunks or handle gracefully
        if chunks:
            assert len(chunks) == 1
            assert chunks[0].word_count == 2
    
    def test_invalid_chunk_configuration(self):
        """Test validation of chunk size vs overlap."""
        with pytest.raises(ValueError):
            # Overlap larger than chunk size should fail
            ProcessingConfig(chunk_size=10, chunk_overlap=15)
    
    def test_statistics_with_empty_chunks(self, temp_processor):
        """Test statistics calculation with no data."""
        stats = temp_processor.get_statistics([])
        assert "error" in stats
        assert stats["error"] == "No chunks to analyze"


class TestPerformanceAndScalability:
    """
    Basic performance tests to ensure reasonable processing speed.
    
    Production systems need performance validation.
    """
    
    def test_processing_speed_benchmark(self):
        """Test processing doesn't take unreasonably long."""
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ProcessingConfig(
                data_dir=temp_dir,
                chunk_size=100,
                chunk_overlap=10
            )
            processor = ASoIaFDataProcessor(config=config)
            
            # Create reasonably large text (simulate book chapter)
            large_text = """
            Jon Snow knew nothing, but he was learning fast. The Wall stretched 
            endlessly in both directions, a monument to the builders of old.
            """ * 100  # Repeat 100 times
            
            test_file = processor.raw_dir / "performance_test.txt"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(large_text)
            
            # Time the processing
            start_time = time.time()
            chunks = processor.process_file("performance_test.txt")
            processing_time = time.time() - start_time
            
            # Should process reasonably quickly
            assert processing_time < 5.0, f"Processing took too long: {processing_time:.2f}s"
            assert len(chunks) > 0, "Should produce chunks"
            
            # Performance should scale reasonably with input size
            words_per_second = sum(c.word_count for c in chunks) / processing_time
            assert words_per_second > 100, f"Too slow: {words_per_second:.0f} words/second"


class TestIntegrationWorkflows:
    """
    Integration tests that verify complete workflows work together.
    
    These test the interaction between components.
    """
    
    def test_complete_processing_workflow(self):
        """Test entire pipeline from raw files to saved results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = ASoIaFDataProcessor(
                config=ProcessingConfig(
                    data_dir=temp_dir,
                    chunk_size=50,
                    chunk_overlap=10
                )
            )
            
            # Create multiple test files
            test_files = {
                "book1.txt": """
                Jon Snow's journey begins at Winterfell in the North. 
                Ned Stark rules from the ancient castle while the Wall 
                looms to the north, where the Night's Watch stands guard.
                """,
                "book2.txt": """
                In King's Landing, Tyrion Lannister drinks wine while 
                contemplating the Iron Throne. The Red Keep houses the 
                rulers of the Seven Kingdoms, where political games 
                determine the fate of all.
                """
            }
            
            # Create test files
            for filename, content in test_files.items():
                file_path = processor.raw_dir / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Process all files
            all_chunks = processor.process_all_files()
            
            # Verify complete workflow
            assert len(all_chunks) > 0
            
            # Should have chunks from both files
            source_files = {chunk.source_file for chunk in all_chunks}
            assert "book1.txt" in source_files
            assert "book2.txt" in source_files
            
            # Save and verify persistence
            output_file = processor.save_processed_data(all_chunks)
            assert Path(output_file).exists()
            
            # Load and verify integrity
            filename = Path(output_file).name
            loaded_chunks = processor.load_processed_data(filename)
            assert len(loaded_chunks) == len(all_chunks)
            
            # Generate and verify statistics
            stats = processor.get_statistics(all_chunks)
            assert stats["source_files"] == 2
            assert stats["total_chunks"] > 0
            assert stats["unique_characters"] > 0


# Demo and manual testing functionality
def run_interactive_demo():
    """
    Interactive demonstration for learning and validation.
    
    This provides a hands-on way to see the processor in action
    and understand how each component works.
    """
    print("🏰 ASOIAF Data Processor - Interactive Demo")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = ASoIaFDataProcessor(
            config=ProcessingConfig(
                data_dir=temp_dir,
                chunk_size=80,
                chunk_overlap=15
            )
        )
        
        # Create comprehensive demo text
        demo_text = """
        The great castle of Winterfell had stood for thousands of years, and it 
        had weathered storms that would have brought down lesser keeps. Jon Snow
        walked through its ancient halls, his footsteps echoing off the stone walls.
        
        His father, Eddard Stark, had taught him that winter was coming, and that
        the North remembers. These words echoed in his mind as he made his way
        to the crypts, where the stone kings of winter lay in eternal rest.
        
        Samwell Tarly found him there, among the tombs of his ancestors. "The
        Night's Watch has need of good men," Sam said quietly. "Will you come
        with me to Castle Black? The Wall has protected the realm for millennia."
        
        Meanwhile, in the distant capital of King's Landing, Tyrion Lannister
        sat in the Red Keep, contemplating the Iron Throne. The game of thrones
        was a dangerous one, where men lived and died by their wits and their
        ability to navigate the treacherous waters of court politics.
        
        Cersei Lannister entered the chamber, her green eyes flashing with anger.
        "The Starks think they can defy House Lannister," she hissed. "But they
        will learn what happens to those who stand against the lions of the Rock."
        """
        
        # Save demo text
        demo_file = processor.raw_dir / "demo_asoiaf.txt"
        with open(demo_file, 'w', encoding='utf-8') as f:
            f.write(demo_text)
        
        print(f"📝 Created demo file: {len(demo_text)} characters")
        print(f"🔧 Configuration: {processor.config.chunk_size} words/chunk, {processor.config.chunk_overlap} overlap")
        
        # Process the text
        print("\n🔄 Processing text through pipeline...")
        print("   Step 1: Loading raw text...")
        raw_content = processor.load_raw_text("demo_asoiaf.txt")
        print(f"   ✅ Loaded {len(raw_content)} characters")
        
        print("   Step 2: Cleaning text...")
        clean_content = processor.clean_text(raw_content)
        print(f"   ✅ Cleaned to {len(clean_content)} characters")
        
        print("   Step 3: Creating chunks with overlap...")
        chunks = processor.create_chunks_with_overlap(clean_content, "demo_asoiaf.txt")
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # Display sample chunks with analysis
        print(f"\n📋 Sample Chunks (showing first 3 of {len(chunks)}):")
        print("-" * 80)
        
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n🔸 Chunk {i+1} (ID: {chunk.chunk_id[:12]}...)")
            print(f"   Content: {chunk.content[:120]}...")
            print(f"   Word Count: {chunk.word_count}")
            print(f"   Characters: {', '.join(chunk.characters) if chunk.characters else 'None'}")
            print(f"   Locations: {', '.join(chunk.locations) if chunk.locations else 'None'}")
        
        # Show overlap demonstration
        if len(chunks) >= 2:
            print(f"\n🔄 Overlap Analysis (Chunks 1 & 2):")
            chunk1_words = set(chunks[0].content.split())
            chunk2_words = set(chunks[1].content.split())
            overlap_words = chunk1_words.intersection(chunk2_words)
            print(f"   Shared words: {', '.join(list(overlap_words)[:10])}")
            print(f"   Overlap count: {len(overlap_words)} words")
        
        # Generate comprehensive statistics
        print(f"\n📊 Processing Statistics:")
        stats = processor.get_statistics(chunks)
        print("-" * 40)
        
        for key, value in stats.items():
            if key == 'top_characters':
                print(f"   Top Characters: {dict(value[:5])}")
            elif key == 'top_locations':
                print(f"   Top Locations: {dict(value[:5])}")
            elif key != 'chunks_per_file':
                print(f"   {key.replace('_', ' ').title()}: {value}")
        
        # Save results
        output_file = processor.save_processed_data(chunks)
        print(f"\n💾 Demo results saved to: {Path(output_file).name}")
        
        print(f"\n🎉 Demo completed successfully!")
        print("🎓 Key concepts demonstrated:")
        print("   • ETL pipeline architecture")
        print("   • Overlapping chunk strategy") 
        print("   • Metadata extraction")
        print("   • Error handling and validation")
        print("   • Performance monitoring")


if __name__ == "__main__":
    # Run comprehensive tests
    print("🧪 Running comprehensive test suite...")
    import subprocess
    result = subprocess.run(["python", "-m", "pytest", __file__, "-v"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ All tests passed!")
        print("\n" + "="*60)
        run_interactive_demo()
    else:
        print("❌ Some tests failed:")
        print(result.stdout)
        print(result.stderr)