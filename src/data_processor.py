# src/data_processor.py
"""
ASOIAF Text Processing Pipeline

This is the core data processing module that transforms raw text into
AI-ready chunks with metadata. This demonstrates production-grade
text processing with proper error handling and performance optimization.

Architecture: ETL Pipeline
- Extract: Load raw text files with encoding handling
- Transform: Clean text, create overlapping chunks, extract metadata  
- Load: Save structured chunks with provenance tracking

Key Patterns:
- Strategy Pattern: Different processing strategies
- Template Method: Consistent processing workflow
- Error Isolation: Batch processing with individual error handling
"""

import re
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime

from .config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


@dataclass(frozen=True)  # Immutable for thread safety
class TextChunk:
    """
    Immutable data structure representing a processed text chunk.
    
    Architecture Decision: Dataclass vs regular class
    - Automatic __init__, __repr__, __eq__ methods
    - Type hints for IDE support and validation
    - Frozen=True prevents accidental modification
    - JSON serialization support with asdict()
    
    Metadata Co-location: Store entities with content for fast filtering
    vs. normalized database approach (entities in separate tables)
    Trade-off: Some data duplication vs. query performance
    """
    # Core content
    content: str                    # The actual text content
    chunk_id: str                  # Unique identifier for tracking
    source_file: str               # Provenance - which file this came from
    
    # Position tracking - enables reconstruction and debugging
    start_position: int            # Character position in original text
    end_position: int              # End character position
    
    # Extracted metadata - enables semantic filtering
    characters: List[str]          # Character names found in this chunk
    locations: List[str]           # Location names found in this chunk
    
    # Optional hierarchical context
    book: Optional[str] = None     # Book name if extractable
    chapter: Optional[str] = None  # Chapter name if extractable
    
    # Computed properties
    word_count: int = 0           # For performance optimization and filtering
    
    def __post_init__(self):
        """
        Computed properties and validation after initialization.
        
        This method demonstrates defensive programming:
        - Calculate derived values automatically
        - Validate data integrity
        - Generate IDs if not provided
        """
        if not self.word_count:
            # Calculate word count automatically
            object.__setattr__(self, 'word_count', len(self.content.split()))
        
        if not self.chunk_id:
            # Generate deterministic ID from content hash
            content_hash = hashlib.md5(self.content.encode()).hexdigest()
            chunk_id = f"{self.source_file}_{content_hash[:8]}"
            object.__setattr__(self, 'chunk_id', chunk_id)


class ASoIaFDataProcessor:
    """
    Main data processing pipeline for ASOIAF text corpus.
    
    Architecture: Single Responsibility Principle
    - One class, one job: process ASOIAF text
    - Methods are focused and testable
    - Configuration externalized via dependency injection
    
    Processing Strategy: ETL Pipeline
    1. Extract: Load files with proper encoding handling
    2. Transform: Clean → Chunk → Extract metadata → Validate
    3. Load: Serialize to JSON with error handling
    """
    
    def __init__(self, config=None):
        """
        Initialize processor with configuration.
        
        Dependency Injection: Accept config object instead of hardcoding
        - Enables testing with different configurations
        - Supports environment-specific settings
        - Makes behavior explicit and configurable
        """
        self.config = config or settings.processing
        
        # Set up directories
        self.data_dir = Path(self.config.data_dir)
        self.raw_dir = Path(self.config.raw_data_dir)
        self.processed_dir = Path(self.config.processed_data_dir)
        
        # Ensure directories exist - defensive programming
        for directory in [self.data_dir, self.raw_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized processor with data_dir={self.data_dir}")
        
        # Load ASOIAF-specific knowledge base
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """
        Initialize entity recognition patterns.
        
        Architecture Decision: In-memory vs Database
        - In-memory: Faster lookup, simpler deployment
        - Database: More scalable, easier to update
        
        Current choice: In-memory (good for <10k entities)
        Future evolution: Database-backed for larger knowledge bases
        """
        # ASOIAF Character Knowledge Base
        # In production, this would come from a database or API
        self.characters = {
            # Stark family
            "Jon Snow", "Arya Stark", "Sansa Stark", "Bran Stark", 
            "Rickon Stark", "Robb Stark", "Catelyn Stark", "Ned Stark", 
            "Eddard Stark", "Lyanna Stark",
            
            # Lannister family  
            "Tyrion Lannister", "Jaime Lannister", "Cersei Lannister",
            "Tywin Lannister", "Joffrey Baratheon", "Tommen Baratheon",
            "Myrcella Baratheon",
            
            # Targaryen family
            "Daenerys Targaryen", "Viserys Targaryen", "Aegon Targaryen",
            "Rhaegar Targaryen", "Aemon Targaryen",
            
            # Other major characters
            "Samwell Tarly", "Davos Seaworth", "Stannis Baratheon",
            "Robert Baratheon", "Petyr Baelish", "Varys", "Theon Greyjoy",
            "Sandor Clegane", "Gregor Clegane", "Bronn", "Oberyn Martell"
        }
        
        # ASOIAF Location Knowledge Base
        self.locations = {
            # Major regions
            "Westeros", "Essos", "The North", "The Reach", "Dorne",
            "The Iron Islands", "The Riverlands", "The Vale", 
            "The Stormlands", "The Crownlands", "The Westerlands",
            
            # Major cities and castles
            "King's Landing", "Winterfell", "Casterly Rock", "Storm's End",
            "Dragonstone", "Highgarden", "Sunspear", "The Eyrie",
            "Riverrun", "Pyke", "Oldtown",
            
            # The Wall and beyond
            "The Wall", "Castle Black", "Eastwatch-by-the-Sea",
            "Shadow Tower", "Beyond the Wall",
            
            # Essos locations
            "Braavos", "Pentos", "Meereen", "Qarth", "Volantis",
            "King's Landing", "Dragonstone",
            
            # Geographic features
            "Blackwater Bay", "The Trident", "The Narrow Sea"
        }
        
        # Compile regex patterns for performance
        # Why regex? Fast pattern matching for known entities
        # Alternative: spaCy NER (more accurate, much slower)
        self.character_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(char) for char in self.characters) + r')\b',
            re.IGNORECASE
        )
        
        self.location_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(loc) for loc in self.locations) + r')\b',
            re.IGNORECASE  
        )
        
        logger.info(f"Loaded {len(self.characters)} characters and {len(self.locations)} locations")
    
    def load_raw_text(self, filename: str) -> str:
        """
        Load raw text with robust encoding handling.
        
        Error Handling Strategy:
        1. Try UTF-8 first (most common)
        2. Fallback to latin-1 for legacy files
        3. Provide clear error messages
        4. Don't fail silently
        
        This demonstrates production-ready file handling.
        """
        file_path = self.raw_dir / filename
        
        try:
            # UTF-8 is the standard encoding
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Successfully loaded {len(content)} characters from {filename}")
                return content
                
        except UnicodeDecodeError:
            # Fallback for older text files
            logger.warning(f"UTF-8 failed for {filename}, trying latin-1 encoding")
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    logger.info(f"Successfully loaded {filename} with latin-1 encoding")
                    return content
            except Exception as e:
                logger.error(f"Failed to read {filename} with any encoding: {e}")
                raise
                
        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        except Exception as e:
            error_msg = f"Unexpected error loading {filename}: {e}"
            logger.error(error_msg)
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Text Cleaning Rationale:
        - OCR errors in digitized books need fixing
        - Inconsistent spacing breaks chunking
        - Quote normalization prevents encoding issues
        - Chapter markers are metadata, not content
        
        Each cleaning step solves a real problem in text processing.
        """
        if not text or not text.strip():
            return ""
        
        # Remove excessive whitespace (common in digitized texts)
        text = re.sub(r'\s+', ' ', text)
        
        # Fix hyphenated line breaks from OCR
        # "exam-\nple" → "example"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Normalize quotes for consistency
        text = re.sub(r'["""]', '"', text)  # Smart quotes → regular quotes
        text = re.sub(r'['']', "'", text)   # Smart apostrophes → regular
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([.!?;:,])\s*', r'\1 ', text)
        
        # Remove common book artifacts
        text = re.sub(r'Chapter \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces that cleaning might have introduced
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_metadata(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract character and location entities from text.
        
        Named Entity Recognition Strategy:
        - Regex-based: Fast, works for known entities
        - Alternative: spaCy NER (slower, finds unknown entities)
        - Alternative: OpenAI API (most accurate, costs money)
        
        Current choice optimizes for speed and cost over perfect accuracy.
        """
        # Find character mentions
        character_matches = self.character_pattern.findall(text)
        characters = list(set(character_matches))  # Remove duplicates
        
        # Find location mentions  
        location_matches = self.location_pattern.findall(text)
        locations = list(set(location_matches))
        
        logger.debug(f"Extracted {len(characters)} characters, {len(locations)} locations")
        return characters, locations
    
    def create_chunks_with_overlap(self, text: str, source_file: str) -> List[TextChunk]:
        """
        Split text into overlapping chunks for optimal RAG performance.
        
        Chunking Strategy: Overlapping Windows
        - Chunk Size: Balance between context and precision
        - Overlap: Prevent information loss at boundaries
        - Step Size: chunk_size - overlap (how far to advance)
        
        Why Overlapping?
        Example without overlap:
        Chunk 1: "Jon Snow traveled from Winterfell"
        Chunk 2: "to Castle Black where he joined"
        → Query "Jon Snow at Castle Black" might miss context
        
        With overlap:
        Chunk 1: "Jon Snow traveled from Winterfell to"
        Chunk 2: "from Winterfell to Castle Black where he"
        → Better context preservation
        """
        if not text or not text.strip():
            return []
        
        words = text.split()
        chunks = []
        
        # Calculate step size
        step_size = self.config.chunk_size - self.config.chunk_overlap
        
        if step_size <= 0:
            raise ValueError("Chunk overlap must be smaller than chunk size")
        
        for i in range(0, len(words), step_size):
            # Extract chunk words
            chunk_words = words[i:i + self.config.chunk_size]
            
            # Skip chunks that are too small (quality control)
            if len(chunk_words) < self.config.min_chunk_size:
                break
            
            chunk_text = ' '.join(chunk_words)
            
            # Extract metadata for this chunk
            characters, locations = self.extract_metadata(chunk_text)
            
            # Create chunk with all metadata
            chunk = TextChunk(
                content=chunk_text,
                chunk_id="",  # Will be auto-generated
                source_file=source_file,
                start_position=i,
                end_position=i + len(chunk_words),
                characters=characters,
                locations=locations
            )
            
            chunks.append(chunk)
            logger.debug(f"Created chunk {len(chunks)}: {len(chunk_words)} words, "
                        f"{len(characters)} characters, {len(locations)} locations")
        
        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks
    
    def process_file(self, filename: str) -> List[TextChunk]:
        """
        Complete processing pipeline for a single file.
        
        ETL Pipeline Implementation:
        1. Extract: Load raw text with encoding handling
        2. Transform: Clean → Chunk → Extract metadata
        3. Load: Return structured chunks
        
        Error Handling: Fail fast with clear error messages
        """
        logger.info(f"Processing file: {filename}")
        
        try:
            # EXTRACT: Load raw text
            raw_text = self.load_raw_text(filename)
            if not raw_text:
                logger.warning(f"File {filename} is empty")
                return []
            
            # TRANSFORM: Clean text
            if self.config.clean_text:
                cleaned_text = self.clean_text(raw_text)
            else:
                cleaned_text = raw_text
            
            # TRANSFORM: Create chunks with metadata
            chunks = self.create_chunks_with_overlap(cleaned_text, filename)
            
            logger.info(f"Successfully processed {filename}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            error_msg = f"Failed to process {filename}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def process_all_files(self) -> List[TextChunk]:
        """
        Batch process all files with error isolation.
        
        Error Isolation Strategy:
        - Continue processing even if individual files fail
        - Log errors but don't crash entire batch
        - Return all successfully processed chunks
        
        This is crucial for production systems where partial success
        is better than total failure.
        """
        all_chunks = []
        
        # Find all text files
        text_files = list(self.raw_dir.glob("*.txt"))
        
        if not text_files:
            logger.warning(f"No .txt files found in {self.raw_dir}")
            return all_chunks
        
        logger.info(f"Found {len(text_files)} files to process")
        
        # Process each file with error isolation
        successful_files = 0
        failed_files = 0
        
        for file_path in text_files:
            try:
                chunks = self.process_file(file_path.name)
                all_chunks.extend(chunks)
                successful_files += 1
                
            except Exception as e:
                # Log error but continue with other files
                logger.error(f"Skipping {file_path.name}: {e}")
                failed_files += 1
                continue
        
        logger.info(f"Batch processing complete: {successful_files} successful, {failed_files} failed")
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        return all_chunks
    
    def save_processed_data(self, chunks: List[TextChunk]) -> str:
        """
        Save processed chunks to JSON with metadata.
        
        Storage Format Decision: JSON vs Binary vs Database
        - JSON: Human readable, version controllable, language agnostic
        - Binary (pickle): Faster, smaller, Python-only
        - Database: Queryable, concurrent access, more complex
        
        JSON chosen for development/educational phase.
        Production would likely use vector database directly.
        """
        if not chunks:
            logger.warning("No chunks to save")
            return ""
        
        # Create timestamped filename for version tracking
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.processed_dir / f"asoiaf_chunks_{timestamp}.json"
        
        try:
            # Convert chunks to serializable format
            chunks_data = [asdict(chunk) for chunk in chunks]
            
            # Add processing metadata
            output_data = {
                "metadata": {
                    "processing_timestamp": timestamp,
                    "processor_version": "1.0.0",
                    "chunk_count": len(chunks),
                    "configuration": {
                        "chunk_size": self.config.chunk_size,
                        "chunk_overlap": self.config.chunk_overlap,
                        "clean_text": self.config.clean_text
                    }
                },
                "chunks": chunks_data
            }
            
            # Write with proper formatting for debugging
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(chunks)} chunks to {output_file}")
            return str(output_file)
            
        except Exception as e:
            error_msg = f"Failed to save chunks: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def load_processed_data(self, filename: str) -> List[TextChunk]:
        """
        Load processed chunks from JSON file.
        
        Deserialization with validation to ensure data integrity.
        """
        file_path = self.processed_dir / filename
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both old format (direct chunks) and new format (with metadata)
            if 'chunks' in data:
                chunks_data = data['chunks']
                metadata = data.get('metadata', {})
                logger.info(f"Loaded file with metadata: {metadata}")
            else:
                chunks_data = data  # Old format
            
            # Convert back to TextChunk objects with validation
            chunks = []
            for chunk_data in chunks_data:
                try:
                    chunk = TextChunk(**chunk_data)
                    chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Skipping invalid chunk: {e}")
                    continue
            
            logger.info(f"Loaded {len(chunks)} valid chunks from {filename}")
            return chunks
            
        except FileNotFoundError:
            error_msg = f"Processed data file not found: {filename}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load processed data: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_statistics(self, chunks: List[TextChunk]) -> Dict:
        """
        Generate comprehensive statistics about the processed corpus.
        
        Statistics Purpose:
        - Data quality assessment
        - Processing validation
        - Performance monitoring
        - Stakeholder reporting
        
        These metrics help identify issues and track improvements.
        """
        if not chunks:
            return {"error": "No chunks to analyze"}
        
        # Basic statistics
        total_words = sum(chunk.word_count for chunk in chunks)
        word_counts = [chunk.word_count for chunk in chunks]
        
        # Character and location analysis
        all_characters = [char for chunk in chunks for char in chunk.characters]
        all_locations = [loc for chunk in chunks for loc in chunk.locations]
        
        # Count frequencies
        char_counts = {}
        for char in all_characters:
            char_counts[char] = char_counts.get(char, 0) + 1
            
        loc_counts = {}
        for loc in all_locations:
            loc_counts[loc] = loc_counts.get(loc, 0) + 1
        
        # Source file analysis
        file_counts = {}
        for chunk in chunks:
            file_counts[chunk.source_file] = file_counts.get(chunk.source_file, 0) + 1
        
        stats = {
            # Basic metrics
            "total_chunks": len(chunks),
            "total_words": total_words,
            "average_chunk_size": total_words / len(chunks),
            "min_chunk_size": min(word_counts),
            "max_chunk_size": max(word_counts),
            
            # Entity metrics
            "unique_characters": len(set(all_characters)),
            "unique_locations": len(set(all_locations)),
            "total_character_mentions": len(all_characters),
            "total_location_mentions": len(all_locations),
            
            # Source file metrics
            "source_files": len(set(chunk.source_file for chunk in chunks)),
            "chunks_per_file": dict(file_counts),
            
            # Top entities (for content validation)
            "top_characters": sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_locations": sorted(loc_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        return stats


# Command-line interface for development and testing
if __name__ == "__main__":
    """
    CLI for running the data processor independently.
    
    Good Python practices:
    - argparse for command-line interfaces
    - Main guard for importable modules
    - Example usage and help text
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ASOIAF Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
    """
    Examples:
    python src/data_processor.py --initialize
    python src/data_processor.py --chunk-size 300 --overlap 30
    python src/data_processor.py --stats-only
    """)
    
    parser.add_argument("--chunk-size", type=int, default=500,
                       help="Words per chunk (default: 500)")
    parser.add_argument("--overlap", type=int, default=50,
                       help="Overlap between chunks (default: 50)")
    parser.add_argument("--initialize", action="store_true",
                       help="Create sample data for testing")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only show statistics for existing processed data")
    
    args = parser.parse_args()
    
    # Create custom configuration
    from .config import ProcessingConfig
    custom_config = ProcessingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap
    )
    
    # Initialize processor
    processor = ASoIaFDataProcessor(config=custom_config)
    
    if args.initialize:
        # Create sample ASOIAF text for testing
        sample_text = """
        Jon Snow stood atop the Wall, gazing out at the vast expanse of the North. 
        The wind howled through the crenellations of Castle Black, carrying with it 
        the scent of snow and pine. Samwell Tarly approached from behind, his heavy 
        footsteps echoing in the courtyard.
        
        "The Night's Watch has protected the realm for thousands of years," Sam said, 
        his breath visible in the cold air. "From Eastwatch-by-the-Sea to the Shadow Tower, 
        we stand guard against the darkness beyond."
        
        Jon turned to face his friend, his bastard sword at his side. "Winter is 
        coming, Sam. And when it does, will we be ready?"
        
        Meanwhile, far to the south in King's Landing, Tyrion Lannister poured 
        himself another cup of wine. The Red Keep's walls seemed to close in on 
        him as he contemplated the game of thrones that swirled around the Iron Throne.
        
        Cersei Lannister entered the chamber, her green eyes flashing with anger. 
        "The Starks think they can defy House Lannister," she hissed. "They will 
        learn what happens to those who stand against us."
        
        In Winterfell, Arya Stark practiced with her needle in the godswood. 
        The heart tree's red leaves rustled in the wind, and she could almost 
        hear her father's voice: "When the snows fall and the white winds blow, 
        the lone wolf dies, but the pack survives."
        """
        
        sample_file = processor.raw_dir / "sample_asoiaf.txt"
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        print(f"✅ Created sample data: {sample_file}")
        print("Run the processor again without --initialize to process the sample data.")
        exit(0)
    
    if args.stats_only:
        # Show statistics for existing processed data
        processed_files = list(processor.processed_dir.glob("asoiaf_chunks_*.json"))
        if not processed_files:
            print("❌ No processed data found. Run processor first.")
            exit(1)
        
        latest_file = max(processed_files, key=lambda p: p.stat().st_mtime)
        chunks = processor.load_processed_data(latest_file.name)
        stats = processor.get_statistics(chunks)
        
        print("\n📊 Processing Statistics")
        print("=" * 50)
        for key, value in stats.items():
            if key in ['top_characters', 'top_locations']:
                print(f"{key}: {dict(value[:5])}")  # Show top 5
            elif key == 'chunks_per_file':
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")
        exit(0)
    
    # Main processing pipeline
    try:
        print("🏰 ASOIAF Data Processor")
        print("=" * 50)
        print(f"Configuration:")
        print(f"  Chunk Size: {custom_config.chunk_size} words")
        print(f"  Overlap: {custom_config.chunk_overlap} words")
        print(f"  Data Directory: {custom_config.data_dir}")
        
        # Process all files
        chunks = processor.process_all_files()
        
        if chunks:
            # Save processed data
            output_file = processor.save_processed_data(chunks)
            
            # Display statistics
            stats = processor.get_statistics(chunks)
            print("\n📊 Processing Results")
            print("=" * 50)
            for key, value in stats.items():
                if key in ['top_characters', 'top_locations']:
                    print(f"{key}: {dict(value[:5])}")
                elif key != 'chunks_per_file':
                    print(f"{key}: {value}")
            
            print(f"\n💾 Results saved to: {Path(output_file).name}")
            print("\n✅ Processing completed successfully!")
            
        else:
            print("❌ No chunks created. Check your input files.")
            print("💡 Use --initialize to create sample data for testing.")
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"❌ Error: {e}")
        exit(1)