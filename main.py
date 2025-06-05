import asyncio
import aiohttp
import feedparser
import yaml
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import spacy
import importlib.util
from gradio_client import Client

# Check if edge_tts is available for fallback
edge_tts_available = importlib.util.find_spec("edge_tts") is not None
if edge_tts_available:
    import edge_tts

# === ADVANCED CONFIGURATION ===
CONFIG = {
    "ollama_api": {
        "base_url": "http://localhost:11434"
    },
    "models": {
        "summary_model": "mistral:latest",
        "broadcast_model": "mistral-small:24b-instruct-2501-q8_0",
        "embedding_model": "nomic-embed-text"
    },
    "tts": {
        "chatterbox_url": "ResembleAI/Chatterbox",
        "exaggeration": 0.5,
        "temperature": 0.8,
        "seed": 0,
        "cfg_pace": 0.5,
        "reference_audio": None  # Optional reference audio file path
    },
    "processing": {
        "max_articles_per_feed": 10,
        "min_article_length": 100,
        "max_clusters": 20,
        "similarity_threshold": 0.7,
        "sentiment_weight": 0.3,
        "freshness_weight": 0.4,
        "relevance_weight": 0.3
    },
    "output": {
        "max_broadcast_length": 3000,
        "target_segments": 20,
        "include_transitions": True
    }
}

# === DATA STRUCTURES ===
@dataclass
class Article:
    title: str
    content: str
    url: str
    published: datetime
    source: str
    summary: str = ""
    sentiment_score: float = 0.0
    importance_score: float = 0.0
    embedding: List[float] = None
    cluster_id: int = -1

@dataclass
class BroadcastSegment:
    topic: str
    content: str
    articles: List[Article]
    importance: float
    duration_estimate: int

class AdvancedNewsGenerator:
    def __init__(self, feeds_file: str = "feeds.yaml"):
        self.feeds_file = feeds_file
        self.db_path = "news_cache.db"
        self.setup_logging()
        self.setup_database()
        self.setup_nlp()
        
    def setup_logging(self):
        """Setup advanced logging with performance metrics"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('news_generator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_database(self):
        """Setup SQLite database for caching and deduplication"""
        # Register adapter for datetime objects to prevent deprecation warnings
        sqlite3.register_adapter(dt.datetime, lambda val: val.isoformat())
        sqlite3.register_converter("timestamp", lambda val: dt.datetime.fromisoformat(val.decode()))
        
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                url TEXT UNIQUE,
                published TIMESTAMP,
                source TEXT,
                content_hash TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        
    def setup_nlp(self):
        """Initialize NLP models and tools"""
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.logger.warning("Could not initialize NLTK sentiment analyzer")
            self.sentiment_analyzer = None
            
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    async def fetch_feeds_async(self) -> List[Article]:
        """Asynchronously fetch and parse RSS feeds"""
        with open(self.feeds_file, 'r') as f:
            feeds_config = yaml.safe_load(f)
        
        feeds = feeds_config.get('feeds', [])
        articles = []
        
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_single_feed(session, feed) for feed in feeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    articles.extend(result)
                else:
                    self.logger.error(f"Feed fetch error: {result}")
                    
        return articles

    async def fetch_single_feed(self, session: aiohttp.ClientSession, feed_url: str) -> List[Article]:
        """Fetch and parse a single RSS feed"""
        try:
            async with session.get(feed_url, timeout=30) as response:
                content = await response.text()
                
            feed = feedparser.parse(content)
            articles = []
            
            for entry in feed.entries[:CONFIG["processing"]["max_articles_per_feed"]]:
                # Extract content
                content = self.extract_content(entry)
                if len(content) < CONFIG["processing"]["min_article_length"]:
                    continue
                    
                # Parse publication date
                published = self.parse_date(entry)
                
                # Check for duplicates
                content_hash = hashlib.md5(content.encode()).hexdigest()
                if self.is_duplicate(content_hash):
                    continue
                
                article = Article(
                    title=entry.get('title', ''),
                    content=content,
                    url=entry.get('link', ''),
                    published=published,
                    source=feed.feed.get('title', feed_url),
                )
                
                articles.append(article)
                self.cache_article(article, content_hash)
                
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching feed {feed_url}: {e}")
            return []

    def extract_content(self, entry) -> str:
        """Extract and clean article content"""
        content = ""
        
        # Try different content fields
        for field in ['content', 'summary', 'description']:
            if hasattr(entry, field):
                if field == 'content' and entry.content:
                    content = entry.content[0].value if entry.content else ""
                else:
                    content = getattr(entry, field, "")
                break
        
        # Clean HTML and normalize text
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content

    def parse_date(self, entry) -> datetime:
        """Parse publication date with fallback"""
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                return datetime(*entry.published_parsed[:6])
        except:
            pass
        
        return datetime.now()

    def is_duplicate(self, content_hash: str) -> bool:
        """Check if article is duplicate using content hash"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT 1 FROM articles WHERE content_hash = ? AND processed_at > ?",
            (content_hash, datetime.now() - timedelta(days=7))
        )
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def cache_article(self, article: Article, content_hash: str):
        """Cache article to prevent reprocessing"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO articles 
                (id, title, content, url, published, source, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                hashlib.md5(article.url.encode()).hexdigest(),
                article.title, article.content, article.url,
                article.published, article.source, content_hash
            ))
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error caching article: {e}")
        finally:
            conn.close()

    async def process_articles_advanced(self, articles: List[Article]) -> List[Article]:
        """Advanced processing pipeline for articles"""
        if not articles:
            return []
            
        self.logger.info(f"Processing {len(articles)} articles with advanced algorithms")
        
        # Parallel processing for efficiency
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Generate summaries
            summary_futures = {
                executor.submit(self.generate_summary, article): article 
                for article in articles
            }
            
            for future in as_completed(summary_futures):
                article = summary_futures[future]
                try:
                    article.summary = future.result()
                except Exception as e:
                    self.logger.error(f"Summary generation failed: {e}")
                    article.summary = article.content[:200] + "..."
            
            # Calculate sentiment scores
            sentiment_futures = {
                executor.submit(self.calculate_sentiment, article): article 
                for article in articles
            }
            
            for future in as_completed(sentiment_futures):
                article = sentiment_futures[future]
                try:
                    article.sentiment_score = future.result()
                except Exception as e:
                    self.logger.error(f"Sentiment analysis failed: {e}")
                    article.sentiment_score = 0.0
        
        # Generate embeddings for clustering
        articles = await self.generate_embeddings(articles)
        
        # Cluster similar articles
        articles = self.cluster_articles(articles)
        
        # Calculate importance scores
        articles = self.calculate_importance_scores(articles)
        
        return articles

    def generate_summary(self, article: Article) -> str:
        """Generate summary using Ollama with advanced prompting"""
        prompt = f"""
        Summarize the following news article in 2-3 sentences. Focus on the key facts, impact, and relevance.
        Make it suitable for a professional news broadcast.
        
        Title: {article.title}
        Content: {article.content}
        
        Summary:"""
        
        try:
            response = requests.post(
                f"{CONFIG['ollama_api']['base_url']}/api/generate",
                json={
                    'model': CONFIG["models"]["summary_model"],
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.3,
                        'top_p': 0.9,
                        'max_tokens': 150
                    }
                },
                timeout=90
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                self.logger.error(f"Ollama API error: {response.status_code}")
                return article.content[:200] + "..."
                
        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
            return article.content[:200] + "..."

    def calculate_sentiment(self, article: Article) -> float:
        """Calculate sentiment score using multiple methods"""
        scores = []
        
        # NLTK VADER
        if self.sentiment_analyzer:
            vader_score = self.sentiment_analyzer.polarity_scores(article.content)
            scores.append(vader_score['compound'])
        
        # TextBlob
        try:
            blob = TextBlob(article.content)
            scores.append(blob.sentiment.polarity)
        except:
            pass
        
        return np.mean(scores) if scores else 0.0

    async def generate_embeddings(self, articles: List[Article]) -> List[Article]:
        """Generate embeddings for semantic analysis with improved numerical stability"""
        # Default embedding dimension for the model
        # nomic-embed-text typically uses 768 dimensions
        embedding_dimension = 768
        
        try:
            texts = [f"{article.title} {article.summary}" for article in articles]
            
            # First, get a sample embedding to determine the actual dimension
            if texts:
                try:
                    sample_response = requests.post(
                        f"{CONFIG['ollama_api']['base_url']}/api/embeddings",
                        json={
                            'model': CONFIG["models"]["embedding_model"],
                            'prompt': texts[0]
                        },
                        timeout=90
                    )
                    
                    if sample_response.status_code == 200:
                        # Get actual dimension from the first embedding
                        embedding_dimension = len(sample_response.json()['embedding'])
                        self.logger.info(f"Using embedding dimension: {embedding_dimension}")
                except Exception as e:
                    self.logger.warning(f"Could not determine embedding dimension: {e}")
            
            # Use Ollama embedding model with error handling
            embeddings = []
            valid_count = 0
            
            for i, text in enumerate(texts):
                try:
                    response = requests.post(
                        f"{CONFIG['ollama_api']['base_url']}/api/embeddings",
                        json={
                            'model': CONFIG["models"]["embedding_model"],
                            'prompt': text
                        },
                        timeout=90
                    )
                    
                    if response.status_code == 200:
                        embedding = response.json()['embedding']
                        
                        # Validate embedding - check for NaN or infinity values
                        if all(np.isfinite(x) for x in embedding) and not np.allclose(embedding, 0):
                            embeddings.append(embedding)
                            valid_count += 1
                        else:
                            self.logger.warning(f"Invalid embedding values for article {i+1}")
                            # Generate a small random embedding instead of zeros
                            np.random.seed(i)  # For reproducibility
                            random_embedding = np.random.normal(0, 0.01, embedding_dimension).tolist()
                            embeddings.append(random_embedding)
                    else:
                        self.logger.warning(f"Failed to get embedding for article {i+1}: {response.status_code}")
                        # Generate a small random embedding
                        np.random.seed(i)  # For reproducibility
                        random_embedding = np.random.normal(0, 0.01, embedding_dimension).tolist()
                        embeddings.append(random_embedding)
                except Exception as e:
                    self.logger.warning(f"Error generating embedding for article {i+1}: {e}")
                    # Generate a small random embedding
                    np.random.seed(i)  # For reproducibility
                    random_embedding = np.random.normal(0, 0.01, embedding_dimension).tolist()
                    embeddings.append(random_embedding)
                
            self.logger.info(f"Generated {valid_count} valid embeddings out of {len(texts)} articles")
            
            # Assign embeddings to articles
            for article, embedding in zip(articles, embeddings):
                article.embedding = embedding
                
        except Exception as e:
            self.logger.error(f"Embedding generation error: {e}")
            # Fallback to TF-IDF
            self.generate_tfidf_embeddings(articles)
        
        return articles

    def generate_tfidf_embeddings(self, articles: List[Article]):
        """Fallback TF-IDF embeddings"""
        texts = [f"{article.title} {article.summary}" for article in articles]
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        try:
            embeddings = vectorizer.fit_transform(texts).toarray()
            for article, embedding in zip(articles, embeddings):
                article.embedding = embedding.tolist()
        except Exception as e:
            self.logger.error(f"TF-IDF embedding error: {e}")

    def cluster_articles(self, articles: List[Article]) -> List[Article]:
        """Cluster articles by topic similarity with maximum numerical stability"""
        if len(articles) < 2:
            return articles
            
        # Filter out None embeddings and ensure numerical stability
        valid_articles = []
        valid_embeddings = []
        
        for article in articles:
            if article.embedding is None:
                article.cluster_id = -1  # Mark as unclustered
                continue
                
            # Convert to numpy for more stable operations
            embedding_np = np.array(article.embedding, dtype=np.float64)
            
            # Check for NaN or infinity values
            if not np.all(np.isfinite(embedding_np)):
                self.logger.warning(f"Invalid embedding values found for article: {article.title}")
                article.cluster_id = -1  # Mark as unclustered
                continue
                
            # Check for all zeros or very small values
            if np.allclose(embedding_np, 0, atol=1e-10):
                self.logger.warning(f"Zero embedding found for article: {article.title}")
                article.cluster_id = -1  # Mark as unclustered
                continue
                
            valid_articles.append(article)
            valid_embeddings.append(embedding_np)
            
        if len(valid_embeddings) < 2:
            # Not enough valid embeddings for clustering
            self.logger.warning("Not enough valid embeddings for clustering")
            return articles
            
        # Safe PCA to reduce dimensions and improve stability
        try:
            from sklearn.decomposition import PCA
            # Reduce dimensions to a more manageable size (e.g., 50)
            pca_dim = min(50, min(len(valid_embeddings) - 1, valid_embeddings[0].shape[0]))
            
            if pca_dim > 0:
                pca = PCA(n_components=pca_dim, random_state=42)
                reduced_embeddings = pca.fit_transform(valid_embeddings)
                self.logger.info(f"Reduced embedding dimension from {valid_embeddings[0].shape[0]} to {pca_dim} using PCA")
            else:
                # Can't do PCA with just one sample or one feature
                reduced_embeddings = valid_embeddings
        except Exception as e:
            self.logger.warning(f"PCA reduction failed: {e}. Using original embeddings.")
            reduced_embeddings = valid_embeddings
        
        # Extra safety: replace any remaining NaN/Inf with zeros
        reduced_embeddings = np.nan_to_num(reduced_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Normalize embeddings to unit length to improve numerical stability
        try:
            # Add a small epsilon to avoid division by zero
            norms = np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # Ensure no zeros
            normalized_embeddings = reduced_embeddings / norms
        except Exception as e:
            self.logger.warning(f"Normalization failed: {e}. Using non-normalized embeddings.")
            normalized_embeddings = reduced_embeddings
        
        try:
            n_clusters = min(CONFIG["processing"]["max_clusters"], len(valid_articles))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(normalized_embeddings)
            
            # Assign cluster IDs
            for article, label in zip(valid_articles, cluster_labels):
                article.cluster_id = int(label)
                
        except Exception as e:
            self.logger.error(f"Clustering error: {e}")
            # Assign default cluster to prevent further issues
            for i, article in enumerate(valid_articles):
                article.cluster_id = i % min(3, len(valid_articles))
        
        return articles

    def calculate_importance_scores(self, articles: List[Article]) -> List[Article]:
        """Calculate importance scores using multiple factors"""
        for article in articles:
            # Freshness score (newer articles get higher scores)
            hours_old = (datetime.now() - article.published).total_seconds() / 3600
            freshness = max(0, 1 - (hours_old / 24))  # Decay over 24 hours
            
            # Sentiment impact (stronger emotions = more important)
            sentiment_impact = abs(article.sentiment_score)
            
            # Content quality (length and readability)
            content_quality = min(1.0, len(article.content) / 1000)
            
            # Combine scores
            article.importance_score = (
                CONFIG["processing"]["freshness_weight"] * freshness +
                CONFIG["processing"]["sentiment_weight"] * sentiment_impact +
                CONFIG["processing"]["relevance_weight"] * content_quality
            )
        
        return articles

    def create_broadcast_segments(self, articles: List[Article]) -> List[BroadcastSegment]:
        """Create broadcast segments from clustered articles"""
        segments = []
        
        # Group articles by cluster
        clusters = {}
        for article in articles:
            cluster_id = article.cluster_id
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(article)
        
        # Create segments from clusters
        for cluster_id, cluster_articles in clusters.items():
            if not cluster_articles:
                continue
            
            # Sort by importance
            cluster_articles.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Determine topic from top articles
            topic = self.extract_topic(cluster_articles[:3])
            
            # Select representative articles
            selected_articles = cluster_articles[:2]  # Top 2 articles per cluster
            
            # Calculate segment importance
            avg_importance = np.mean([a.importance_score for a in selected_articles])
            
            segment = BroadcastSegment(
                topic=topic,
                content="",  # Will be generated later
                articles=selected_articles,
                importance=avg_importance,
                duration_estimate=30  # seconds
            )
            
            segments.append(segment)
        
        # Sort segments by importance
        segments.sort(key=lambda x: x.importance, reverse=True)
        
        # Limit to target number of segments
        return segments[:CONFIG["output"]["target_segments"]]

    def extract_topic(self, articles: List[Article]) -> str:
        """Extract main topic from article cluster"""
        # Simple approach: find common keywords in titles
        titles = [article.title for article in articles]
        
        if self.nlp:
            # Use spaCy for better topic extraction
            all_text = " ".join(titles)
            doc = self.nlp(all_text)
            
            # Extract entities and noun phrases
            entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']]
            noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
            
            if entities:
                return entities[0]
            elif noun_phrases:
                return noun_phrases[0]
        
        # Fallback: use first few words of most important article
        if articles:
            return articles[0].title.split()[:3]
        
        return "General News"

    async def generate_broadcast_script(self, segments: List[BroadcastSegment]) -> str:
        """Generate cohesive broadcast script"""
        script_parts = []
        
        # Opening
        script_parts.append(self.generate_opening())
        
        # Process each segment
        for i, segment in enumerate(segments):
            # Generate segment content
            segment_script = await self.generate_segment_script(segment)
            segment.content = segment_script
            
            script_parts.append(segment_script)
            
            # Add transition if not last segment
            if i < len(segments) - 1 and CONFIG["output"]["include_transitions"]:
                script_parts.append(self.generate_transition(segment, segments[i + 1]))
        
        # Closing
        script_parts.append(self.generate_closing())
        
        full_script = "\n\n".join(script_parts)
        
        # Trim to target length if needed
        if len(full_script) > CONFIG["output"]["max_broadcast_length"]:
            full_script = full_script[:CONFIG["output"]["max_broadcast_length"]] + "..."
        
        # Clean the script for TTS to remove any metadata, hashtags, or special formatting
        clean_script = self.clean_script_for_tts(full_script)
        
        return clean_script

    def generate_opening(self) -> str:
        """Generate broadcast opening"""
        current_time = datetime.now().strftime("%A, %B %d")
        return f"Today's news update for {current_time}. The latest developments from around the world."

    async def generate_segment_script(self, segment: BroadcastSegment) -> str:
        """Generate script for a specific segment"""
        # Prepare context from articles
        context = []
        for article in segment.articles:
            context.append(f"Title: {article.title}")
            context.append(f"Summary: {article.summary}")
            context.append(f"Source: {article.source}")
            context.append("---")
        
        context_text = "\n".join(context)
        
        prompt = f"""
        Create a professional news segment about "{segment.topic}" based on the following information.
        Write it as a news anchor would deliver it, in a conversational but authoritative tone.
        Keep it to 2-3 sentences and focus on the key points.
        IMPORTANT: Start directly with the news content. DO NOT include any prefixes, labels, or metadata.
        DO NOT start with phrases like "In today's news" or "Here's the latest on".
        Just provide the clean news text that should be read aloud.
        
        Context:
        {context_text}
        
        News Segment:"""
        
        try:
            response = requests.post(
                f"{CONFIG['ollama_api']['base_url']}/api/generate",
                json={
                    'model': CONFIG["models"]["broadcast_model"],
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.4,
                        'top_p': 0.9,
                        'max_tokens': 200
                    }
                },
                timeout=45
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                self.logger.error(f"Broadcast generation API error: {response.status_code}")
                return self.generate_fallback_segment(segment)
                
        except Exception as e:
            self.logger.error(f"Broadcast generation error: {e}")
            return self.generate_fallback_segment(segment)

    def generate_fallback_segment(self, segment: BroadcastSegment) -> str:
        """Generate fallback segment without LLM"""
        if segment.articles:
            article = segment.articles[0]
            return f"In {segment.topic} news, {article.title}. {article.summary[:150]}"
        return f"We're following developments in {segment.topic}."

    def generate_transition(self, current_segment: BroadcastSegment, next_segment: BroadcastSegment) -> str:
        """Generate transition between segments"""
        transitions = [
            f"Next, {next_segment.topic}.",
            f"In {next_segment.topic} news.",
            f"Regarding {next_segment.topic}."
        ]
        return np.random.choice(transitions)

    def generate_closing(self) -> str:
        """Generate broadcast closing"""
        return "End of news update."
        
    def clean_script_for_tts(self, script: str) -> str:
        """Clean script text to be suitable for TTS reading
        Removes hashtags, metadata, and special formatting"""
        # Remove hashtags
        clean_text = re.sub(r'#\w+', '', script)
        
        # Remove URLs
        clean_text = re.sub(r'https?://\S+', '', clean_text)
        
        # Remove special characters not meant to be read
        clean_text = re.sub(r'[*_~`|]', '', clean_text)
        
        # Remove markdown style headers
        clean_text = re.sub(r'^#+\s+', '', clean_text, flags=re.MULTILINE)
        
        # Remove any metadata in format like [meta: value]
        clean_text = re.sub(r'\[[^\]]+\]', '', clean_text)
        
        # Remove any parenthetical source references like (Source: CNN)
        clean_text = re.sub(r'\([Ss]ource:[^)]+\)', '', clean_text)
        
        # Remove excessive newlines
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
        
        # Final whitespace cleanup
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        clean_text = re.sub(r' +', ' ', clean_text)
        
        return clean_text

    async def generate_audio(self, script: str, output_file: str):
        """Generate audio using TTS with improved error handling and prioritization"""
        try:
            # Extra text cleaning for TTS-specific issues
            script = self._prepare_text_for_tts(script)
            
            # Try Chatterbox first as the preferred TTS engine
            try:
                # Get configuration values
                exaggeration = CONFIG["tts"]["exaggeration"]
                temperature = CONFIG["tts"]["temperature"]
                seed = CONFIG["tts"]["seed"]
                cfg_pace = CONFIG["tts"]["cfg_pace"]
                reference_audio = CONFIG["tts"]["reference_audio"]
                
                self.logger.info(f"Generating audio with Chatterbox: script length {len(script)} chars")
                
                # Create Gradio client for Chatterbox (using the Hugging Face space)
                client = Client(CONFIG["tts"]["chatterbox_url"])
                
                # Get API info to find the correct endpoint
                api_info = client.endpoints
                self.logger.info(f"Available Chatterbox APIs: {api_info}")
                
                # Find the text-to-speech endpoint
                api_name = None
                # Check if any endpoint has 'text_to_speech' or similar in the name
                for endpoint in api_info.values():
                    self.logger.info(f"Checking endpoint: {endpoint.api_name}")
                    if 'text' in endpoint.api_name.lower() or 'speech' in endpoint.api_name.lower() or 'tts' in endpoint.api_name.lower() or 'audio' in endpoint.api_name.lower():
                        api_name = endpoint.api_name
                        break
                
                # If no specific TTS endpoint found, use the first one
                if not api_name and api_info:
                    first_endpoint = next(iter(api_info.values()))
                    api_name = first_endpoint.api_name
                
                if not api_name:
                    raise ValueError("Could not find a valid API endpoint for Chatterbox")
                    
                # Prepare reference audio if provided
                audio_prompt = None
                if reference_audio:
                    audio_prompt = reference_audio
                
                # Break the text into chunks for more reliable processing
                # Chatterbox works best with shorter segments
                audio_chunks = []
                failed_chunks = []
                
                # Split script into sentences and make chunks smaller to avoid quota issues
                sentences = re.split(r'(?<=[.!?])\s+', script)
                
                # Process in smaller batches of sentences (just 1 sentence per chunk to minimize quota usage)
                batch_size = 1
                for i in range(0, len(sentences), batch_size):
                    chunk = " ".join(sentences[i:i+batch_size])
                    if not chunk.strip():
                        continue
                        
                    chunk_num = i//batch_size + 1
                    total_chunks = (len(sentences)-1)//batch_size + 1
                    self.logger.info(f"Processing chunk {chunk_num}/{total_chunks}, length: {len(chunk)} chars")
                    
                    # Add delay between API calls to avoid rate limiting
                    if i > 0:
                        await asyncio.sleep(1.0)  # Longer delay to respect API limits
                    
                    # Call Chatterbox API to generate audio for this chunk with retries
                    max_retries = 2  # Increased retries since Chatterbox is now primary
                    for retry in range(max_retries + 1):
                        try:
                            result = client.predict(
                                chunk,                      # text_input
                                audio_prompt,               # audio_prompt_path_input
                                exaggeration,               # exaggeration_input
                                temperature,                # temperature_input
                                seed,                       # seed_num_input
                                cfg_pace,                   # cfgw_input
                                api_name=api_name
                            )
                            
                            if isinstance(result, str):
                                audio_chunks.append(result)
                                break  # Success, exit retry loop
                            else:
                                self.logger.warning(f"Unexpected chunk result type: {type(result)}")
                        except Exception as chunk_error:
                            error_msg = str(chunk_error)
                            self.logger.warning(f"Error processing chunk {chunk_num}: {error_msg}")
                            
                            # Check for quota errors - if found, stop trying with Chatterbox
                            if "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                                self.logger.warning("Quota exceeded, stopping Chatterbox attempts")
                                raise ValueError("Chatterbox quota exceeded")
                            
                            # Otherwise just record the failure and continue
                            if retry == max_retries:
                                failed_chunks.append((chunk_num, chunk))
                            else:
                                self.logger.info(f"Retrying chunk {chunk_num} (attempt {retry+2}/{max_retries+1})")
                                await asyncio.sleep(1.5)  # Wait a bit before retrying
                
                # Combine audio chunks if any were generated
                if audio_chunks:
                    self.logger.info(f"Successfully generated {len(audio_chunks)}/{total_chunks} audio chunks")
                    
                    if failed_chunks:
                        self.logger.warning(f"Failed to generate {len(failed_chunks)} chunks due to errors/quotas")
                        
                    # Only attempt to combine if we have at least one chunk
                    combined_audio = self._combine_audio_files(audio_chunks, output_file)
                    self.logger.info(f"Audio generated with Chatterbox: {output_file}")
                    return
                else:
                    raise ValueError("No audio chunks were successfully generated")
                    
            except Exception as e:
                self.logger.warning(f"Chatterbox failed: {e}. Trying Edge TTS as fallback.")
                
            # Fall back to Edge TTS if Chatterbox fails
            if edge_tts_available:
                try:
                    voice = "en-US-JennyNeural"  # Default voice - one of the best quality voices
                    
                    self.logger.info(f"Generating audio with Edge TTS: script length {len(script)} chars")
                    communicate = edge_tts.Communicate(script, voice)
                    await communicate.save(output_file)
                    self.logger.info(f"Audio generated with Edge TTS: {output_file}")
                    return
                except Exception as e:
                    self.logger.error(f"Edge TTS also failed: {e}")
                    raise Exception("Both Chatterbox and Edge TTS failed to generate audio")
            else:
                self.logger.error("Chatterbox failed and Edge TTS is not available")
                raise Exception("No text-to-speech engines available")
                
        except Exception as e:
            self.logger.error(f"Audio generation error: {e}")
            raise
            
    def _prepare_text_for_tts(self, text: str) -> str:
        """Additional preparation for TTS to avoid issues"""
        # Remove any quotes which can cause problems
        text = text.replace('"', '').replace('"', '').replace('"', '')
        
        # Replace special characters with their spelled-out versions
        text = text.replace('%', ' percent ')
        text = text.replace('&', ' and ')
        text = text.replace('/', ' or ')
        text = text.replace('$', ' dollars ')
        text = text.replace('#', ' number ')
        
        # Remove ellipses which can cause pauses
        text = text.replace('...', '.')
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def _combine_audio_files(self, audio_paths: List[str], output_path: str) -> str:
        """Combine multiple audio files into one with improved error handling"""
        # If there's only one path, just copy it
        if len(audio_paths) == 1:
            import shutil
            shutil.copy(audio_paths[0], output_path)
            return output_path
            
        # First try using pydub
        try:
            # Check if pydub is available without forcing a reload
            import importlib.util
            pydub_spec = importlib.util.find_spec("pydub")
            if pydub_spec is not None:
                import pydub
                from pydub import AudioSegment
                
                self.logger.info("Using pydub to combine audio files")
                
                # Load the first segment
                combined = AudioSegment.from_file(audio_paths[0])
                
                # Add the rest
                for path in audio_paths[1:]:
                    try:
                        segment = AudioSegment.from_file(path)
                        combined += segment
                    except Exception as e:
                        self.logger.warning(f"Error adding segment {path}: {e}. Skipping this segment.")
                
                # Export
                combined.export(output_path, format="mp3")
                return output_path
            else:
                self.logger.warning("pydub not available, skipping to ffmpeg method")
                
        except Exception as e:
            self.logger.warning(f"Error using pydub: {e}. Trying ffmpeg method.")
            
        # Fallback: use ffmpeg directly if available
        try:
            import subprocess
            import os
            import tempfile
            
            # Create a temporary file list for ffmpeg in the correct format
            temp_file_path = os.path.join(tempfile.gettempdir(), "audio_file_list.txt")
            with open(temp_file_path, "w") as f:
                for path in audio_paths:
                    # Use absolute paths with proper escaping
                    abs_path = os.path.abspath(path)
                    # Format according to ffmpeg concat requirements (escape single quotes)
                    formatted_path = abs_path.replace("'", "'\\''")
                    f.write(f"file '{formatted_path}'\n")
            
            # Run ffmpeg with improved parameters
            self.logger.info("Using ffmpeg to combine audio files")
            try:
                result = subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", temp_file_path, 
                     "-c", "copy", output_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Clean up
                os.remove(temp_file_path)
                return output_path
                
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"ffmpeg command failed: {e.stderr}")
                
                # Try alternative ffmpeg approach with manual concatenation
                try:
                    self.logger.info("Trying alternative ffmpeg approach")
                    # First convert all files to the same format
                    temp_files = []
                    
                    for i, path in enumerate(audio_paths):
                        temp_out = os.path.join(tempfile.gettempdir(), f"temp_audio_{i}.wav")
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", path, "-acodec", "pcm_s16le", 
                             "-ar", "44100", "-ac", "2", temp_out],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        temp_files.append(temp_out)
                    
                    # Then concatenate using the concat filter
                    filter_complex = "|".join([f"[{i}:0]" for i in range(len(temp_files))])
                    filter_complex += f"concat=n={len(temp_files)}:v=0:a=1[out]"
                    
                    inputs = []
                    for temp in temp_files:
                        inputs.extend(["-i", temp])
                    
                    subprocess.run(
                        ["ffmpeg", "-y"] + inputs + 
                        ["-filter_complex", filter_complex, "-map", "[out]", output_path],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Clean up temp files
                    for temp in temp_files:
                        os.remove(temp)
                    
                    os.remove(temp_file_path)
                    return output_path
                    
                except subprocess.CalledProcessError as e2:
                    self.logger.warning(f"Alternative ffmpeg approach failed: {e2}")
                    # Continue to manual concatenation
            
        except Exception as e:
            self.logger.warning(f"Error with ffmpeg method: {e}. Falling back to manual concatenation.")
        
        # Last resort: manually concatenate the WAV files
        # This assumes they have the same format and can be directly concatenated
        try:
            self.logger.info("Using manual binary concatenation as last resort")
            import os
            import shutil
            
            # Create a temporary output file first to avoid corrupting the original in case of errors
            temp_output = f"{output_path}.temp"
            
            with open(temp_output, 'wb') as outfile:
                # Check if the first file exists and is readable
                if not os.path.exists(audio_paths[0]) or not os.access(audio_paths[0], os.R_OK):
                    raise IOError(f"Cannot read first audio file: {audio_paths[0]}")
                    
                # Write the header from the first file
                with open(audio_paths[0], 'rb') as infile:
                    # Use a reasonable header size (44 bytes is standard for WAV)
                    header = infile.read(44)
                    outfile.write(header)
                    data = infile.read()
                    outfile.write(data)
                
                # Append data from the rest of the files (skipping headers)
                for path in audio_paths[1:]:
                    if not os.path.exists(path) or not os.access(path, os.R_OK):
                        self.logger.warning(f"Cannot read audio file: {path}, skipping")
                        continue
                        
                    with open(path, 'rb') as infile:
                        infile.seek(44)  # Skip header
                        data = infile.read()
                        outfile.write(data)
            
            # If successful, move the temp file to the final output
            shutil.move(temp_output, output_path)
            return output_path
            
        except Exception as e:
            self.logger.error(f"All audio combination methods failed: {e}. Copying first chunk only.")
            import shutil
            shutil.copy(audio_paths[0], output_path)
            return output_path

    def save_markdown(self, script: str, segments: List[BroadcastSegment], output_file: str):
        """Save enhanced markdown with metadata"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        markdown_content = f"""# News Broadcast
**Generated:** {timestamp}  
**Articles Processed:** {sum(len(s.articles) for s in segments)}  
**Segments:** {len(segments)}

---

## Broadcast Script

{script}

---

## Article Sources

"""
        
        for i, segment in enumerate(segments, 1):
            markdown_content += f"\n### Segment {i}: {segment.topic}\n"
            markdown_content += f"**Importance Score:** {segment.importance:.2f}\n\n"
            
            for j, article in enumerate(segment.articles, 1):
                markdown_content += f"{j}. **{article.title}**\n"
                markdown_content += f"   - Source: {article.source}\n"
                markdown_content += f"   - Published: {article.published.strftime('%Y-%m-%d %H:%M')}\n"
                markdown_content += f"   - Sentiment: {article.sentiment_score:.2f}\n"
                markdown_content += f"   - URL: {article.url}\n\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        self.logger.info(f"Markdown saved: {output_file}")

    async def run(self):
        """Main execution pipeline"""
        start_time = datetime.now()
        self.logger.info("Starting advanced news generation pipeline")
        
        try:
            # Fetch articles
            self.logger.info("Fetching articles from RSS feeds...")
            articles = await self.fetch_feeds_async()
            self.logger.info(f"Fetched {len(articles)} articles")
            
            if not articles:
                self.logger.warning("No articles found")
                return
            
            # Process articles with advanced algorithms
            self.logger.info("Processing articles with advanced algorithms...")
            processed_articles = await self.process_articles_advanced(articles)
            
            # Create broadcast segments
            self.logger.info("Creating broadcast segments...")
            segments = self.create_broadcast_segments(processed_articles)
            self.logger.info(f"Created {len(segments)} segments")
            
            # Generate broadcast script
            self.logger.info("Generating broadcast script...")
            script = await self.generate_broadcast_script(segments)
            
            # Generate output files
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            md_file = f"digest_{timestamp}.md"
            mp3_file = f"digest_{timestamp}.mp3"
            
            # Save markdown
            self.logger.info("Saving markdown file...")
            self.save_markdown(script, segments, md_file)
            
            # Generate audio
            self.logger.info("Generating audio file...")
            await self.generate_audio(script, mp3_file)
            
            # Performance metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Pipeline completed in {duration:.2f} seconds")
            self.logger.info(f"Generated files: {md_file}, {mp3_file}")
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            raise

def main():
    """Main entry point"""
    generator = AdvancedNewsGenerator()
    asyncio.run(generator.run())

if __name__ == "__main__":
    main()
