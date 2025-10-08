import requests
from bs4 import BeautifulSoup
import json
import re
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
import time
from collections import defaultdict, Counter
import pickle
import os
import math
class AdvancedDocumentationScraper:
    """Enhanced scraper with better content extraction."""
    
    def __init__(self, base_url: str = "https://docs.capillarytech.com/"):
        self.base_url = base_url
        self.visited_urls = set()
        self.knowledge_base = []
        self.max_pages = 100
        self.delay = 1
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to documentation domain."""
        parsed = urlparse(url)
        base_parsed = urlparse(self.base_url)
        return parsed.netloc == base_parsed.netloc
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def extract_structured_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract structured content with enhanced parsing."""
        content = {
            'url': url,
            'title': '',
            'description': '',
            'sections': [],
            'code_examples': [],
            'api_endpoints': [],
            'full_text': '',
            'keywords': []
        }
        
        # Extract title
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            content['title'] = self.clean_text(title_tag.get_text())
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            content['description'] = self.clean_text(meta_desc['content'])
        
        # Extract sections with headings
        current_section = None
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol']):
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                # Start new section
                if current_section:
                    content['sections'].append(current_section)
                current_section = {
                    'heading': self.clean_text(element.get_text()),
                    'level': element.name,
                    'content': []
                }
            elif current_section:
                text = self.clean_text(element.get_text())
                if len(text) > 20:
                    current_section['content'].append(text)
        
        if current_section:
            content['sections'].append(current_section)
        
        # Extract code examples
        for code_block in soup.find_all(['pre', 'code']):
            code_text = code_block.get_text(strip=True)
            if len(code_text) > 15 and code_text not in content['code_examples']:
                content['code_examples'].append(code_text)
        
        # Extract API endpoints (URLs in text)
        text_content = soup.get_text()
        api_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+/api[^\s<>"{}|\\^`\[\]]*'
        endpoints = re.findall(api_pattern, text_content)
        content['api_endpoints'] = list(set(endpoints))
        
        # Build full searchable text
        full_text_parts = [content['title'], content['description']]
        for section in content['sections']:
            full_text_parts.append(section['heading'])
            full_text_parts.extend(section['content'])
        content['full_text'] = ' '.join(full_text_parts)
        
        # Extract keywords from title and headings
        keywords = []
        for section in content['sections']:
            keywords.extend(self._extract_keywords(section['heading']))
        content['keywords'] = list(set(keywords))
        
        return content
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Common technical terms that should be preserved
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 
                    'at', 'to', 'for', 'of', 'with', 'by', 'from'}
        return [w for w in words if len(w) > 3 and w not in stopwords]
    
    def scrape_page(self, url: str) -> Optional[Dict]:
        """Scrape a single page with enhanced extraction."""
        if url in self.visited_urls or len(self.visited_urls) >= self.max_pages:
            return None
        
        try:
            print(f"📄 Scraping: {url}")
            response = requests.get(url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Educational Research Bot)'
            })
            response.raise_for_status()
            
            self.visited_urls.add(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer']):
                script.decompose()
            
            content = self.extract_structured_content(soup, url)
            
            # Find new links
            links = []
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                    links.append(full_url)
            
            time.sleep(self.delay)
            return content, links
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {str(e)}")
            return None, []
    
    def scrape_documentation(self, start_url: Optional[str] = None):
        """Scrape documentation with progress tracking."""
        if start_url is None:
            start_url = self.base_url
        
        to_visit = [start_url]
        
        print(f"\n🚀 Starting documentation scrape...")
        print(f"   Target: {self.max_pages} pages\n")
        
        while to_visit and len(self.visited_urls) < self.max_pages:
            current_url = to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
            
            result = self.scrape_page(current_url)
            
            if result:
                content, new_links = result
                if content and content['full_text']:
                    self.knowledge_base.append(content)
                    to_visit.extend(new_links[:5])  # Add limited new links
            
            # Progress indicator
            if len(self.visited_urls) % 10 == 0:
                print(f"   Progress: {len(self.visited_urls)}/{self.max_pages} pages")
        
        print(f"\n✅ Scraping complete! Collected {len(self.knowledge_base)} pages.\n")
    
    def save_knowledge_base(self, filename: str = "capillary_advanced_kb.pkl"):
        """Save scraped data to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.knowledge_base, f)
        print(f"💾 Knowledge base saved to {filename}")
    
    def load_knowledge_base(self, filename: str = "capillary_advanced_kb.pkl") -> bool:
        """Load scraped data from file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.knowledge_base = pickle.load(f)
            print(f"✅ Loaded {len(self.knowledge_base)} pages from cache")
            return True
        return False


class TFIDFRanker:
    """TF-IDF based document ranking system."""
    
    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.vocab = set()
        self.idf_scores = {}
        self.doc_vectors = []
        self._build_tfidf()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 
                    'at', 'to', 'for', 'of', 'with', 'by', 'from', 'this', 'that'}
        return [w for w in words if len(w) > 2 and w not in stopwords]
    
    def _build_tfidf(self):
        """Build TF-IDF vectors for all documents."""
        print("🔨 Building TF-IDF index...")
        
        # Build vocabulary and document frequency
        doc_freq = Counter()
        doc_terms = []
        
        for doc in self.documents:
            terms = self._tokenize(doc['full_text'])
            doc_terms.append(terms)
            unique_terms = set(terms)
            self.vocab.update(unique_terms)
            for term in unique_terms:
                doc_freq[term] += 1
        
        # Calculate IDF scores
        num_docs = len(self.documents)
        for term in self.vocab:
            self.idf_scores[term] = math.log(num_docs / (1 + doc_freq[term]))
        
        # Build TF-IDF vectors
        for terms in doc_terms:
            term_freq = Counter(terms)
            tfidf_vector = {}
            for term, freq in term_freq.items():
                tf = freq / len(terms) if terms else 0
                tfidf_vector[term] = tf * self.idf_scores.get(term, 0)
            self.doc_vectors.append(tfidf_vector)
        
        print(f"   Vocabulary size: {len(self.vocab)}")
        print(f"   Documents indexed: {len(self.doc_vectors)}\n")
    
    def rank_documents(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Rank documents by relevance to query using cosine similarity."""
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return []
        
        # Build query vector
        query_freq = Counter(query_terms)
        query_vector = {}
        for term, freq in query_freq.items():
            tf = freq / len(query_terms)
            query_vector[term] = tf * self.idf_scores.get(term, 0)
        
        # Calculate cosine similarity with each document
        scores = []
        for idx, doc_vector in enumerate(self.doc_vectors):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                scores.append((idx, similarity))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _cosine_similarity(self, vec1: Dict, vec2: Dict) -> float:
        """Calculate cosine similarity between two vectors."""
        # Dot product
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) 
                         for term in set(vec1.keys()) | set(vec2.keys()))
        
        # Magnitudes
        mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        return dot_product / (mag1 * mag2)


class AdvancedCapillaryChatbot:
    """Advanced chatbot with TF-IDF ranking and smart responses."""
    
    def __init__(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base
        self.ranker = TFIDFRanker(knowledge_base)
    
    def _extract_answer_snippet(self, doc: Dict, query: str, max_length: int = 300) -> str:
        """Extract the most relevant snippet from document."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        best_snippet = ""
        max_overlap = 0
        
        # Check each section
        for section in doc['sections']:
            section_text = ' '.join(section['content'])
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', section_text)
            
            for i, sentence in enumerate(sentences):
                # Create window of 2-3 sentences
                window = ' '.join(sentences[i:min(i+3, len(sentences))])
                if len(window) > max_length:
                    window = window[:max_length] + "..."
                
                # Count query word overlap
                window_words = set(re.findall(r'\b\w+\b', window.lower()))
                overlap = len(query_words & window_words)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_snippet = window
        
        return best_snippet if best_snippet else doc['description']
    
    def answer_query(self, query: str, detailed: bool = False) -> str:
        """Generate intelligent answer to user query."""
        # Get top relevant documents
        results = self.ranker.rank_documents(query, top_k=3)
        
        if not results:
            return self._handle_no_results(query)
        
        # Build response
        response_parts = []
        
        # Check if query is about API
        is_api_query = any(word in query.lower() for word in ['api', 'endpoint', 'request', 'authentication'])
        
        response_parts.append("📚 **Answer from CapillaryTech Documentation:**\n")
        
        for rank, (doc_idx, score) in enumerate(results, 1):
            doc = self.knowledge_base[doc_idx]
            
            # Skip low relevance results
            if score < 0.1:
                continue
            
            response_parts.append(f"\n{'='*60}")
            response_parts.append(f"📄 **{doc['title']}**")
            response_parts.append(f"🔗 Source: {doc['url']}")
            response_parts.append(f"📊 Relevance: {score:.2%}\n")
            
            # Add description if available
            if doc['description']:
                response_parts.append(f"📝 {doc['description']}\n")
            
            # Extract relevant snippet
            snippet = self._extract_answer_snippet(doc, query)
            if snippet:
                response_parts.append(f"💡 **Key Information:**")
                response_parts.append(f"{snippet}\n")
            
            # Add API endpoints if query is API-related
            if is_api_query and doc['api_endpoints']:
                response_parts.append(f"🔌 **API Endpoints:**")
                for endpoint in doc['api_endpoints'][:3]:
                    response_parts.append(f"   • {endpoint}")
                response_parts.append("")
            
            # Add code example if available
            if doc['code_examples'] and (detailed or rank == 1):
                response_parts.append(f"💻 **Code Example:**")
                response_parts.append(f"```")
                response_parts.append(doc['code_examples'][0][:200])
                if len(doc['code_examples'][0]) > 200:
                    response_parts.append("...")
                response_parts.append(f"```\n")
            
            # Only show top result in detail unless detailed mode
            if not detailed and rank == 1:
                break
        
        # Add suggestion for more info
        response_parts.append(f"\n{'='*60}")
        response_parts.append("💡 Tip: Visit the source URLs for complete documentation")
        
        return '\n'.join(response_parts)
    
    def _handle_no_results(self, query: str) -> str:
        """Provide helpful response when no results found."""
        suggestions = []
        
        # Suggest available topics
        topics = self.get_topics()
        if topics:
            suggestions.append("\n📚 Available topics in the documentation:")
            for topic in topics[:10]:
                suggestions.append(f"   • {topic}")
        
        return (
            f"❌ I couldn't find specific information about '{query}' "
            f"in the CapillaryTech documentation.\n"
            + '\n'.join(suggestions) +
            "\n\n💡 Try rephrasing your question or use the 'topics' command."
        )
    
    def get_topics(self) -> List[str]:
        """Get list of available topics."""
        topics = []
        for doc in self.knowledge_base:
            if doc['title'] and len(doc['title']) > 5:
                topics.append(doc['title'])
        return sorted(set(topics))
    
    def get_statistics(self) -> Dict:
        """Get chatbot statistics."""
        total_sections = sum(len(doc['sections']) for doc in self.knowledge_base)
        total_code = sum(len(doc['code_examples']) for doc in self.knowledge_base)
        total_apis = sum(len(doc['api_endpoints']) for doc in self.knowledge_base)
        
        return {
            'total_documents': len(self.knowledge_base),
            'total_sections': total_sections,
            'code_examples': total_code,
            'api_endpoints': total_apis,
            'vocabulary_size': len(self.ranker.vocab)
        }
    
    def interactive_chat(self):
        """Start interactive chat session."""
        print("\n" + "="*70)
        print("  🤖 CapillaryTech Documentation Chatbot (Advanced Edition)")
        print("="*70)
        
        # Show statistics
        stats = self.get_statistics()
        print(f"\n📊 Knowledge Base Statistics:")
        print(f"   • Documents: {stats['total_documents']}")
        print(f"   • Sections: {stats['total_sections']}")
        print(f"   • Code Examples: {stats['code_examples']}")
        print(f"   • Vocabulary: {stats['vocabulary_size']:,} terms")
        
        print("\n💬 I'm ready to answer your questions about CapillaryTech!")
        print("\n📝 Commands:")
        print("   • Type your question to search")
        print("   • 'topics' - List available topics")
        print("   • 'stats' - Show knowledge base statistics")
        print("   • 'help' - Show this help message")
        print("   • 'quit' or 'exit' - Exit chatbot")
        print("\n" + "="*70 + "\n")
        
        while True:
            try:
                user_input = input("🤔 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n👋 Thank you for using the CapillaryTech Chatbot! Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\n📖 Help:")
                    print("   • Ask questions about CapillaryTech documentation")
                    print("   • Use keywords like 'API', 'authentication', 'loyalty', etc.")
                    print("   • Type 'topics' to see available topics")
                    print("   • Type 'stats' for knowledge base statistics")
                    continue
                
                if user_input.lower() == 'topics':
                    topics = self.get_topics()
                    print(f"\n📚 Available Topics ({len(topics)}):\n")
                    for i, topic in enumerate(topics[:30], 1):
                        print(f"   {i:2d}. {topic}")
                    if len(topics) > 30:
                        print(f"\n   ... and {len(topics) - 30} more topics")
                    continue
                
                if user_input.lower() == 'stats':
                    stats = self.get_statistics()
                    print(f"\n📊 Knowledge Base Statistics:")
                    print(f"   • Total Documents: {stats['total_documents']}")
                    print(f"   • Total Sections: {stats['total_sections']}")
                    print(f"   • Code Examples: {stats['code_examples']}")
                    print(f"   • API Endpoints: {stats['api_endpoints']}")
                    print(f"   • Vocabulary Size: {stats['vocabulary_size']:,} terms")
                    continue
                
                # Generate answer
                print("\n🤖 Bot:\n")
                response = self.answer_query(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("   Please try rephrasing your question.")


def main():
    """Main function to run the advanced chatbot."""
    print("\n" + "="*70)
    print("  🚀 CapillaryTech Documentation Chatbot - Advanced Edition")
    print("="*70 + "\n")
    
    # Initialize scraper
    scraper = AdvancedDocumentationScraper()
    
    # Try to load existing knowledge base
    if not scraper.load_knowledge_base():
        print("📥 No cached data found. Starting documentation scrape...")
        print("⏱️  This will take approximately 2-5 minutes...\n")
        
        # Scrape documentation
        scraper.scrape_documentation()
        
        # Save for future use
        if scraper.knowledge_base:
            scraper.save_knowledge_base()
        else:
            print("\n❌ Error: No data collected. Please check your internet connection.")
            return
    
    if not scraper.knowledge_base:
        print("❌ Error: Knowledge base is empty.")
        return
    
    # Initialize advanced chatbot
    print("\n🔧 Initializing chatbot...\n")
    chatbot = AdvancedCapillaryChatbot(scraper.knowledge_base)
    
    # Start interactive session
    chatbot.interactive_chat()


if __name__ == "__main__":
    main()