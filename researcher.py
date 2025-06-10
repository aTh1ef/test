import streamlit as st
import requests
import json
from typing import Dict, List, Any, TypedDict, Annotated
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import operator
from typing import Optional
import time
import random
import logging

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Configuration
@dataclass
class Config:
    TAVILY_API_KEY: str = ""
    LM_STUDIO_URL: str = "http://127.0.0.1:1234/v1"
    LM_STUDIO_MODEL: str = "google/gemma-3-1b"
    REQUEST_TIMEOUT: int = 180
    MAX_RETRIES: int = 3

# State definition for LangGraph
class ResearchState(TypedDict):
    topic: str
    research_plan: str
    search_results: str
    analysis: str
    final_report: str
    messages: Annotated[List[Any], operator.add]
    next_step: str
    error: Optional[str]

class LMStudioLLM(LLM):
    """Custom LLM wrapper for LM Studio with improved error handling and dynamic loading"""
    
    base_url: str = "http://localhost:1234/v1"
    model: str = "google/gemma-3-1b"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 180
    max_retries: int = 3
    _last_used: datetime = None
    _idle_timeout: timedelta = timedelta(minutes=10)
    
    @property
    def _llm_type(self) -> str:
        return "lm_studio"
    
    def _should_unload(self) -> bool:
        """Check if the model should be unloaded due to inactivity."""
        if self._last_used is None:
            return False
        return datetime.now() - self._last_used > self._idle_timeout
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LM Studio API with retry logic and dynamic loading"""
        
        # Update last used time
        self._last_used = datetime.now()
        
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    logger.warning(f"Attempt {attempt + 1} failed with status {response.status_code}")
                    if attempt == self.max_retries - 1:
                        raise Exception(f"Failed after {self.max_retries} attempts. Status: {response.status_code}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("Failed to get response from LM Studio")

class TavilySearchTool(BaseTool):
    """Tavily AI-powered search tool"""
    
    name: str = "tavily_search"
    description: str = "Search the web using Tavily AI search API for current information"
    api_key: str = ""
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        logger.info("TavilySearchTool initialized with API key length: %d", len(api_key) if api_key else 0)
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """Execute search using Tavily API"""
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",  # Changed to basic for faster results
            "include_answer": True,
            "max_results": max_results
        }
        
        try:
            logger.info("Making Tavily API request for query: %s", query)
            response = requests.post(url, json=payload, timeout=30)
            
            logger.info("Tavily API response status: %d", response.status_code)
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Successfully got Tavily search results")
                
                # Create a clean, professional report format
                formatted_output = f"""
## Search Results for: "{query}"

### Summary
{data.get("answer", "No summary available")}

### Sources
"""
                
                sources = data.get("results", [])
                if sources:
                    for i, source in enumerate(sources, 1):
                        formatted_output += f"""
**{i}. {source.get("title", "Untitled Source")}**
- **URL:** {source.get("url", "#")}
- **Content:** {source.get("content", "No content available")[:300]}...

---
"""
                
                return formatted_output
            else:
                error_msg = f"Tavily API Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Search Error: {str(e)}"
            logger.error(error_msg)
            return error_msg

class WebScraperTool(BaseTool):
    """Tool for scraping web content"""
    
    name: str = "web_scraper"
    description: str = "Scrape content from web pages for detailed analysis"
    
    def _run(self, url: str) -> str:
        """Scrape content from a given URL"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Use BeautifulSoup if available, otherwise return raw text
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:10000]  # Limit content length
                
            except ImportError:
                # Fallback to raw text if BeautifulSoup is not available
                return response.text[:10000]
                
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return f"Scraping Error: {str(e)}"

class ResearchGraph:
    """Research system using LangGraph for orchestration"""
    
    def __init__(
        self,
        tavily_api_key: str,
        lm_studio_url: str = "http://localhost:1234/v1",
        model_name: str = "google/gemma-3-1b"
    ):
        self.tavily_api_key = tavily_api_key
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.llm = None
        self._last_used = None
        self._idle_timeout = timedelta(minutes=10)
    
    def _get_llm(self) -> LMStudioLLM:
        """Get or create LLM instance with dynamic loading."""
        current_time = datetime.now()
        
        # Check if we need to create a new instance
        if self.llm is None or (
            self._last_used is not None 
            and current_time - self._last_used > self._idle_timeout
        ):
            self.llm = LMStudioLLM(
                base_url=self.lm_studio_url,
                model=self.model_name
            )
        
        self._last_used = current_time
        return self.llm
    
    def research_topic(self, topic: str) -> Dict[str, str]:
        """Research a topic using the graph-based system"""
        try:
            # Get LLM instance
            llm = self._get_llm()
            
            # Create Tavily search tool
            search_tool = TavilySearchTool(api_key=self.tavily_api_key)
            
            # Create research plan
            research_plan = self._create_research_plan(topic, llm)
            
            # Execute search queries
            search_results = self._execute_searches(topic, search_tool)
            
            # Analyze results
            analysis = self._analyze_results(search_results, llm)
            
            # Generate final report
            final_report = self._generate_report(topic, analysis, llm)
            
            return {
                "research_plan": research_plan,
                "search_results": search_results,
                "analysis": analysis,
                "final_report": final_report
            }
            
        except Exception as e:
            logger.error(f"Error in research process: {str(e)}")
            return {
                "error": f"Research failed: {str(e)}",
                "research_plan": "",
                "search_results": "",
                "analysis": "",
                "final_report": ""
            }
    
    def _create_research_plan(self, topic: str, llm: LMStudioLLM) -> str:
        """Create a research plan for the topic"""
        prompt = f"""Create a detailed research plan for investigating: {topic}
        
        Include:
        1. Key questions to answer
        2. Specific areas to investigate
        3. Types of sources to consult
        4. Potential challenges to address
        
        Format the plan in markdown."""
        
        try:
            return llm(prompt)
        except Exception as e:
            logger.error(f"Error creating research plan: {str(e)}")
            return "Error: Could not create research plan"
    
    def _execute_searches(self, topic: str, search_tool: TavilySearchTool) -> str:
        """Execute search queries and compile results"""
        try:
            # Execute search and get raw response
            raw_response = search_tool._run(topic)
            
            # Since _run already returns formatted markdown, we can return it directly
            if raw_response and not raw_response.startswith("Error:"):
                return raw_response
            else:
                logger.error(f"Search failed: {raw_response}")
                return "Error: Search failed to return valid results"
            
        except Exception as e:
            logger.error(f"Error executing searches: {str(e)}")
            return "Error: Search failed"
    
    def _analyze_results(self, search_results: str, llm: LMStudioLLM) -> str:
        """Analyze search results"""
        prompt = f"""Analyze these search results and identify:
        
        1. Key findings and insights
        2. Common themes
        3. Conflicting information
        4. Areas needing more research
        
        Search results:
        {search_results}
        
        Provide analysis in markdown format."""
        
        try:
            return llm(prompt)
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            return "Error: Analysis failed"
    
    def _generate_report(self, topic: str, analysis: str, llm: LMStudioLLM) -> str:
        """Generate final research report"""
        prompt = f"""Create a comprehensive research report on: {topic}
        
        Using this analysis:
        {analysis}
        
        Include:
        1. Executive summary
        2. Key findings
        3. Supporting evidence
        4. Conclusions
        5. Recommendations for further research
        
        Format in markdown with clear sections."""
        
        try:
            return llm(prompt)
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return "Error: Report generation failed"

def test_tavily_api(api_key: str) -> bool:
    """Test if Tavily API is working."""
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": "test query",
        "search_depth": "basic",
        "include_answer": True,
        "max_results": 1
    }
    try:
        logger.info("Testing Tavily API connection...")
        response = requests.post(url, json=payload, timeout=10)
        logger.info(f"Tavily API test response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Tavily API test error: {response.text}")
            return False
        return True
    except Exception as e:
        logger.error(f"Tavily API test failed: {str(e)}")
        return False

def test_connections(tavily_key: str, lm_studio_url: str, model_name: str) -> Dict[str, bool]:
    """Test API connections"""
    results = {"tavily": False, "lm_studio": False}
    
    # Test Tavily API
    if tavily_key:
        results["tavily"] = test_tavily_api(tavily_key)
    
    # Test LM Studio
    try:
        test_llm = LMStudioLLM(base_url=lm_studio_url, model=model_name, timeout=30)
        test_response = test_llm("Say 'Hello'")
        results["lm_studio"] = "Error:" not in test_response and "timed out" not in test_response.lower()
    except:
        results["lm_studio"] = False
    
    return results

def main():
    # Page configuration
    st.set_page_config(
        page_title="Research Intelligence Hub",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Clean, professional CSS
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #fafafa;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
    }
    
    /* Status indicators */
    .status-success {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .status-error {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .status-warning {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    
    .sidebar-section h3 {
        color: #374151;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        margin-top: 0;
    }
    
    /* Progress indicator */
    .progress-step {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: #f3f4f6;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .progress-step.active {
        background: #ede9fe;
        border-left-color: #8b5cf6;
    }
    
    .progress-step.complete {
        background: #d1fae5;
        border-left-color: #10b981;
    }
    
    /* Clean button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Remove default streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Research Intelligence Hub</h1>
        <p>AI-Powered Research & Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3>🔑 API Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        tavily_key = st.text_input(
            "Tavily API Key",
            type="password",
            placeholder="Enter your Tavily API key...",
            help="Get your API key from tavily.com"
        )
        
        st.markdown("""
        <div class="sidebar-section">
            <h3>🤖 Model Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        lm_studio_url = st.text_input(
            "LM Studio URL",
            value="http://localhost:1234/v1",
            help="Local LM Studio endpoint"
        )
        
        model_name = st.text_input(
            "Model Name",
            value="google/gemma-3-1b",
            help="Model identifier in LM Studio"
        )
        
        # Connection status
        if st.button("Test Connections", use_container_width=True):
            with st.spinner("Testing connections..."):
                status = test_connections(tavily_key, lm_studio_url, model_name)
                
                if status["tavily"]:
                    st.markdown('<div class="status-success">✅ Tavily API Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-error">❌ Tavily API Failed</div>', unsafe_allow_html=True)
                
                if status["lm_studio"]:
                    st.markdown('<div class="status-success">✅ LM Studio Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-error">❌ LM Studio Failed</div>', unsafe_allow_html=True)
    
    # Main content area
    col1 = st.columns(1)[0]
    
    with col1:
        st.markdown("""
        <div class="card">
            <h2 style="margin-top: 0; color: #374151;">Research Query</h2>
        </div>
        """, unsafe_allow_html=True)
        
        topic = st.text_area(
            "Enter your research topic or question:",
            placeholder="Example: What are the latest developments in quantum computing applications for cryptography?",
            height=120,
            help="Be specific and detailed for better research results"
        )
        
        if st.button("🚀 Start Research", use_container_width=True, type="primary"):
            if not topic.strip():
                st.error("Please enter a research topic.")
            elif not tavily_key:
                st.error("Please provide a Tavily API key in the sidebar.")
            else:
                # Research execution
                with st.spinner("Conducting research..."):
                    try:
                        research_graph = ResearchGraph(tavily_key, lm_studio_url, model_name)
                        results = research_graph.research_topic(topic)
                        
                        # Store results in session state
                        st.session_state.research_results = results
                        st.session_state.research_topic = topic
                        
                        st.success("Research completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Research failed: {str(e)}")
    
    # Display results if available
    if hasattr(st.session_state, 'research_results') and st.session_state.research_results:
        st.markdown("---")
        
        # Results header
        st.markdown(f"""
        <div class="card">
            <h2 style="margin-top: 0; color: #374151;">Research Results: {st.session_state.research_topic}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Results tabs
        tab1, tab2 = st.tabs(["📋 Research Plan", "🔍 Search Results"])
        
        with tab1:
            st.markdown("""
            <div class="card">
            """, unsafe_allow_html=True)
            
            if st.session_state.research_results.get("research_plan"):
                st.markdown(st.session_state.research_results["research_plan"])
            else:
                st.warning("Research plan not available")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="card">
            """, unsafe_allow_html=True)
            
            if st.session_state.research_results.get("search_results"):
                st.markdown(st.session_state.research_results["search_results"])
            else:
                st.warning("Search results not available")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show errors if any
        if st.session_state.research_results.get("error"):
            st.markdown(f"""
            <div class="status-warning">
                ⚠️ {st.session_state.research_results["error"]}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()