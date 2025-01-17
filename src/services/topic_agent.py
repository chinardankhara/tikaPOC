import os
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Deque
from collections import deque
from openai import AzureOpenAI
from .topic_search import TopicSearcher
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AgentState:
    """Maintains agent's memory of excluded topics and recent queries."""
    excluded_topic_ids: Set[str] = field(default_factory=set)
    recent_queries: Deque[str] = field(default_factory=lambda: deque(maxlen=5))

class TopicAgent:
    def __init__(self, openai_api_key: str):
        """Initialize the agent with necessary components."""
        self.state = AgentState()
        self.searcher = TopicSearcher()
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-15-preview"
        )
        
    def _rewrite_query(self) -> str:
        """
        Rewrite the query using context from recent queries.
        Only called when there are multiple queries in history.
        """
        # Convert deque to list for better prompt formatting
        query_history = list(self.state.recent_queries)
        
        messages = [
            {
                "role": "system",
                "content": """You are a query rewriting assistant. Given a sequence of user queries, 
                rewrite them into a single, comprehensive query that captures the user's evolving intent. 
                Focus on creating a search-friendly query that works well with embedding-based search.
                Return ONLY the rewritten query, nothing else."""
            },
            {
                "role": "user",
                "content": f"""Previous queries (in chronological order):
                {query_history[:-1]}
                
                Current query: {query_history[-1]}
                
                Rewrite these queries into a single, comprehensive query."""
            }
        ]
        
        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=messages,
            temperature=0.3,  # Lower temperature for more focused rewrites
            max_tokens=100    # Rewritten query should be concise
        )
        
        return response.choices[0].message.content.strip()
    
    def process_query(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Process user input and return relevant topics.
        
        Args:
            user_input: The user's query string
            
        Returns:
            List of topic dictionaries containing id, display_name, and description
        """
        # Update recent queries
        self.state.recent_queries.append(user_input)
        if len(self.state.recent_queries) > 4:
            self.state.recent_queries.popleft()  # Remove the oldest query if more than 4
        
        # Rewrite query using context if we have multiple queries
        query_to_search = (
            self._rewrite_query() 
            if len(self.state.recent_queries) > 1 
            else user_input
        )
        
        # Search with excluded topics
        return self.searcher.search_topics(
            query=query_to_search,
            excluded_topic_ids=self.state.excluded_topic_ids
        )
    
    def exclude_topics(self, topic_ids: List[str]) -> None:
        """Add topics to the excluded set."""
        self.state.excluded_topic_ids.update(topic_ids)
    
    def reset_memory(self) -> None:
        """Reset the agent's memory (both excluded topics and query history)."""
        self.state = AgentState()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current agent state for debugging."""
        return {
            "excluded_topics_count": len(self.state.excluded_topic_ids),
            "recent_queries": list(self.state.recent_queries),
            "has_context": len(self.state.recent_queries) > 1
        }
