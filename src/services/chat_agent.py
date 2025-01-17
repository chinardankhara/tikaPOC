from typing import Dict, List, Any
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from .topic_agent import TopicAgent

load_dotenv()

class ChatManager:
    def __init__(self) -> None:
        """Initialize chat manager with OpenAI client and topic agent."""
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-15-preview"
        )
        self.topic_agent = TopicAgent()
        
    def _format_topics(self, topics: List[Dict[str, Any]]) -> str:
        """Format topic results into a readable message."""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful research assistant. Format the given topics into a 
                natural, readable response. For each topic, include its name and description.
                If the topics seem off-target, suggest how the user might refine their search."""
            },
            {
                "role": "user",
                "content": f"Format these topics into a response: {topics}"
            }
        ]
        
        response = self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    def handle_message(self, user_message: str) -> str:
        """
        Main message handler. Searches for topics and formats response.
        
        Args:
            user_message: The user's input message
            
        Returns:
            Response string to display to user
        """
        # Search for topics
        topics = self.topic_agent.process_query(user_message)
        
        if not topics:
            return ("I couldn't find any relevant research topics. Could you try rephrasing "
                   "your query or being more specific?")
        
        return self._format_topics(topics)
    
    def exclude_current_topics(self, topic_ids: List[str]) -> None:
        """Exclude topics from future searches."""
        self.topic_agent.exclude_topics(topic_ids)
    
    def reset_conversation(self) -> None:
        """Reset the conversation and agent memory."""
        self.topic_agent.reset_memory() 