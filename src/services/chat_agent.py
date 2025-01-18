from typing import Dict, List, Any
from openai import AzureOpenAI
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from .topic_agent import TopicAgent

# Set up Key Vault client
vault_url = "https://tikasecrets.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=vault_url, credential=credential)

class ChatManager:
    def __init__(self) -> None:
        """Initialize chat manager with OpenAI client and topic agent."""
        self.client = AzureOpenAI(
            azure_endpoint=secret_client.get_secret("AZURE-OPENAI-ENDPOINT").value,
            api_key=secret_client.get_secret("AZURE-OPENAI-KEY").value,
            api_version="2024-12-01-preview"
        )
        self.deployment = secret_client.get_secret("AZURE-OPENAI-DEPLOYMENT").value
        print(f"Using OpenAI deployment: {self.deployment}")
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
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,  # Use the deployment name from Key Vault
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            # Fallback to simple formatting if API fails
            return "\n".join([
                f"Topic {i+1}: {topic['display_name']}\n{topic['description']}"
                for i, topic in enumerate(topics)
            ])
    
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