from typing import Dict, List, Any
from openai import AzureOpenAI
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from .topic_agent import TopicAgent
from pydantic import BaseModel

# Set up Key Vault client
vault_url = "https://tikasecrets.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=vault_url, credential=credential)

class MessageClassification(BaseModel):
    is_off_topic: bool
    redirect_message: str | None

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
        """Format topic results into a concise, Socratic dialogue."""
        messages = [
            {
                "role": "system",
                "content": """You are a Socratic research guide who helps researchers zero in on the exact research topic in a database.
                
                When discussing research topics:
                - Each topic should have a very brief description about what it means to guide the researcher
                - Follow with ONE focused question: "Do any of these fit your interests?"
                - Keep the total response under 150 words but don't stop your responses abruptly
                - Use direct, clear language (avoid phrases like "fascinating", "interesting", etc.)
                - Include concrete examples or methods, not just topic names
                - Talk in paragraph style, never use numbered bullets
                
                
                Remember: Your goal is to help them choose a specific research direction through 
                clear information and targeted questions. No fluff, just substance."""
            },
            {
                "role": "user",
                "content": f"As a Socratic guide, present the key findings from these topics and ask a focused question: {topics}"
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.3,
                max_tokens=150    # Increased slightly to allow for more detail
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return ("I found some research topics in this area. What aspect would you like to explore?")
    
    def _classify_message(self, user_message: str) -> MessageClassification:
        """Classify if message is off-topic and generate appropriate redirect."""
        messages = [
            {
                "role": "system",
                "content": """You are a message classifier for an academic research topic search system.
                Your job is to:
                1. Detect if a message is appropriate for academic research topic search
                2. Identify inappropriate, harmful, or off-topic queries
                3. Generate a redirect message for off-topic queries

                Examples of OFF-TOPIC messages:
                - General greetings ("hi", "hello")
                - Personal questions
                - Harmful/inappropriate requests
                - Non-research requests
                - Casual conversation

                Examples of ON-TOPIC messages:
                - "quantum computing"
                - "climate change research"
                - "machine learning applications"
                - "sociology studies"

                Always err on the side of caution with potentially harmful queries."""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.deployment,
                messages=messages,
                response_format=MessageClassification,
                temperature=0.1  # Lower temperature for more consistent classification
            )
            return response.choices[0].message.parsed
        except Exception as e:
            print(f"Classification error: {str(e)}")
            # If classification fails, treat as off-topic for safety
            return MessageClassification(
                is_off_topic=True, 
                redirect_message="Let's focus on finding academic research topics. What subject would you like to explore?"
            )
    
    def handle_message(self, user_message: str) -> str:
        """Main message handler with off-topic detection."""
        # First classify the message
        classification = self._classify_message(user_message)
        
        if classification.is_off_topic:
            return classification.redirect_message or "Let's focus on finding research topics. What subject would you like to explore?"
            
        # Original topic search logic
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