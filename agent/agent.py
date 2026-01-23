"""
LangChain Agent for AstroLens.

Simplified agent that works with LangChain v1.2+.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .tools import ALL_TOOLS, set_db
from annotator.prompts import CHAT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class AstroLensAgent:
    """
    Simplified agent for conversational image analysis.
    
    Uses direct LLM calls with tool descriptions for simpler compatibility.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.3,
    ):
        self.provider = provider
        self.temperature = temperature
        self.llm = None
        self.chat_history = []

        if provider == "openai":
            self.model = model or "gpt-4o"
            self._init_openai()
        elif provider == "ollama":
            self.model = model or "llava"
            self._init_ollama()
        else:
            self.model = "none"
            logger.info("LLM provider set to 'none'; agent will use heuristics")

    def _init_openai(self):
        """Initialize OpenAI LLM."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set; agent will use fallback")
            return

        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=api_key,
            )
            logger.info(f"Initialized OpenAI agent with model {self.model}")
        except ImportError as e:
            logger.error(f"Failed to initialize OpenAI: {e}")

    def _init_ollama(self):
        """Initialize Ollama LLM."""
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

        try:
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                from langchain_community.chat_models import ChatOllama
            self.llm = ChatOllama(
                model=self.model,
                base_url=ollama_url,
                temperature=self.temperature,
            )
            logger.info(f"Initialized Ollama agent with model {self.model}")
        except ImportError as e:
            logger.error(f"Failed to initialize Ollama: {e}")

    def chat(self, message: str, db=None) -> dict:
        """
        Process a chat message and return response.
        
        Args:
            message: User message
            db: Database session for tools to use
        
        Returns:
            Dict with 'output' and optionally 'tool_calls'
        """
        # Set database for tools
        if db:
            set_db(db)

        message_lower = message.lower()
        
        # ACTION requests - run heuristics directly (don't ask LLM)
        action_keywords = ["analyze", "analyse", "list", "show", "stats", "statistic", "help"]
        if any(kw in message_lower for kw in action_keywords):
            return self._heuristic_response(message, db)

        # If LLM not available, use heuristic fallback
        if self.llm is None:
            return self._heuristic_response(message, db)

        try:
            # Build prompt with tool descriptions
            tool_descriptions = "\n".join([
                f"- {tool.name}: {tool.description}" 
                for tool in ALL_TOOLS
            ])
            
            system_prompt = f"""{CHAT_SYSTEM_PROMPT}

Available tools:
{tool_descriptions}

When the user asks to perform an action, explain what you would do and call the appropriate tool.
If you need to use a tool, respond with the tool name and arguments.
"""
            
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            
            messages = [SystemMessage(content=system_prompt)]
            
            # Add chat history
            for h in self.chat_history[-10:]:  # Keep last 10 messages
                if h["role"] == "user":
                    messages.append(HumanMessage(content=h["content"]))
                else:
                    messages.append(AIMessage(content=h["content"]))
            
            messages.append(HumanMessage(content=message))
            
            # Get response
            response = self.llm.invoke(messages)
            reply = response.content if hasattr(response, 'content') else str(response)
            
            # Check if response mentions a tool
            tool_calls = self._extract_and_run_tools(reply, db)
            
            if tool_calls:
                # If tools were called, append their results
                reply = reply + "\n\n" + "\n".join(tool_calls)
            
            # Update history
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": reply})
            
            return {"output": reply, "tool_calls": tool_calls}
            
        except Exception as e:
            logger.error(f"Agent error: {e}")
            # Fallback to heuristics on error
            return self._heuristic_response(message, db)

    def _extract_and_run_tools(self, reply: str, db) -> list:
        """Check if LLM response mentions tools and run them."""
        tool_results = []
        reply_lower = reply.lower()
        
        # Simple keyword matching for tool invocation
        if "list" in reply_lower and "image" in reply_lower:
            from .tools import list_images
            result = list_images.invoke({"limit": 10, "anomaly_only": False})
            tool_results.append(f"[Tool: list_images]\n{result}")
        
        if "statistic" in reply_lower or "stats" in reply_lower:
            from .tools import get_statistics
            result = get_statistics.invoke({})
            tool_results.append(f"[Tool: get_statistics]\n{result}")
        
        if "anomal" in reply_lower:
            from .tools import list_images
            result = list_images.invoke({"limit": 10, "anomaly_only": True})
            tool_results.append(f"[Tool: list_images (anomalies)]\n{result}")
        
        # If LLM mentions analyze_image but doesn't do it, do it for them
        if "analyze_image" in reply_lower or "analyse" in reply_lower:
            heuristic = self._heuristic_response("analyze all", db)
            tool_results.append(f"[Action: Analyzing...]\n{heuristic['output']}")
        
        return tool_results

    def _heuristic_response(self, message: str, db) -> dict:
        """Simple heuristic responses when LLM is not available."""
        message_lower = message.lower()

        # Check statistics FIRST (before "show" which would match "show statistics")
        if "stats" in message_lower or "statistic" in message_lower:
            from .tools import get_statistics
            result = get_statistics.invoke({})
            return {"output": result}

        if "list" in message_lower or "show" in message_lower:
            if "anomal" in message_lower:
                from .tools import list_images
                result = list_images.invoke({"limit": 10, "anomaly_only": True})
                return {"output": f"Here are the anomalies:\n{result}"}
            else:
                from .tools import list_images
                result = list_images.invoke({"limit": 10, "anomaly_only": False})
                return {"output": f"Here are your images:\n{result}"}

        if "analyze" in message_lower or "analyse" in message_lower:
            # Analyze images
            from .tools import list_images, analyze_image
            
            # Get list of unanalyzed images directly
            unanalyzed_list = list_images.invoke({"limit": 50, "unanalyzed_only": True})
            
            # Find unanalyzed ones
            unanalyzed = []
            for line in unanalyzed_list.split('\n'):
                if line.startswith('•') and 'ID' in line:
                    try:
                        id_part = line.split('ID')[1].split(':')[0].strip()
                        unanalyzed.append(int(id_part))
                    except:
                        pass
            
            if not unanalyzed:
                return {"output": "✓ All images are already analyzed!\n\nUse 'list images' to see your collection."}
            
            results = []
            for img_id in unanalyzed[:5]:  # Limit to 5 for speed
                result = analyze_image.invoke({"image_id": img_id})
                results.append(f"• Image {img_id}: {result}")
            
            return {
                "output": f"✓ Analyzed {len(results)} images:\n\n" + "\n".join(results)
            }

        if "help" in message_lower:
            return {
                "output": (
                    "I can help you with:\n"
                    "• 'list images' - Show uploaded images\n"
                    "• 'show anomalies' - List flagged anomalies\n"
                    "• 'analyze all images' - Run ML analysis\n"
                    "• 'statistics' - Show collection stats\n"
                    "• 'help' - Show this message"
                )
            }

        return {
            "output": (
                "I understand you said: '" + message + "'\n\n"
                "I can help with: list images, show anomalies, analyze images, statistics.\n"
                "Type 'help' for more options."
            )
        }

    def reset_memory(self):
        """Clear conversation history."""
        self.chat_history = []
