import json
import redis.asyncio as redis
from typing import List, Any
import os
from dotenv import load_dotenv
load_dotenv()

REDIS_HOST=os.environ["REDIS_HOST"]
REDIS_PASSWORD=os.environ["REDIS_PASSWORD"]


class RedisPython:
    """
    Asynchronous client wrapper for Redis operations handling Agentic Memory.
    """
    def __init__(self):
        self.r = redis.Redis(host=REDIS_HOST, password=REDIS_PASSWORD, decode_responses=True, db=1)

    def get_client(self) -> redis.Redis:
        """Returns the raw asynchronous Redis client object."""
        return self.r

    async def close_connection(self):
        """Ensure connection is closed cleanly when the app shuts down."""
        await self.r.close()
        
    async def store_turn(self, chat_id: str, new_messages: List[Any]) -> bool:
        """
        Stores an entire conversational turn (User -> Tools -> Assistant) 
        as a single JSON array element in a Redis List.
        """
        turn_data = []
        for msg in new_messages:
            if hasattr(msg, "model_dump"):
                turn_data.append(msg.model_dump(exclude_none=True))
            elif hasattr(msg, "dict"):
                turn_data.append(msg.dict(exclude_none=True))
            else:
                turn_data.append(msg)
                
        messages_json = json.dumps(turn_data)
        await self.r.rpush(chat_id, messages_json)
        await self.r.expire(chat_id, 1800) 
        return True

    async def get_mapped_history(self, chat_id: str, turns: int = 3) -> List[dict]:
        """
        Retrieves the last N turns from Redis and flattens them into 
        a single, chronologically ordered list of message dictionaries.
        """
        raw_history = await self.r.lrange(name=chat_id, start=-turns, end=-1)
        
        formatted_messages = []
        for turn_str in raw_history:
            try:
                turn_messages = json.loads(turn_str)
                formatted_messages.extend(turn_messages)
            except Exception:
                continue
                
        return formatted_messages
