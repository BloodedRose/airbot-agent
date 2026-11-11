from sqlalchemy import Column, Integer, String, Boolean, JSON, Enum, Table, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base
import enum

class ProviderEnum(str, enum.Enum):
    ollama = "ollama"
    openai = "openai"
    gemini = "gemini"
    anthropic = "anthropic"

class TransportEnum(str, enum.Enum):
    stdio = "stdio"
    sse = "sse"

class ModelConfig(Base):
    __tablename__ = "model_configs"

    id = Column(Integer, primary_key=True, index=True)
    display_name = Column(String(100), unique=True, index=True, nullable=False)
    actual_name = Column(String(100), nullable=False)                        
    provider = Column(Enum(ProviderEnum), nullable=False)
    base_url = Column(String(255), nullable=True)
    api_key = Column(String(255), nullable=True)
    additional_headers = Column(JSON, nullable=True)
    
    is_active = Column(Boolean, default=True)

class MCPServerConfig(Base):
    __tablename__ = "mcp_servers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False) # e.g., "github-mcp", "local-filesystem"
    transport = Column(Enum(TransportEnum), nullable=False)
    
    # Stdio configurations (Command line execution)
    command = Column(String(255), nullable=True) # e.g., "node", "python", "npx"
    args = Column(JSON, nullable=True)           # e.g., ["-y", "@modelcontextprotocol/server-postgres"]
    env = Column(JSON, nullable=True)            # Environment variables for the tool
    
    # SSE configurations (HTTP/Websockets)
    sse_url = Column(String(255), nullable=True) # e.g., "http://localhost:5000/sse"
    
    is_active = Column(Boolean, default=True)


#------------------- User Configurations -----------------

user_model_access = Table(
    "user_model_access",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("model_id", Integer, ForeignKey("model_configs.id"), primary_key=True)
)

user_mcp_access = Table(
    "user_mcp_access",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("mcp_id", Integer, ForeignKey("mcp_servers.id"), primary_key=True)
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    allowed_models = relationship("ModelConfig", secondary=user_model_access, lazy="selectin")
    allowed_mcp_servers = relationship("MCPServerConfig", secondary=user_mcp_access, lazy="selectin")