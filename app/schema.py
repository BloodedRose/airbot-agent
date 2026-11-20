from sqlalchemy import Column, Integer, String, Boolean, JSON, Enum, Table, ForeignKey, BigInteger, Text, DateTime, Index
from sqlalchemy.orm import relationship, Mapped, mapped_column
from .database import Base
import enum, uuid
from datetime import datetime

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


#--------------------------- Artifacts Configurations -----------------

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(80), primary_key=True, default=lambda: new_id("file"))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    extension: Mapped[str] = mapped_column(String(16), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    row_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    schema_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), default="unknown")
    source_query_context_id: Mapped[str | None] = mapped_column(
        String(80), ForeignKey("query_contexts.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class QueryContext(Base):
    __tablename__ = "query_contexts"

    id: Mapped[str] = mapped_column(String(80), primary_key=True, default=lambda: new_id("qry"))
    conversation_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    service_name: Mapped[str] = mapped_column(String(255), nullable=False)
    entity_name: Mapped[str] = mapped_column(String(255), nullable=False)
    query_plan: Mapped[dict] = mapped_column(JSON, nullable=False)
    metadata_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

class QueryExecution(Base):
    __tablename__ = "query_executions"
    __table_args__ = (
        Index("ix_query_exec_context_created", "query_context_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True, default=lambda: new_id("exec"))
    query_context_id: Mapped[str] = mapped_column(String(80), ForeignKey("query_contexts.id"), nullable=False)
    parent_execution_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    operation: Mapped[str] = mapped_column(String(32), default="new")
    status: Mapped[str] = mapped_column(String(32), default="completed")
    odata_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(String(80), ForeignKey("artifacts.id"), nullable=True)
    row_count: Mapped[int] = mapped_column(BigInteger, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
