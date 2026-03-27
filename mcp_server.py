cat > mcp_server.py << 'EOF'
import os
import uuid
import json
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import psycopg2
from psycopg2.extras import DictCursor

from embed_service import EmbedServiceManager

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCP Server - Memory Hub",
    version="1.0.0",
    description="MCP Server for semantic memory management"
)

embed_manager = EmbedServiceManager()
qdrant_client = AsyncQdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333")
)

def get_db():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        database=os.getenv("POSTGRES_DB", "memory_hub")
    )
    return conn

class MemorySaveRequest(BaseModel):
    content: str
    type: str
    project: str
    tags: Optional[List[str]] = None
    assistant: Optional[str] = "mcp_server"

class MemorySearchRequest(BaseModel):
    query: str
    project: Optional[str] = None
    limit: int = 5
    min_score: float = 0.5

class MemoryDeleteRequest(BaseModel):
    memory_id: str

async def verify_api_key(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing API key")

    expected_key = os.getenv("MCP_API_KEY", "change-me-to-secure-key")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    token = authorization.split(" ")[1]
    if token != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return token

@app.get("/health")
async def health_check():
    embed_health = await embed_manager.health_check()

    try:
        qdrant_health = await qdrant_client.get_collections()
        qdrant_status = True
    except Exception as e:
        qdrant_status = False
        logger.error(f"Qdrant health check failed: {str(e)}")

    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        postgres_status = True
    except Exception as e:
        postgres_status = False
        logger.error(f"PostgreSQL health check failed: {str(e)}")

    return {
        "status": "ok" if all([qdrant_status, postgres_status]) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "providers": embed_health,
        "qdrant": qdrant_status,
        "postgres": postgres_status
    }

@app.post("/tools/memory_save")
async def memory_save(
    request: MemorySaveRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        memory_id = str(uuid.uuid4())
        now = datetime.utcnow()

        logger.info(f"Generating embedding for memory: {memory_id}")
        embedding = await embed_manager.embed(request.content)

        logger.info(f"Saving to Qdrant: {memory_id}")
        point = PointStruct(
            id=hash(memory_id) % (2**63),
            vector=embedding,
            payload={
                "memory_id": memory_id,
                "content": request.content,
                "type": request.type,
                "project": request.project,
                "tags": request.tags or [],
                "assistant": request.assistant,
                "created_at": now.isoformat()
            }
        )

        await qdrant_client.upsert(
            collection_name=os.getenv("QDRANT_COLLECTION", "memories"),
            points=[point]
        )

        logger.info(f"Saving to PostgreSQL: {memory_id}")
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO memories (id, content, type, project, tags, assistant, embedding_provider, embedding_model, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            memory_id,
            request.content,
            request.type,
            request.project,
            json.dumps(request.tags or []),
            request.assistant,
            "ollama",
            os.getenv("OLLAMA_MODEL", "nomic-embed-text"),
            now,
            now
        ))
        conn.commit()

        cursor.execute("""
            INSERT INTO audit_log (time, action, memory_id, project, assistant, details)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            now,
            "CREATE",
            memory_id,
            request.project,
            request.assistant,
            json.dumps({"content_length": len(request.content)})
        ))
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Memory saved successfully: {memory_id}")
        return {
            "success": True,
            "memory_id": memory_id,
            "timestamp": now.isoformat(),
            "embedding_provider": "ollama"
        }

    except Exception as e:
        logger.error(f"Error saving memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/memory_search")
async def memory_search(
    request: MemorySearchRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        logger.info(f"Searching for: {request.query}")

        query_embedding = await embed_manager.embed(request.query)

        results = await qdrant_client.search(
            collection_name=os.getenv("QDRANT_COLLECTION", "memories"),
            query_vector=query_embedding,
            limit=request.limit,
            score_threshold=request.min_score,
            query_filter={
                "must": [
                    {
                        "key": "project",
                        "match": {"value": request.project}
                    }
                ]
            } if request.project else None
        )

        memories = []
        for result in results:
            memories.append({
                "memory_id": result.payload.get("memory_id"),
                "content": result.payload.get("content"),
                "type": result.payload.get("type"),
                "project": result.payload.get("project"),
                "tags": result.payload.get("tags", []),
                "score": result.score,
                "created_at": result.payload.get("created_at")
            })

        logger.info(f"Found {len(memories)} memories")
        return {
            "success": True,
            "count": len(memories),
            "query": request.query,
            "memories": memories
        }

    except Exception as e:
        logger.error(f"Error searching memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/memory_list")
async def memory_list(
    project: Optional[str] = None,
    type_filter: Optional[str] = None,
    limit: int = 10,
    api_key: str = Depends(verify_api_key)
):
    try:
        logger.info(f"Listing memories: project={project}, type={type_filter}")

        conn = get_db()
        cursor = conn.cursor(cursor_factory=DictCursor)

        query = "SELECT * FROM memories WHERE 1=1"
        params = []

        if project:
            query += " AND project = %s"
            params.append(project)

        if type_filter:
            query += " AND type = %s"
            params.append(type_filter)

        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        memories = [
            {
                "memory_id": row["id"],
                "content": row["content"],
                "type": row["type"],
                "project": row["project"],
                "tags": row["tags"],
                "assistant": row["assistant"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None
            }
            for row in rows
        ]

        logger.info(f"Listed {len(memories)} memories")
        return {
            "success": True,
            "count": len(memories),
            "memories": memories
        }

    except Exception as e:
        logger.error(f"Error listing memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/tools/memory_delete")
async def memory_delete(
    request: MemoryDeleteRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        logger.info(f"Deleting memory: {request.memory_id}")

        await qdrant_client.delete(
            collection_name=os.getenv("QDRANT_COLLECTION", "memories"),
            points_selector={"has": [{"key": "memory_id", "match": {"value": request.memory_id}}]}
        )

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = %s", (request.memory_id,))

        cursor.execute("""
            INSERT INTO audit_log (time, action, memory_id, details)
            VALUES (%s, %s, %s, %s)
        """, (
            datetime.utcnow(),
            "DELETE",
            request.memory_id,
            json.dumps({"deleted": True})
        ))
        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Memory deleted: {request.memory_id}")
        return {
            "success": True,
            "memory_id": request.memory_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup():
    logger.info("MCP Server starting up")

    try:
        collection_name = os.getenv("QDRANT_COLLECTION", "memories")
        collections = await qdrant_client.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)

        if not collection_exists:
            logger.info(f"Creating Qdrant collection: {collection_name}")
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        else:
            logger.info(f"Qdrant collection exists: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to verify Qdrant: {str(e)}")

@app.on_event("shutdown")
async def shutdown():
    logger.info("MCP Server shutting down")
    await embed_manager.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_PORT", "8000")),
        log_level="info"
    )
EOF
