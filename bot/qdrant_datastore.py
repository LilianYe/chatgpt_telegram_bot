import os
import uuid
from typing import Dict, List, Optional
import arrow
from grpc._channel import _InactiveRpcError
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from qdrant_client.http.models import PayloadSchemaType
from pydantic import BaseModel
from qdrant_client.http import models as rest
import qdrant_client
import config

QDRANT_URL = config.qdrant_url
QDRANT_PORT = config.qdrant_port
QDRANT_GRPC_PORT = config.qdrant_grpc_port
QDRANT_COLLECTION = config.qdrant_collection
EMBEDDING_DIMENSION = config.embedding_dim

def to_unix_timestamp(date_str: str) -> int:
    """
    Convert a date string to a unix timestamp (seconds since epoch).

    Args:
        date_str: The date string to convert.

    Returns:
        The unix timestamp corresponding to the date string.

    If the date string cannot be parsed as a valid date format, returns the current unix timestamp and prints a warning.
    """
    # Try to parse the date string using arrow, which supports many common date formats
    try:
        date_obj = arrow.get(date_str)
        return int(date_obj.timestamp())
    except arrow.parser.ParserError:
        # If the parsing fails, return the current unix timestamp and print a warning
        return int(arrow.now().timestamp())


class ChunkMetadata(BaseModel):
    url: Optional[str] = None
    created_at: Optional[str] = None
    author: Optional[str] = None


class Chunk(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class ChunkWithScore(Chunk):
    score: float


class ChunkMetadataFilter(BaseModel):
    author: Optional[str] = None
    start_date: Optional[str] = None  # any date string format
    end_date: Optional[str] = None  # any date string format


class Query(BaseModel):
    query: str
    filter: Optional[ChunkMetadataFilter] = None
    top_k: Optional[int] = 3


class QueryWithEmbedding(Query):
    embedding: List[float]


class QueryResult(BaseModel):
    query: str
    results: List[ChunkWithScore]


class QdrantDataStore():
    UUID_NAMESPACE = uuid.UUID("3896d314-1e95-4a3a-b45a-945f9f0b541d")
    def __init__(
        self,
        collection_name: Optional[str] = None,
        vector_size: int = EMBEDDING_DIMENSION,
        distance: str = "Cosine",
        recreate_collection: bool = False,
    ):
        """
        Args:
            collection_name: Name of the collection to be used
            vector_size: Size of the embedding stored in a collection
            distance:
                Any of "Cosine" / "Euclid" / "Dot". Distance function to measure
                similarity
        """
        self.client = qdrant_client.QdrantClient(
            url=QDRANT_URL,
            port=int(QDRANT_PORT),
            grpc_port=int(QDRANT_GRPC_PORT),
            prefer_grpc=False,
            timeout=10
        )
        self.collection_name = collection_name or QDRANT_COLLECTION

        # Set up the collection so the points might be inserted or queried
        self._set_up_collection(vector_size, distance, recreate_collection)

    async def upsert(self, chunks: List[Chunk]) -> List[str]:
        """
        Takes in a list of chunks and inserts them into the database.
        """
        points = [
            self._convert_chunk_to_point(chunk)
            for chunk in chunks
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,  # type: ignore
            wait=True,
        )

    async def query(
        self,
        queries: List[QueryWithEmbedding],
    ) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching chunks and scores.
        """
        search_requests = [
            self._convert_query_to_search_request(query) for query in queries
        ]
        results = self.client.search_batch(
            collection_name=self.collection_name,
            requests=search_requests,
        )
        return [
            QueryResult(
                query=query.query,
                results=[
                    self._convert_scored_point_to_chunk_with_score(point)
                    for point in result
                ],
            )
            for query, result in zip(queries, results)
        ]

    async def delete(
        self,
        filter: Optional[ChunkMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """
        Removes vectors by filter, or everything in the datastore.
        Returns whether the operation was successful.
        """
        if filter is None and delete_all is None:
            raise ValueError(
                "Please provide one of the parameters: filter or delete_all."
            )

        if delete_all:
            points_selector = rest.Filter()
        else:
            points_selector = self._convert_metadata_filter_to_qdrant_filter(
                filter
            )

        response = self.client.delete(
            collection_name=self.collection_name,
            points_selector=points_selector,  # type: ignore
        )
        return "COMPLETED" == response.status

    def _convert_chunk_to_point(
        self, chunk: Chunk
    ) -> rest.PointStruct:
        created_at = (
            to_unix_timestamp(chunk.metadata.created_at)
            if chunk.metadata.created_at is not None
            else None
        )
        return rest.PointStruct(
            id=self._create_chunk_id(chunk.id),
            vector=chunk.embedding,  # type: ignore
            payload={
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata.dict(),
                "created_at": created_at,
            },
        )

    def _create_chunk_id(self, external_id: Optional[str]) -> str:
        if external_id is None:
            return uuid.uuid4().hex
        return uuid.uuid5(self.UUID_NAMESPACE, external_id).hex

    def _convert_query_to_search_request(
        self, query: QueryWithEmbedding
    ) -> rest.SearchRequest:
        return rest.SearchRequest(
            vector=query.embedding,
            filter=self._convert_metadata_filter_to_qdrant_filter(query.filter),
            limit=query.top_k,  # type: ignore
            with_payload=True,
            with_vector=False,
        )

    def _convert_metadata_filter_to_qdrant_filter(
        self,
        metadata_filter: Optional[ChunkMetadataFilter] = None,
    ) -> Optional[rest.Filter]:
        if metadata_filter is None:
            return None

        must_conditions, should_conditions = [], []

        # Equality filters for the payload attributes
        if metadata_filter:
            meta_attributes_keys = {
                "author": "metadata.author",
            }

            for meta_attr_name, payload_key in meta_attributes_keys.items():
                attr_value = getattr(metadata_filter, meta_attr_name)
                if attr_value is None:
                    continue

                must_conditions.append(
                    rest.FieldCondition(
                        key=payload_key, match=rest.MatchValue(value=attr_value)
                    )
                )

            # Date filters use range filtering
            start_date = metadata_filter.start_date
            end_date = metadata_filter.end_date
            if start_date or end_date:
                gte_filter = (
                    to_unix_timestamp(start_date) if start_date is not None else None
                )
                lte_filter = (
                    to_unix_timestamp(end_date) if end_date is not None else None
                )
                must_conditions.append(
                    rest.FieldCondition(
                        key="created_at",
                        range=rest.Range(
                            gte=gte_filter,
                            lte=lte_filter,
                        ),
                    )
                )

        if 0 == len(must_conditions) and 0 == len(should_conditions):
            return None

        return rest.Filter(must=must_conditions, should=should_conditions)

    def _convert_scored_point_to_chunk_with_score(
        self, scored_point: rest.ScoredPoint
    ) -> ChunkWithScore:
        payload = scored_point.payload or {}
        return ChunkWithScore(
            id=scored_point.id,
            text=scored_point.payload.get("text"),  # type: ignore
            metadata=scored_point.payload.get("metadata"),  # type: ignore
            embedding=scored_point.vector,  # type: ignore
            score=scored_point.score,
        )

    def _set_up_collection(
        self, vector_size: int, distance: str, recreate_collection: bool
    ):
        distance = rest.Distance[distance.upper()]

        if recreate_collection:
            self._recreate_collection(distance, vector_size)

        try:
            collection_info = self.client.get_collection(self.collection_name)
            current_distance = collection_info.config.params.vectors.distance  # type: ignore
            current_vector_size = collection_info.config.params.vectors.size  # type: ignore

            if current_distance != distance:
                raise ValueError(
                    f"Collection '{self.collection_name}' already exists in Qdrant, "
                    f"but it is configured with a similarity '{current_distance.name}'. "
                    f"If you want to use that collection, but with a different "
                    f"similarity, please set `recreate_collection=True` argument."
                )

            if current_vector_size != vector_size:
                raise ValueError(
                    f"Collection '{self.collection_name}' already exists in Qdrant, "
                    f"but it is configured with a vector size '{current_vector_size}'. "
                    f"If you want to use that collection, but with a different "
                    f"vector size, please set `recreate_collection=True` argument."
                )
        except (UnexpectedResponse, _InactiveRpcError, ResponseHandlingException):
            self._recreate_collection(distance, vector_size)

    def _recreate_collection(self, distance: rest.Distance, vector_size: int):
        self.client.recreate_collection(
            self.collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )

        # Create the payload index for the created_at attribute, to make the lookup
        # by range filters faster
        self.client.create_payload_index(
            self.collection_name,
            field_name="created_at",
            field_schema=PayloadSchemaType.INTEGER,
        )
