# Minimal subset of kg_gen/kg_gen.py from KGGen (https://github.com/stair-lab/kg-gen,
# v0.4.0, commit 6259b4c, MIT licence, see LICENSE). Modified for this thesis: only the
# methods that source/mine_eval.py calls are kept (model initialisation, deduplicate,
# to_nx, generate_embeddings, retrieve and its helpers); the method bodies are unchanged.
# Used only for the MINE dataset.

from typing import Union, Optional

from kg_gen.steps._3_deduplicate import run_deduplication, DeduplicateMethod
from kg_gen.models import Graph
import dspy
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Configure dspy logging to only show errors
import logging

logger = logging.getLogger(__name__)

dspy_logger = logging.getLogger("dspy")
dspy_logger.setLevel(logging.CRITICAL)


class KGGen:
    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_tokens: int = 16000,  # minimum for gpt-5 family models
        temperature: float = 0.0,
        reasoning_effort: str = None,
        api_key: str = None,
        api_base: str = None,
        retrieval_model: Optional[str] = None,
        disable_cache: bool = False,
    ):
        """Initialize KGGen with optional model configuration

        Args:
            model: Name of model to use (e.g. 'gpt-4')
            temperature: Temperature for model sampling
            api_key: API key for model access
            api_base: Specify the base URL endpoint for making API calls to a language model service
        """
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.retrieval_model: Optional[SentenceTransformer] = None
        self.lm = None
        self.disable_cache = disable_cache

        self.init_model(
            model=model,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
            retrieval_model=retrieval_model,
        )

    def validate_temperature(self, temperature: float):
        if "gpt-5" in self.model and temperature < 1.0:
            raise ValueError("Temperature must be 1.0 for gpt-5 family models")

    def validate_max_tokens(self, max_tokens: int):
        if "gpt-5" in self.model and max_tokens < 16000:
            raise ValueError("Max tokens must be 16000 for gpt-5 family models")

    def init_model(
        self,
        model: str = None,
        reasoning_effort: str = None,
        max_tokens: int = None,
        temperature: float = None,
        retrieval_model: str = None,
        api_key: str = None,
        api_base: str = None,
    ):
        """Initialize or reinitialize the model with new parameters

        Args:
            model: Name of model to use (e.g. 'gpt-4')
            temperature: Temperature for model sampling
            api_key: API key for model access
            api_base: API base for model access
            retrieval_model: Name of retrieval model to use
            reasoning_effort: Reasoning effort for model
            max_tokens: Maximum tokens for model
            temperature: Temperature for model sampling
        """

        # Update instance variables if new values provided
        if model is not None:
            self.model = model
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if api_key is not None:
            self.api_key = api_key
        if api_base is not None:
            self.api_base = api_base
        if temperature is not None:
            self.temperature = temperature
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort
        if retrieval_model is not None:
            self.retrieval_model = SentenceTransformer(retrieval_model)

        self.validate_temperature(self.temperature)
        self.validate_max_tokens(self.max_tokens)

        # Initialize dspy LM with current settings
        if self.api_key:
            self.lm = dspy.LM(
                model=self.model,
                api_key=self.api_key,
                reasoning={"effort": self.reasoning_effort}
                if self.reasoning_effort
                else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_base=self.api_base,
                cache=not self.disable_cache,
                model_type="responses" if self.model.startswith("openai/") else "chat",
            )
        else:
            self.lm = dspy.LM(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_base=self.api_base,
                reasoning={"effort": self.reasoning_effort}
                if self.reasoning_effort
                else None,
                cache=not self.disable_cache,
                model_type="responses" if self.model.startswith("openai/") else "chat",
            )

    def deduplicate(
        self,
        graph: Graph,
        method: DeduplicateMethod = DeduplicateMethod.FULL,
        semhash_similarity_threshold: float = 0.95,  # recommended to keep at 0.95
        model: str = None,
        temperature: float = None,
        api_key: str = None,
        api_base: str = None,
        context: str = "",  # TODO: implement context
    ) -> Graph:
        # Reinitialize dspy with new parameters if any are provided
        if any([model, temperature, api_key, api_base]):
            self.init_model(
                model=model or self.model,
                temperature=temperature or self.temperature,
                api_key=api_key or self.api_key,
                api_base=api_base or self.api_base,
            )

        return run_deduplication(
            lm=self.lm,
            graph=graph,
            method=method,
            retrieval_model=self.retrieval_model,
            semhash_similarity_threshold=semhash_similarity_threshold,
        )

    def _parse_embedding_model(
        self, model: Optional[SentenceTransformer] = None
    ) -> Optional[SentenceTransformer]:
        if model is None:
            model = self.retrieval_model
        if model is None:
            raise ValueError("No retrieval model provided")
        return model

    @staticmethod
    def to_nx(graph: Graph) -> nx.DiGraph:
        G = nx.DiGraph()
        for entity in graph.entities:
            G.add_node(entity)

        for relation in graph.relations:
            source, rel, target = relation
            G.add_edge(source, target, relation=rel)
        return G

    def generate_embeddings(
        self,
        graph: Union[Graph, nx.DiGraph],
        model: Optional[SentenceTransformer] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        model = self._parse_embedding_model(model)
        if isinstance(graph, Graph):
            graph = self.to_nx(graph)

        node_embeddings = {node: model.encode(node).tolist() for node in graph.nodes}
        relation_embeddings = {
            rel: model.encode(rel).tolist()
            # TODO: this is triggering index out of range error
            for rel in set(edge[2]["relation"] for edge in graph.edges(data=True))
        }
        return node_embeddings, relation_embeddings

    def retrieve(
        self,
        query: str,
        node_embeddings: dict[str, np.ndarray],
        graph: nx.DiGraph,
        model: Optional[SentenceTransformer] = None,
        k: int = 8,
        verbose: bool = False,
    ) -> tuple[list[tuple[str, float]], set[str], str]:
        model = self._parse_embedding_model(model)
        top_nodes = self.retrieve_relevant_nodes(query, node_embeddings, model, k)
        context = set()
        for node, _ in top_nodes:
            node_context = self.retrieve_context(node, graph)
            if verbose:
                print(f"Context for node {node}: {node_context}")
            context.update(node_context)
        context_text = " ".join(context)
        if verbose:
            print(f"Combined context: '{context_text}'\n---")
        return top_nodes, context, context_text

    @staticmethod
    def retrieve_relevant_nodes(
        query: str,
        node_embeddings: dict[str, np.ndarray],
        model: SentenceTransformer,
        k: int = 8,
    ) -> list[tuple[str, float]]:
        query_embedding = model.encode(query).reshape(1, -1)
        similarities = []
        for node, embed in node_embeddings.items():
            target_embedding = np.array(embed).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, target_embedding)[0][0]
            similarities.append((node, similarity))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities[:k]

    @staticmethod
    def retrieve_context(node: str, graph: nx.DiGraph, depth: int = 2) -> list[str]:
        context = set()

        def explore_neighbors(current_node, current_depth):
            if current_depth > depth:
                return
            # Outgoing edges
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]["relation"]
                context.add(f"{current_node} {rel} {neighbor}.")
                explore_neighbors(neighbor, current_depth + 1)
            # Incoming edges
            for neighbor in graph.predecessors(current_node):
                rel = graph[neighbor][current_node]["relation"]
                context.add(f"{neighbor} {rel} {current_node}.")
                explore_neighbors(neighbor, current_depth + 1)

        explore_neighbors(node, 1)
        return list(context)
