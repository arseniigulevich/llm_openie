"""Minimal subset of KGGen (https://github.com/stair-lab/kg-gen, v0.4.0, commit 6259b4c, MIT).

Only the parts that source/mine_eval.py needs for the MINE score are kept: the Graph model,
graph deduplication, conversion to networkx, node embeddings and retrieval. Knowledge-graph
generation, clustering, aggregation, visualisation, the CLI and the Neo4j integration are
removed. Used only for the MINE dataset.
"""

from .kg_gen import KGGen
from .models import Graph

__all__ = ["KGGen", "Graph"]
