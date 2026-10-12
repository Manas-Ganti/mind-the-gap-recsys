"""Stage 1: candidate retrieval. Registry maps config names to classes."""
from recsys.retrieval.graph import RP3betaRetriever
from recsys.retrieval.itemknn import ItemKNNRetriever
from recsys.retrieval.popular import PopularRetriever

RETRIEVERS = {
    "popular": PopularRetriever,
    "itemknn": ItemKNNRetriever,
    "graph": RP3betaRetriever,
}


def get_retriever_class(name: str):
    if name == "two_tower":   # imported lazily: pulls in torch
        from recsys.retrieval.two_tower import TwoTowerRetriever
        return TwoTowerRetriever
    return RETRIEVERS[name]


METHOD_ORDER = ["popular", "itemknn", "graph", "two_tower"]
