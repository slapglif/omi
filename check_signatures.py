import inspect
from omi.storage.graph_palace import GraphPalace
from omi.embeddings import EmbeddingCache, OllamaEmbedder
from omi.belief import BeliefNetwork, ContradictionDetector

print("GraphPalace.__init__:", inspect.signature(GraphPalace.__init__))
print("EmbeddingCache.__init__:", inspect.signature(EmbeddingCache.__init__))
print("OllamaEmbedder.__init__:", inspect.signature(OllamaEmbedder.__init__))
print("BeliefNetwork.__init__:", inspect.signature(BeliefNetwork.__init__))
print("ContradictionDetector.__init__:", inspect.signature(ContradictionDetector.__init__))
