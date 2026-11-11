from dotvectordb.GraphRetriever import GraphRetriever
import time, asyncio

class DotRAG(GraphRetriever):
    def __init__(self, *args, **kwargs):
         super().__init__(*args, **kwargs)

    async def find_anchors(self, query):
         pass

    async def rank_anchors(self, anchors):
         pass

    async def traverse_with_logical_reasoning(self, anchor):
         pass

    async def low_level_search(self):
         pass

    async def high_level_search(self):
         pass

    async def generate(self):
         pass

    async def aquery(self, verbose=False):
        pass