"""Custom relationship prompts preserve the existing structured model contract."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from dotvectordb.construction_prompts.relationship_prompts import (
    RELATIONSHIP_PROMPT, ENTITY_EXAMPLE, INPUT_EXAMPLE, OUTPUT_EXAMPLE, Relationships,
)


class PromptCaptured(BaseException):
    pass


source = Path(__file__).resolve().parents[1] / 'dotvectordb/graph/relationship_indexer.py'
tree = ast.parse(source.read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RelationshipIndexer')
methods = [node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
           and node.name in ('__init__', 'extract_relation')]
namespace = dict(os=os, json=json, RELATIONSHIP_PROMPT=RELATIONSHIP_PROMPT,
                 ENTITY_EXAMPLE=ENTITY_EXAMPLE, INPUT_EXAMPLE=INPUT_EXAMPLE,
                 OUTPUT_EXAMPLE=OUTPUT_EXAMPLE, Relationships=Relationships,
                 StaticModel=SimpleNamespace(from_pretrained=Mock(return_value=object())))
cls = ast.ClassDef(name='RelationshipUnderTest', bases=[], keywords=[], body=methods, decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), 'exec'), namespace)


class RelationshipPromptTests(unittest.IsolatedAsyncioTestCase):
    async def check_prompt(self, options, template, input_example, output_example, entity_example):
        model = AsyncMock(side_effect=PromptCaptured)
        owner = namespace['RelationshipUnderTest'](
            conn=object(), lock=object(), language_model=model, hash_func=lambda text: 123,
            embedding_model=SimpleNamespace(model_name='test'), add_embedding_func=AsyncMock(),
            summary_language='English', **options,
        )
        owner.get_entities_connected_to_chunk = AsyncMock(return_value=[10, 20])
        owner.prepare_local_entity_mapping_for_chunk = AsyncMock(return_value=({1: 10, 2: 20}, {1: 'A', 2: 'B'}))
        owner.get_sentence_highlighted_version_of_chunk = AsyncMock(return_value=('[L1] A knows B.', range(1, 2)))
        with self.assertRaises(PromptCaptured):
            await owner.extract_relation('Example chunk')
        model.assert_awaited_once_with(
            template.format(entities=json.dumps({1: 'A', 2: 'B'}), chunk='[L1] A knows B.',
                            input_example=input_example, output_example=output_example,
                            entity_example=entity_example, language='English'),
            structuredInput=Relationships,
        )

    async def test_defaults(self):
        await self.check_prompt({}, RELATIONSHIP_PROMPT, INPUT_EXAMPLE, OUTPUT_EXAMPLE, ENTITY_EXAMPLE)

    async def test_custom_prompt_and_examples(self):
        template = '{entities}\n{chunk}\n{entity_example}\n{input_example}\n{output_example}\n{language}'
        await self.check_prompt(dict(relationship_extract_prompt=template,
                                    relationship_input_example='Custom input', relationship_output_example='Custom output',
                                    relationship_entity_example='Custom entities'),
                                template, 'Custom input', 'Custom output', 'Custom entities')

    async def test_custom_examples_with_default_prompt(self):
        await self.check_prompt(dict(relationship_input_example='', relationship_output_example='Custom output'),
                                RELATIONSHIP_PROMPT, '', 'Custom output', ENTITY_EXAMPLE)


if __name__ == '__main__':
    unittest.main()
