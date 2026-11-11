"""Entity extraction accepts custom templates and independently optional examples."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from dotvectordb.graph.entity_indexer import EntityIndexer
from dotvectordb.construction_prompts.entity_prompts import (
    ENTITY_RECOGNITION_PROMPT, INPUT_EXAMPLE, OUTPUT_EXAMPLE, ExtractedEntitiesResponse,
)


class PromptCaptured(BaseException):
    """Stop at the model boundary before any database writes."""


class EntityPromptTests(unittest.IsolatedAsyncioTestCase):
    async def check_prompt(self, options, template, input_example, output_example):
        model = AsyncMock(side_effect=PromptCaptured)
        indexer = EntityIndexer(
            conn=object(), lock=object(), language_model=model,
            embedding_model=SimpleNamespace(model_name='test'), hash_func=lambda text: 123,
            entity_groups={'Person': 'People'}, add_embedding_func=AsyncMock(),
            summary_language='English', **options,
        )
        with self.assertRaises(PromptCaptured):
            await indexer.extract_entity('1: Person', {1: 'Person'}, 'Example chunk')
        model.assert_awaited_once_with(
            template.format(entity_types='1: Person', chunk='Example chunk',
                            input_example=input_example, output_example=output_example, language='English'),
            structuredInput=ExtractedEntitiesResponse,
        )

    async def test_defaults_preserve_existing_prompt(self):
        await self.check_prompt({}, ENTITY_RECOGNITION_PROMPT, INPUT_EXAMPLE, OUTPUT_EXAMPLE)

    async def test_custom_prompt_and_examples(self):
        template = '{entity_types}\n{chunk}\n{input_example}\n{output_example}\n{language}'
        await self.check_prompt(
            dict(entity_extract_prompt=template, entity_input_example='Custom input', entity_output_example='Custom output'),
            template, 'Custom input', 'Custom output',
        )

    async def test_custom_examples_with_default_template_and_empty_example(self):
        await self.check_prompt(
            dict(entity_input_example='', entity_output_example='Custom output'),
            ENTITY_RECOGNITION_PROMPT, '', 'Custom output',
        )


if __name__ == '__main__':
    unittest.main()
