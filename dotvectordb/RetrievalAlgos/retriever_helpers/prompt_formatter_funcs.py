import re
import time
def entity_ranking_key(entity):
    return (
            -int(entity["explicitly_mentioned"]),
            -entity["combined_score"],
            -entity["hybrid_score"],
            -entity["token_overlap_f1"],
            entity["entity_name"],
    )

def token_overlap_f1(left, right):
        """Return token-overlap F1 for two entity-name strings."""
        left_tokens = set(re.findall(r"\w+", str(left).casefold()))
        right_tokens = set(re.findall(r"\w+", str(right).casefold()))
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        if overlap == 0:
            return 0.0
        precision = overlap / len(left_tokens)
        recall = overlap / len(right_tokens)
        return 2 * precision * recall / (precision + recall)

def entity_name_is_explicitly_mentioned(query, entity_name):
        """Return whether every entity-name token appears in the query."""
        query_tokens = set(re.findall(r"\w+", str(query).casefold()))
        entity_tokens = set(
            re.findall(r"\w+", str(entity_name).casefold())
        )
        return bool(entity_tokens) and entity_tokens <= query_tokens

def retain_longest_explicit_entity_mentions(entities):
        """Suppress shorter entity names contained in a longer named anchor."""
        token_sets = {
            int(entity["entity_id"]): set(
                re.findall(r"\w+", str(entity["entity_name"]).casefold())
            )
            for entity in entities
        }
        return [
            entity
            for entity in entities
            if not any(
                token_sets[int(entity["entity_id"])]
                < token_sets[int(other["entity_id"])]
                for other in entities
                if int(other["entity_id"]) != int(entity["entity_id"])
            )
        ]

def create_phase_reporter(query):
        """Create a concurrency-safe phase timer for one query."""
        phase_start = time.perf_counter()
        total_start = phase_start

        def report_phase(name):
            nonlocal phase_start
            now = time.perf_counter()
            print(
                f"V300 PHASE | {name} | phase={now - phase_start:.4f}s | "
                f"total={now - total_start:.4f}s | query={query}"
            )
            phase_start = now

        return report_phase

def debug_print(*args, **kwargs):
        """Disabled debug-output hook for the retrieval pipeline."""
        return None

def format_retrieval_target_types(entity_types):
        """Format alternative allowed endpoint types as an OR expression."""
        targets = [f"{entity_type}s".upper() for entity_type in entity_types]
        if len(targets) < 2:
            return "".join(targets)
        return ", ".join(targets[:-1]) + f" OR {targets[-1]}"

def format_first_hop_structural_preview(
        cls,
        anchor_name,
        anchor_entity_types,
        reached_entity_types,
    ):
        """Describe a first hop as an irreversible frontier transition."""
        anchor_types = cls.format_retrieval_target_types(
            anchor_entity_types
        )
        reached_types = cls.format_retrieval_target_types(
            reached_entity_types
        )
        return (
            f"STEP 1: Starting from the known anchor {anchor_name} "
            f"({anchor_types}), find all directly connected {reached_types}. "
            f"CONSEQUENCE IF THIS APPROACH IS SELECTED: {anchor_name} CAN NO "
            f"LONGER BE EXPLORED AFTER THIS HOP. ONLY THE REACHED "
            f"{reached_types} CAN BE EXPLORED IN STEP 2. The anchor "
            f"{anchor_name} is removed from the branch's search state, so "
            f"STEP 2 cannot start from it, revisit it, or use any unused "
            f"direct connection from it. Every later result "
            f"must belong to one continuous chain beginning "
            f"{anchor_name} -> {reached_types}."
        )
def format_continuation_question(
        cls,
        anchor_name,
        reached_entity_types,
        next_entity_types,
    ):
        """Present every possible second-hop type as a numbered question."""
        reached_targets = cls.format_retrieval_target_types(
            reached_entity_types
        )
        options = [
            f"{option_number}) Look for all "
            f"{cls.format_retrieval_target_types(entity_type_group)} "
            f"associated with those {reached_targets}"
            for option_number, entity_type_group in enumerate(
                next_entity_types,
                start=1,
            )
        ]
        return (
            f"STEP 1 has removed {anchor_name} from the available search "
            f"state and permanently replaced it with the reached "
            f"{reached_targets}. STEP 2 has access only to those exact "
            f"{reached_targets}; it has no access to {anchor_name} and cannot "
            f"use any unused edge from that anchor. "
            f"Which STEP 2 continuation should be used?\n"
            + "\n".join(options)
        )

def path_ranking_key(path):
        return (
            # The first model selects the global trajectory priority. Nothing
            # downstream may promote a lower-ranked trajectory above it.
            -int(path.get("first_hop_trajectory_rank", 10**9)),
            float(path.get("first_hop_relevance_score", 0.0)),
            # Answer-bearing terminal types must survive the final contest.
            # The LLM predicts these once from the query; this does not alter
            # ordering between different first-hop trajectories.
            int(path.get("terminal_type_match", False)),
            # The evidence-depth model may prioritize paths only within the
            # trajectory priority established by the first model.
            int(path.get("model_preferred", True)),
            int(path.get("oracle_round_robin_priority", 0)),
            float(path["cumulative_score"]),
            len(path["relationship_path"]),
        )