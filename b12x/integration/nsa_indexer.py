"""Public NSA indexer integration surface."""

from __future__ import annotations

from b12x.attention.nsa_indexer import (
    NSAIndexerExtendLogitsMetadata,
    NSAIndexerPagedDecodeMetadata,
    clear_nsa_indexer_caches,
    get_paged_mqa_logits_metadata,
    make_nsa_indexer_contract_phantoms,
    pack_nsa_index_k_cache_reference,
    sparse_nsa_extend_logits_reference,
    sparse_nsa_index_decode_logits_paged,
    sparse_nsa_index_extend_logits,
    sparse_nsa_paged_logits_reference,
    unpack_nsa_index_k_cache_reference,
    uses_paged_mqa_schedule_metadata,
)

__all__ = [
    "NSAIndexerExtendLogitsMetadata",
    "NSAIndexerPagedDecodeMetadata",
    "clear_nsa_indexer_caches",
    "get_paged_mqa_logits_metadata",
    "make_nsa_indexer_contract_phantoms",
    "pack_nsa_index_k_cache_reference",
    "sparse_nsa_extend_logits_reference",
    "sparse_nsa_index_decode_logits_paged",
    "sparse_nsa_index_extend_logits",
    "sparse_nsa_paged_logits_reference",
    "unpack_nsa_index_k_cache_reference",
    "uses_paged_mqa_schedule_metadata",
]
