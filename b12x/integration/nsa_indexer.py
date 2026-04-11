"""Public NSA indexer integration surface."""

from __future__ import annotations

from b12x.attention.nsa_indexer import (
    NSAIndexerDecodeMetadata,
    NSAIndexerPagedDecodeMetadata,
    NSAIndexerExtendMetadata,
    clear_nsa_indexer_caches,
    pack_nsa_index_k_cache_reference,
    sparse_nsa_index_decode_logits_paged,
    sparse_nsa_index_decode_topk,
    sparse_nsa_index_extend_topk,
    sparse_nsa_paged_logits_reference,
    sparse_nsa_index_reference,
    unpack_nsa_index_k_cache_reference,
)

__all__ = [
    "NSAIndexerDecodeMetadata",
    "NSAIndexerPagedDecodeMetadata",
    "NSAIndexerExtendMetadata",
    "clear_nsa_indexer_caches",
    "pack_nsa_index_k_cache_reference",
    "sparse_nsa_index_decode_logits_paged",
    "sparse_nsa_paged_logits_reference",
    "sparse_nsa_index_decode_topk",
    "sparse_nsa_index_extend_topk",
    "sparse_nsa_index_reference",
    "unpack_nsa_index_k_cache_reference",
]
