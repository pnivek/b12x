# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Static persistent work scheduler.

This is a narrow fork of the grouped GEMM scheduler used by the fused MoE
kernel. The static MoE path only needs persistent assignment over
`(m_tile, n_tile, expert)`
with per-expert `row_counts`, so this module keeps just that subset instead of
depending on the full grouped GEMM scheduler implementation.
"""

from __future__ import annotations

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Integer, dsl_user_op, extract_mlir_values, new_from_mlir_values
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo


class StaticSchedulerParams:
    def __init__(
        self,
        row_counts: cute.Tensor,
        active_row_counts: cute.Tensor,
        active_experts: cute.Tensor,
        active_expert_count: cute.Tensor,
        c_tiler: Tuple[int, int],
        num_tiles_n: Int32,
        cluster_shape_mnk: cute.Shape,
        *,
        loc=None,
        ip=None,
    ):
        if cluster_shape_mnk[2] != 1:
            raise ValueError(f"unsupported cluster_shape_k {cluster_shape_mnk[2]}")

        self.row_counts = row_counts
        self.active_row_counts = active_row_counts
        self.active_experts = active_experts
        self.active_expert_count = active_expert_count
        self.c_tiler = c_tiler
        self.num_tiles_n = num_tiles_n
        self._cluster_shape_mnk = cluster_shape_mnk
        self.cluster_shape_mn = cluster_shape_mnk[:2]
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.row_counts,
            self.active_row_counts,
            self.active_experts,
            self.active_expert_count,
            self.c_tiler,
            self.num_tiles_n,
            self._cluster_shape_mnk,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.row_counts,
                self.active_row_counts,
                self.active_experts,
                self.active_expert_count,
                self.c_tiler,
                self.num_tiles_n,
                self._cluster_shape_mnk,
            ],
            self._values_pos,
            strict=True,
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticSchedulerParams(*(tuple(obj_list)), loc=self._loc)

    @dsl_user_op
    def get_grid_shape(
        self, max_active_clusters: Int32, *, loc=None, ip=None
    ) -> Tuple[Integer, Integer, Integer]:
        return (*self.cluster_shape_mn, max_active_clusters)


class StaticScheduler:
    def __init__(
        self,
        params: StaticSchedulerParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        current_batch_idx: Int32,
        accum_tile_m: Int32,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
    ):
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self._current_batch_idx = current_batch_idx
        self._accum_tile_m = accum_tile_m
        self.cta_id_in_cluster = cta_id_in_cluster
        self._num_tiles_executed = num_tiles_executed

    def __extract_mlir_values__(self):
        values = extract_mlir_values(self.num_persistent_clusters)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self._current_batch_idx))
        values.extend(extract_mlir_values(self._accum_tile_m))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        return values

    def __new_from_mlir_values__(self, values) -> "StaticScheduler":
        assert len(values) == 8
        new_num_persistent_clusters = new_from_mlir_values(
            self.num_persistent_clusters, [values[0]]
        )
        new_current_work_linear_idx = new_from_mlir_values(
            self._current_work_linear_idx, [values[1]]
        )
        new_current_batch_idx = new_from_mlir_values(
            self._current_batch_idx, [values[2]]
        )
        new_accum_tile_m = new_from_mlir_values(self._accum_tile_m, [values[3]])
        new_cta_id_in_cluster = new_from_mlir_values(
            self.cta_id_in_cluster, values[4:7]
        )
        new_num_tiles_executed = new_from_mlir_values(
            self._num_tiles_executed, [values[7]]
        )
        return StaticScheduler(
            self.params,
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_current_batch_idx,
            new_accum_tile_m,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
        )

    @dsl_user_op
    @staticmethod
    def create(
        params: StaticSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        *,
        loc=None,
        ip=None,
    ):
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(
            params.cluster_shape_mn, loc=loc, ip=ip
        )

        bidx, bidy, bidz = block_idx
        current_work_linear_idx = Int32(bidz)
        current_batch_idx = Int32(0)
        accum_tile_m = Int32(0)
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        num_tiles_executed = Int32(0)
        return StaticScheduler(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            current_batch_idx,
            accum_tile_m,
            cta_id_in_cluster,
            num_tiles_executed,
        )

    @staticmethod
    def get_grid_shape(
        params: StaticSchedulerParams,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Integer, Integer, Integer]:
        return params.get_grid_shape(max_active_clusters, loc=loc, ip=ip)

    @cute.jit
    def _get_current_work_for_linear_idx(
        self,
        current_work_linear_idx: Int32,
    ) -> WorkTileInfo:
        num_tiles_n = self.params.num_tiles_n
        accum_tile_m = self._accum_tile_m
        batch_idx = self._current_batch_idx
        num_active_experts = self.params.active_expert_count[Int32(0)]

        while (
            batch_idx < num_active_experts
            and (
                accum_tile_m
                + cute.ceil_div(self.params.active_row_counts[batch_idx], self.params.c_tiler[0])
            )
            * num_tiles_n
            <= current_work_linear_idx
        ):
            accum_tile_m += cute.ceil_div(
                self.params.active_row_counts[batch_idx], self.params.c_tiler[0]
            )
            batch_idx += Int32(1)

        self._accum_tile_m = accum_tile_m
        self._current_batch_idx = batch_idx

        is_valid = self._current_batch_idx < num_active_experts
        if is_valid:
            is_valid = (
                self._accum_tile_m
                + cute.ceil_div(
                    self.params.active_row_counts[self._current_batch_idx],
                    self.params.c_tiler[0],
                )
            ) * num_tiles_n > current_work_linear_idx

        cur_cluster_coord = (
            current_work_linear_idx // num_tiles_n - self._accum_tile_m,
            current_work_linear_idx % num_tiles_n,
            self._current_batch_idx,
        )
        cur_tile_coord = tuple(
            Int32(x) * Int32(z) + Int32(y)
            for x, y, z in zip(
                cur_cluster_coord,
                self.cta_id_in_cluster,
                (*self.params.cluster_shape_mn, Int32(1)),
                strict=True,
            )
        )
        return WorkTileInfo(cur_tile_coord, is_valid)

    @dsl_user_op
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self._get_current_work_for_linear_idx(self._current_work_linear_idx)

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        self._current_work_linear_idx += Int32(advance_count) * Int32(
            self.num_persistent_clusters
        )
        self._num_tiles_executed += Int32(1)

    @property
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed


__all__ = ["StaticScheduler", "StaticSchedulerParams", "WorkTileInfo"]
