from __future__ import annotations

import pytest
from xdsl.utils.exceptions import ParseError, VerifyException

from pipeline_validator.dialects.elenor import NestAllocOp, NexusSharedRefOp, NexusSubmitContextOp
from pipeline_validator.workload_ir import (
  effective_submit_dependencies,
  parse_workload_ir,
  print_workload_ir,
)


SHARED_IR = '''builtin.module {
  tile.program @read_weight(
      %task : !nest.task,
      %weight : !nest.l2_buffer<4xi8>)
      resource_contract = #tile.resources<allowed_profiles = [0, 1],
          tile_l1_spm_bytes_per_context = 4> {
    %view = tile.subview %weight offsets = [0] sizes = [4] strides = [1]
        : !nest.l2_view<4xi8>
    %scratch = tile.alloc shape = [4] dtype = "i8" : !tile.l1_buffer<4xi8>
    %loaded = tile.load.async %view into %scratch : !tile.event<"load">
    tile.await %loaded
    tile.signal input_released(%task)
    tile.return
  }

  nest.context @loader(%W : !nest.global_memref<4xi8>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 0, l2_spm_bytes = 4, requested_contexts_per_tile = 1> {
    %w = nest.alloc slot = "W" role = "in" sharing = "readonly"
        shape = [4] dtype = "i8" : !nest.l2_buffer<4xi8>
    %src = nest.subview %W offsets = [0] sizes = [4] strides = [1]
        : !nest.global_view<4xi8>
    %prefetched = nest.dma.prefetch.async %src into %w : !nest.event<"prefetched">
    %published = nest.publish %w depends_on(%prefetched) : !nest.event<"published">
    nest.release %w depends_on(%prefetched, %published)
    nest.return
  }

  nest.context @reader(
      %OUT : !nest.global_memref<4xi8>,
      %weight : !nest.l2_buffer<4xi8>) placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 1, l2_spm_bytes = 0, requested_contexts_per_tile = 1> {
    %out_view = nest.subview %OUT offsets = [0] sizes = [4] strides = [1]
        : !nest.global_view<4xi8>
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %input_done, %no_writer = nest.dispatch.tasks.async
        @read_weight l1_mode = 1 tasks(%tasks) globals() bindings(%weight)
        ins(%weight) outs()
        signal_policy { input_released = #nest.aggregate<all_tasks> }
        : (!nest.event<"grid">, !nest.event<"input_done">, !nest.event<"">)
    %stored = nest.dma.store.async %weight into %out_view depends_on(%input_done)
        : !nest.event<"stored">
    nest.release %weight depends_on(%input_done, %stored)
    nest.return
  }

  nexus.program @run(
      %W : !nest.global_memref<4xi8>) {
    %loaded = nexus.submit_context.async @loader(%W) : !nexus.event<"loaded">
    %shared_w = nexus.shared.ref %loaded slot = "W" : !nest.l2_buffer<4xi8>
    %reader_done = nexus.submit_context.async @reader(%W, %shared_w)
        : !nexus.event<"reader_done">
    nexus.return
  }
}
'''


PRIVATE_IR = '''builtin.module {
  nest.context @private() placement = 1
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 0, l2_spm_bytes = 4, requested_contexts_per_tile = 1> {
    %w = nest.alloc slot = "W" role = "in" sharing = "private"
        shape = [4] dtype = "i8" : !nest.l2_buffer<4xi8>
    nest.release %w
    nest.return
  }
}
'''


def _rejected(text: str, message: str) -> None:
  with pytest.raises(VerifyException, match=message):
    parse_workload_ir(text, source_name="<l2-sharing-source-negative>")


def test_shared_source_roundtrips_and_implies_producer_completion() -> None:
  module = parse_workload_ir(SHARED_IR, source_name="<l2-sharing-source>")
  printed = print_workload_ir(module)
  assert 'sharing = "readonly"' in printed
  assert "nest.publish" in printed
  assert "nexus.shared.ref" in printed

  reparsed = parse_workload_ir(printed, source_name="<l2-sharing-source-roundtrip>")
  model = next(op for op in reparsed.body.block.ops if op.name == "nexus.program")
  operations = list(model.body.block.ops)
  loader = next(op for op in operations if isinstance(op, NexusSubmitContextOp))
  shared_ref = next(op for op in operations if isinstance(op, NexusSharedRefOp))
  reader = [op for op in operations if isinstance(op, NexusSubmitContextOp)][1]
  assert isinstance(shared_ref.producer.owner, NexusSubmitContextOp)
  assert not reader.depends_on
  assert effective_submit_dependencies(reader) == (loader.result,)
  assert reader.actuals[0].type == loader.actuals[0].type
  assert len(reader.actuals) == 2


def test_reference_is_bound_to_one_producer_submit_instance() -> None:
  text = SHARED_IR.replace(
    "nexus.program @run(\n      %W : !nest.global_memref<4xi8>) {",
    "nexus.program @run(\n      %W : !nest.global_memref<4xi8>,\n"
    "      %W2 : !nest.global_memref<4xi8>) {",
    1,
  ).replace(
    '    %loaded = nexus.submit_context.async @loader(%W) : !nexus.event<"loaded">',
    '    %loaded = nexus.submit_context.async @loader(%W) : !nexus.event<"loaded">\n'
    '    %loaded_again = nexus.submit_context.async @loader(%W2) : !nexus.event<"loaded_again">',
    1,
  )
  module = parse_workload_ir(text, source_name="<repeated-producer-template>")
  model = next(op for op in module.body.block.ops if op.name == "nexus.program")
  submits = [op for op in model.body.block.ops if isinstance(op, NexusSubmitContextOp)]
  shared_ref = next(op for op in model.body.block.ops if isinstance(op, NexusSharedRefOp))
  assert shared_ref.producer.owner is submits[0]
  assert effective_submit_dependencies(submits[2]) == (submits[0].result,)


def test_zero_consumer_readonly_export_is_valid() -> None:
  text = SHARED_IR.replace(
    '    %shared_w = nexus.shared.ref %loaded slot = "W" : !nest.l2_buffer<4xi8>\n'
    '    %reader_done = nexus.submit_context.async @reader(%W, %shared_w)\n'
    '        : !nexus.event<"reader_done">\n',
    "",
    1,
  )
  parse_workload_ir(text, source_name="<zero-consumer-export>")


def test_import_release_is_required_and_cannot_precede_a_later_use() -> None:
  missing = SHARED_IR.replace(
    "    nest.release %weight depends_on(%input_done, %stored)\n",
    "",
    1,
  )
  _rejected(missing, "readonly L2 import 'weight' requires exactly one nest.release")

  use_after = SHARED_IR.replace(
    "    %stored = nest.dma.store.async %weight into %out_view depends_on(%input_done)\n"
    '        : !nest.event<"stored">\n'
    "    nest.release %weight depends_on(%input_done, %stored)",
    "    nest.release %weight depends_on(%input_done)\n"
    "    %stored = nest.dma.store.async %weight into %out_view depends_on(%input_done)\n"
    '        : !nest.event<"stored">',
    1,
  )
  _rejected(use_after, "must follow every use")
  duplicate = SHARED_IR.replace(
    "    nest.release %weight depends_on(%input_done, %stored)\n",
    "    nest.release %weight depends_on(%input_done, %stored)\n"
    "    nest.release %weight depends_on(%input_done, %stored)\n",
    1,
  )
  _rejected(duplicate, "readonly L2 import 'weight' requires exactly one nest.release")

  shadow = SHARED_IR.replace(
    "    %out_view = nest.subview %OUT",
    '    %shadow = nest.alloc slot = "weight" role = "in" shape = [1] dtype = "i8"'
    " : !nest.l2_buffer<1xi8>\n"
    "    nest.release %shadow\n"
    "    %out_view = nest.subview %OUT",
    1,
  )
  _rejected(shadow, "duplicate L2 buffer slot or import name 'weight'")


def test_shared_formals_follow_globals_and_submit_actuals_match_categories() -> None:
  misordered = SHARED_IR.replace(
    "      %OUT : !nest.global_memref<4xi8>,\n      %weight : !nest.l2_buffer<4xi8>)",
    "      %weight : !nest.l2_buffer<4xi8>,\n      %OUT : !nest.global_memref<4xi8>)",
    1,
  ).replace("@reader(%W, %shared_w)", "@reader(%shared_w, %W)", 1)
  _rejected(misordered, "global formal 1 may not follow an L2 import")

  category_mismatch = SHARED_IR.replace(
    "@reader(%W, %shared_w)", "@reader(%shared_w, %W)", 1
  )
  _rejected(category_mismatch, "matching nexus.program HBM input")


def test_readonly_export_cannot_be_published_twice() -> None:
  text = SHARED_IR.replace(
    '    %published = nest.publish %w depends_on(%prefetched) : !nest.event<"published">\n',
    '    %published = nest.publish %w depends_on(%prefetched) : !nest.event<"published">\n'
    '    %published_again = nest.publish %w depends_on(%prefetched) '
    ': !nest.event<"published_again">\n',
    1,
  )
  _rejected(text, "readonly L2 slot 'W' requires exactly one nest.publish")


def test_readonly_import_cannot_be_reexported() -> None:
  text = SHARED_IR.replace(
    '    %stored = nest.dma.store.async %weight into %out_view depends_on(%input_done)',
    '    %reexported = nest.publish %weight depends_on(%input_done) '
    ': !nest.event<"reexported">\n'
    '    %stored = nest.dma.store.async %weight into %out_view depends_on(%input_done)',
    1,
  )
  _rejected(text, "readonly L2 import 'weight' may not be published or re-exported")


def test_alloc_private_default_and_readonly_parse_print() -> None:
  for source in (PRIVATE_IR, PRIVATE_IR.replace(' sharing = "private"', "")):
    module = parse_workload_ir(source, source_name="<private-l2-source>")
    context = next(op for op in module.body.block.ops if op.name == "nest.context")
    allocation = next(op for op in context.body.block.ops if isinstance(op, NestAllocOp))
    assert allocation.sharing.data == "private"
    printed = print_workload_ir(module)
    assert 'sharing = "private"' not in printed
    reparsed_context = next(
      op for op in parse_workload_ir(printed).body.block.ops if op.name == "nest.context"
    )
    assert next(
      op for op in reparsed_context.body.block.ops if isinstance(op, NestAllocOp)
    ).sharing.data == "private"


def test_alloc_rejects_unknown_sharing_value_during_parse() -> None:
  with pytest.raises(ParseError, match="sharing"):
    parse_workload_ir(SHARED_IR.replace('sharing = "readonly"', 'sharing = "mutable"', 1))


@pytest.mark.parametrize(
  ("text", "message"),
  [
    (
      SHARED_IR.replace('sharing = "readonly"', 'sharing = "private"', 1),
      "private L2 slot 'W' may not be published",
    ),
    (
      SHARED_IR.replace(
        'nest.publish %w depends_on(%prefetched) : !nest.event<"published">',
        'nest.publish %w : !nest.event<"published">',
        1,
      ),
      "nest.publish.*depend on exactly",
    ),
    (
      SHARED_IR.replace(
        'nest.release %w depends_on(%prefetched, %published)',
        'nest.release %w depends_on(%prefetched)',
        1,
      ),
      "nest.release.*required publish completion",
    ),
    (
      SHARED_IR.replace(
        'nexus.shared.ref %loaded slot = "W" : !nest.l2_buffer<4xi8>',
        'nexus.shared.ref %loaded slot = "missing" : !nest.l2_buffer<4xi8>',
        1,
      ),
      "unknown producer slot 'missing'",
    ),
    (
      SHARED_IR.replace(
        'nexus.shared.ref %loaded slot = "W" : !nest.l2_buffer<4xi8>',
        'nexus.shared.ref %loaded slot = "W" : !nest.l2_buffer<2xf16>',
        1,
      ),
      "shape and dtype",
    ),
  ],
)
def test_invalid_export_and_reference_contracts_are_source_errors(text: str, message: str) -> None:
  _rejected(text, message)


def test_readonly_export_requires_complete_initialization() -> None:
  addition = '''    %empty = nest.alloc slot = "empty" role = "in" sharing = "readonly"
        shape = [1] dtype = "i8" : !nest.l2_buffer<1xi8>
    %empty_pub = nest.publish %empty : !nest.event<"empty_pub">
    nest.release %empty depends_on(%empty_pub)
'''
  text = SHARED_IR.replace(
    "    nest.return\n  }\n\n  nest.context @reader",
    addition + "    nest.return\n  }\n\n  nest.context @reader",
    1,
  )
  _rejected(text, "readonly L2 slot 'empty' must be fully initialized")


def test_one_submit_cannot_import_the_same_backing_twice() -> None:
  text = SHARED_IR.replace(
    "%weight : !nest.l2_buffer<4xi8>) placement = 1",
    "%weight : !nest.l2_buffer<4xi8>,\n      %weight_again : !nest.l2_buffer<4xi8>) placement = 1",
    1,
  ).replace(
    "    nest.release %weight depends_on(%input_done, %stored)\n    nest.return",
    "    nest.release %weight depends_on(%input_done, %stored)\n"
    "    nest.release %weight_again\n    nest.return",
    1,
  ).replace("@reader(%W, %shared_w)", "@reader(%W, %shared_w, %shared_w)", 1)
  _rejected(text, "may not import the same producer slot more than once")


def test_readonly_import_cannot_be_a_dispatch_write_destination() -> None:
  text = (
    SHARED_IR.replace("tile.load.async %view into %scratch", "tile.store.async %scratch into %view", 1)
    .replace("tile.signal input_released(%task)", "tile.signal output_ready(%task)", 1)
    .replace("ins(%weight) outs()", "ins() outs(%weight)", 1)
    .replace(
      "input_released = #nest.aggregate<all_tasks>",
      "output_ready = #nest.aggregate<all_tasks>",
      1,
    )
    .replace("%grid, %input_done, %no_writer", "%grid, %no_reader, %ready", 1)
    .replace(
      '!nest.event<"grid">, !nest.event<"input_done">, !nest.event<"">',
      '!nest.event<"grid">, !nest.event<"">, !nest.event<"ready">',
      1,
    )
    .replace("depends_on(%input_done)", "depends_on(%ready)", 1)
    .replace(
      "nest.release %weight depends_on(%input_done, %stored)",
      "nest.release %weight depends_on(%stored)",
      1,
    )
  )
  _rejected(text, "may not write a readonly L2 import")


def test_shared_reference_may_not_cross_an_l2_profile_epoch() -> None:
  switcher = '''  nest.context @switcher() placement = 1
      resource_contract = #nest.context_resources<l2_mode = 1, allowed_profiles = [1],
          logical_tasks = 0, l2_spm_bytes = 0, requested_contexts_per_tile = 1> {
    nest.return
  }
'''
  text = SHARED_IR.replace("  nexus.program @run(", switcher + "\n  nexus.program @run(", 1).replace(
    '    %shared_w = nexus.shared.ref %loaded slot = "W"',
    '    %switched = nexus.submit_context.async @switcher : !nexus.event<"switched">\n'
    '    %shared_w = nexus.shared.ref %loaded slot = "W"',
    1,
  )
  _rejected(text, "crosses the producer's L2 profile epoch")


CONTEXT_LOCAL_COPY = '''  tile.program @copy_row(
      %task : !nest.task, %src : !nest.l2_buffer<4x8192xi8>,
      %dst : !nest.l2_buffer<4x8192xi8>)
      resource_contract = #tile.resources<allowed_profiles = [0],
          tile_l1_spm_bytes_per_context = 8192> {
    %source = tile.subview %src task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 8192] strides = [1, 1]
        : !nest.l2_view<1x8192xi8>
    %destination = tile.subview %dst task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 8192] strides = [1, 1]
        : !nest.l2_view<1x8192xi8>
    %local = tile.alloc shape = [8192] dtype = "i8" alignment = 256
        : !tile.l1_buffer<8192xi8>
    %loaded = tile.load.async %source into %local : !tile.event<"loaded">
    tile.await %loaded
    tile.signal input_released(%task)
    %stored = tile.store.async %local into %destination : !tile.event<"stored">
    tile.await %stored
    tile.signal output_ready(%task)
    tile.return
  }
'''

CONTEXT_LOCAL_MUTATE = '''  tile.program @swap_halves(
      %task : !nest.task, %scratch : !nest.l2_buffer<4x8192xi8>)
      resource_contract = #tile.resources<allowed_profiles = [0],
          tile_l1_spm_bytes_per_context = 8192> {
    %first = tile.subview %scratch task = %task task_dim = 0
        offsets = [0, 0] sizes = [1, 4096] strides = [1, 1]
        : !nest.l2_view<1x4096xi8>
    %second = tile.subview %scratch task = %task task_dim = 0
        offsets = [0, 4096] sizes = [1, 4096] strides = [1, 1]
        : !nest.l2_view<1x4096xi8>
    %left = tile.alloc shape = [4096] dtype = "i8" alignment = 256
        : !tile.l1_buffer<4096xi8>
    %right = tile.alloc shape = [4096] dtype = "i8" alignment = 256
        : !tile.l1_buffer<4096xi8>
    %load_left = tile.load.async %first into %left : !tile.event<"load_left">
    %load_right = tile.load.async %second into %right : !tile.event<"load_right">
    tile.await %load_left, %load_right
    tile.signal input_released(%task)
    %store_right = tile.store.async %left into %second : !tile.event<"store_right">
    %store_left = tile.store.async %right into %first : !tile.event<"store_left">
    tile.await %store_right, %store_left
    tile.signal output_ready(%task)
    tile.return
  }
'''

CONTEXT_LOCAL_IR = f'''builtin.module {{
{CONTEXT_LOCAL_COPY}{CONTEXT_LOCAL_MUTATE}
  nest.context @mutate(
      %SOURCE : !nest.global_memref<4x8192xi8>,
      %OUT : !nest.global_memref<4x8192xi8>) placement = 15
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 12, l2_spm_bytes = 98304, requested_contexts_per_tile = 1> {{
    %input = nest.alloc slot = "input" role = "in" shape = [4, 8192] dtype = "i8"
        alignment = 256 : !nest.l2_buffer<4x8192xi8>
    %scratch = nest.alloc slot = "scratch" role = "inout" sharing = "context-local"
        shape = [4, 8192] dtype = "i8" alignment = 256 : !nest.l2_buffer<4x8192xi8>
    %result = nest.alloc slot = "result" role = "out" shape = [4, 8192] dtype = "i8"
        alignment = 256 : !nest.l2_buffer<4x8192xi8>
    %source = nest.subview %SOURCE offsets = [0, 0] sizes = [4, 8192] strides = [1, 1]
        : !nest.global_view<4x8192xi8>
    %output = nest.subview %OUT offsets = [0, 0] sizes = [4, 8192] strides = [1, 1]
        : !nest.global_view<4x8192xi8>
    %prefetched = nest.dma.prefetch.async %source into %input : !nest.event<"prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid_produce, %read_input, %ready_produce = nest.dispatch.tasks.async @copy_row
        l1_mode = 0 tasks(%tasks) globals() bindings(%input, %scratch)
        ins(%input) outs(%scratch)
        signal_policy {{ input_released = #nest.aggregate<all_tasks>,
                         output_ready = #nest.aggregate<all_tasks> }}
        depends_on(%prefetched)
        : (!nest.event<"grid_produce">, !nest.event<"read_input">,
           !nest.event<"ready_produce">)
    %grid_mutate, %read_mutate, %ready_mutate = nest.dispatch.tasks.async @swap_halves
        l1_mode = 0 tasks(%tasks) globals() bindings(%scratch)
        ins(%scratch) outs(%scratch)
        signal_policy {{ input_released = #nest.aggregate<all_tasks>,
                         output_ready = #nest.aggregate<all_tasks> }}
        depends_on(%ready_produce)
        : (!nest.event<"grid_mutate">, !nest.event<"read_mutate">,
           !nest.event<"ready_mutate">)
    %grid_read, %read_scratch, %ready_result = nest.dispatch.tasks.async @copy_row
        l1_mode = 0 tasks(%tasks) globals() bindings(%scratch, %result)
        ins(%scratch) outs(%result)
        signal_policy {{ input_released = #nest.aggregate<all_tasks>,
                         output_ready = #nest.aggregate<all_tasks> }}
        depends_on(%ready_mutate)
        : (!nest.event<"grid_read">, !nest.event<"read_scratch">,
           !nest.event<"ready_result">)
    nest.release %input depends_on(%read_input, %prefetched)
    nest.release %scratch depends_on(%ready_produce, %read_mutate, %ready_mutate, %read_scratch)
    %written = nest.dma.store.async %result into %output depends_on(%ready_result)
        : !nest.event<"written">
    nest.release %result depends_on(%written)
    nest.await %grid_produce, %grid_mutate, %grid_read, %written
    nest.return
  }}
  nexus.program @run_local(
      %SOURCE : !nest.global_memref<4x8192xi8>,
      %OUT : !nest.global_memref<4x8192xi8>) {{
    %done = nexus.submit_context.async @mutate(%SOURCE, %OUT) : !nexus.event<"done">
    nexus.await %done
    nexus.return
  }}
}}
'''


CONTEXT_LOCAL_WRITER_ONLY_IR = f'''builtin.module {{
{CONTEXT_LOCAL_COPY}
  nest.context @produce(%SOURCE : !nest.global_memref<4x8192xi8>) placement = 15
      resource_contract = #nest.context_resources<l2_mode = 0, allowed_profiles = [0],
          logical_tasks = 4, l2_spm_bytes = 65536, requested_contexts_per_tile = 1> {{
    %input = nest.alloc slot = "input" role = "in" shape = [4, 8192] dtype = "i8"
        alignment = 256 : !nest.l2_buffer<4x8192xi8>
    %scratch = nest.alloc slot = "scratch" role = "out" sharing = "context-local"
        shape = [4, 8192] dtype = "i8" alignment = 256 : !nest.l2_buffer<4x8192xi8>
    %source = nest.subview %SOURCE offsets = [0, 0] sizes = [4, 8192] strides = [1, 1]
        : !nest.global_view<4x8192xi8>
    %prefetched = nest.dma.prefetch.async %source into %input : !nest.event<"prefetched">
    %tasks = nest.task.range from = 0 to = 4 : !nest.task_range
    %grid, %read_input, %ready_produce = nest.dispatch.tasks.async @copy_row
        l1_mode = 0 tasks(%tasks) globals() bindings(%input, %scratch)
        ins(%input) outs(%scratch)
        signal_policy {{ input_released = #nest.aggregate<all_tasks>,
                         output_ready = #nest.aggregate<all_tasks> }}
        depends_on(%prefetched)
        : (!nest.event<"grid">, !nest.event<"read_input">,
           !nest.event<"ready_produce">)
    nest.release %input depends_on(%read_input, %prefetched)
    nest.release %scratch depends_on(%ready_produce)
    nest.await %grid
    nest.return
  }}
  nexus.program @run_writer(%SOURCE : !nest.global_memref<4x8192xi8>) {{
    %done = nexus.submit_context.async @produce(%SOURCE) : !nexus.event<"done">
    nexus.await %done
    nexus.return
  }}
}}
'''


def test_context_local_source_roundtrip_and_unstored_output() -> None:
  for source in (CONTEXT_LOCAL_IR, CONTEXT_LOCAL_WRITER_ONLY_IR):
    module = parse_workload_ir(source)
    printed = print_workload_ir(module)
    assert 'sharing = "context-local"' in printed
    assert "nest.publish" not in printed
    parse_workload_ir(printed)

  empty = PRIVATE_IR.replace('sharing = "private"', 'sharing = "context-local"')
  parse_workload_ir(empty)
  _rejected(
    CONTEXT_LOCAL_WRITER_ONLY_IR.replace('sharing = "context-local"', 'sharing = "private"'),
    "requires at least one nest.dma.store.async",
  )
  parse_workload_ir(
    CONTEXT_LOCAL_WRITER_ONLY_IR.replace('role = "out" sharing', 'role = "inout" sharing')
  )


def test_context_local_cannot_publish_or_export_to_another_context() -> None:
  published = CONTEXT_LOCAL_WRITER_ONLY_IR.replace(
    "    nest.release %scratch depends_on(%ready_produce)",
    '    %published = nest.publish %scratch depends_on(%ready_produce)'
    ' : !nest.event<"published">\n'
    "    nest.release %scratch depends_on(%ready_produce)",
    1,
  )
  _rejected(published, "context-local L2 slot 'scratch' may not be published")
  referenced = CONTEXT_LOCAL_WRITER_ONLY_IR.replace(
    "    nexus.await %done",
    '    %ref = nexus.shared.ref %done slot = "scratch" : !nest.l2_buffer<4x8192xi8>\n'
    "    nexus.await %done",
    1,
  )
  _rejected(referenced, "producer slot 'scratch' must be a readonly published export")


def test_context_local_respects_roles_and_single_release_lifecycle() -> None:
  _rejected(
    CONTEXT_LOCAL_WRITER_ONLY_IR.replace('role = "out" sharing', 'role = "in" sharing'),
    "input slot 'scratch' may not be written",
  )
  unused = '''    %unused = nest.alloc slot = "unused" role = "out" sharing = "context-local"
        shape = [4, 8192] dtype = "i8" : !nest.l2_buffer<4x8192xi8>
    nest.release %unused
'''
  _rejected(
    CONTEXT_LOCAL_WRITER_ONLY_IR.replace(
      "    nest.await %grid", unused + "    nest.await %grid", 1
    ),
    "requires at least one actual tile writer",
  )
  line = "    nest.release %scratch depends_on(%ready_produce)\n"
  _rejected(
    CONTEXT_LOCAL_WRITER_ONLY_IR.replace(line, "", 1),
    "requires exactly one nest.release",
  )
  _rejected(
    CONTEXT_LOCAL_WRITER_ONLY_IR.replace(line, line + line, 1),
    "requires exactly one nest.release",
  )
  use_after_release = CONTEXT_LOCAL_IR.replace(
    "    %written = nest.dma.store.async %result",
    '    %too_late = nest.dma.store.async %scratch into %output depends_on(%ready_mutate)\n'
    '        : !nest.event<"too_late">\n'
    "    %written = nest.dma.store.async %result",
    1,
  )
  _rejected(use_after_release, "must follow every use of the allocation")


@pytest.mark.parametrize(
  ("event", "grid"),
  [
    ("ready_produce", "grid_produce"),
    ("read_mutate", "grid_mutate"),
    ("ready_mutate", "grid_mutate"),
    ("read_scratch", "grid_read"),
  ],
)
def test_context_local_release_requires_each_reader_and_writer(event: str, grid: str) -> None:
  deps = "%ready_produce, %read_mutate, %ready_mutate, %read_scratch"
  _rejected(
    CONTEXT_LOCAL_IR.replace(deps, deps.replace(f"%{event}", "").replace(", ,", ",").strip(", ")),
    "context-local slot 'scratch' must depend on exactly",
  )
  _rejected(
    CONTEXT_LOCAL_IR.replace(deps, deps.replace(f"%{event}", f"%{grid}")),
    "context-local slot 'scratch' must depend on exactly",
  )


def test_context_local_release_requires_prefetch_and_distinct_events() -> None:
  prefetched_input = CONTEXT_LOCAL_IR.replace(
    'slot = "input" role = "in" shape',
    'slot = "input" role = "in" sharing = "context-local" shape',
    1,
  )
  _rejected(
    prefetched_input.replace(
      "nest.release %input depends_on(%read_input, %prefetched)",
      "nest.release %input depends_on(%read_input)",
      1,
    ),
    "context-local slot 'input' must depend on exactly",
  )
  _rejected(
    CONTEXT_LOCAL_IR.replace(
      "depends_on(%ready_produce, %read_mutate, %ready_mutate, %read_scratch)",
      "depends_on(%ready_produce, %read_mutate, %ready_mutate, %read_scratch, %ready_produce)",
      1,
    ),
    "context-local slot 'scratch' must depend on exactly",
  )


def test_context_local_final_store_must_cover_later_writers() -> None:
  stored_before_mutation = CONTEXT_LOCAL_IR.replace(
    "    %grid_mutate, %read_mutate, %ready_mutate =",
    "    %saved = nest.dma.store.async %scratch into %output"
    " depends_on(%ready_produce)\n"
    '        : !nest.event<"saved">\n'
    "    %grid_mutate, %read_mutate, %ready_mutate =",
    1,
  ).replace(
    "depends_on(%ready_produce, %read_mutate, %ready_mutate, %read_scratch)",
    "depends_on(%ready_produce, %read_mutate, %ready_mutate, %read_scratch, %saved)",
    1,
  )
  _rejected(stored_before_mutation, "final store of slot 'scratch' must depend on every actual writer")
