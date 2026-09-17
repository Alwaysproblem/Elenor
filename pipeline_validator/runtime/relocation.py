"""Mechanical relocation of compiler-listed launch fields; no graph analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ..compiled_program import Relocation
from ..execution_ir import ExecTileGroupTask


def relocate_task(
  task: ExecTileGroupTask, relocations: Sequence[Relocation], event_prefix: str, queue_offset: int
) -> ExecTileGroupTask:
  actions = list(task.actions)
  streams = list(task.streams)
  roles = dict(task.role_bindings)
  event_uses = task.event_uses
  completion = task.completion_event
  for relocation in relocations:
    if relocation.binding_id != task.binding_id:
      continue
    scope, field, kind = relocation.scope, relocation.field, relocation.kind
    if kind == "binding":
      # Actual addresses stay in the launch's separately validated formal map.
      continue
    if scope == "task" and field == "completion_event" and kind == "event":
      completion = event_prefix + completion
    elif scope == "task" and field == "event_uses" and kind == "event_map":
      event_uses = {event_prefix + event: count for event, count in task.event_uses.items()}
    elif scope == "group":
      action = actions[relocation.ordinal]
      if field == "dst" and kind == "event":
        if not action.dst:
          raise ValueError("relocation names an empty event producer")
        action = replace(action, dst=event_prefix + action.dst)
      elif field == "dependencies" and kind == "event_tuple":
        action = replace(action, dependencies=tuple(event_prefix + event for event in action.dependencies))
      elif field == "args" and kind == "event_tuple":
        action = replace(action, args=tuple(event_prefix + event for event in action.args))
      elif field == "args" and kind == "dispatch_events":
        request = action.args[0]
        action = replace(
          action,
          args=(
            replace(
              request,
              input_released_event=event_prefix + request.input_released_event
              if request.input_released_event
              else "",
              output_ready_event=event_prefix + request.output_ready_event
              if request.output_ready_event
              else "",
            ),
          ),
        )
      elif field == "args" and kind == "release_events":
        request = action.args[0]
        action = replace(
          action,
          args=(
            replace(
              request, dependency_events=tuple(event_prefix + event for event in request.dependency_events)
            ),
          ),
        )
      elif field == "args" and kind == "profile_frontier":
        command = action.args[0]
        action = replace(
          action,
          args=(replace(command, frontier=tuple(event_prefix + event for event in command.frontier)),),
        )
      elif field == "args" and kind == "maintenance_dependencies":
        command = action.args[0]
        action = replace(
          action,
          args=(
            replace(command, dependencies=tuple(event_prefix + event for event in command.dependencies)),
          ),
        )
      elif field == "args" and kind == "queue":
        action = replace(action, args=(action.args[0] + queue_offset, *action.args[1:]))
      else:
        raise ValueError(f"unsupported Group relocation {field}/{kind}")
      actions[relocation.ordinal] = action
    elif scope == "stream" and field == "queue_id" and kind == "queue":
      stream = streams[relocation.ordinal]
      streams[relocation.ordinal] = replace(stream, queue_id=stream.queue_id + queue_offset)
    elif scope == "role" and field in ("in_stream", "out_stream") and kind == "queue":
      role = roles[relocation.role_id]
      queue_id = getattr(role, field)
      if queue_id is None:
        raise ValueError("queue relocation names an absent stream")
      roles[relocation.role_id] = replace(role, **{field: queue_id + queue_offset})
    elif scope == "tile" and field == "args":
      role = roles[relocation.role_id]
      instructions = list(role.tile_program.insts)
      instruction = instructions[relocation.ordinal]
      arguments = list(instruction.args)
      if kind == "queue":
        arguments[0] += queue_offset
      elif kind == "event":
        if relocation.argument_index < 0:
          raise ValueError("event relocation requires exact argument index")
        arguments[relocation.argument_index] = event_prefix + arguments[relocation.argument_index]
      else:
        raise ValueError(f"unsupported Tile relocation {kind}")
      instructions[relocation.ordinal] = replace(instruction, args=tuple(arguments))
      roles[relocation.role_id] = replace(
        role, tile_program=replace(role.tile_program, insts=tuple(instructions))
      )
    else:
      raise ValueError(f"unsupported relocation {scope}/{field}/{kind}")
  return replace(
    task,
    actions=tuple(actions),
    streams=tuple(streams),
    role_bindings=roles,
    completion_event=completion,
    event_uses=event_uses,
  )
