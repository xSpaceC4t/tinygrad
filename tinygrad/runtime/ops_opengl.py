from __future__ import annotations
from typing import Optional, cast
import ctypes, functools, hashlib, contextlib
from tinygrad.runtime.autogen import opencl as cl
from tinygrad.helpers import init_c_var, to_char_p_p, from_mv, OSX, DEBUG, getenv, mv_address
from tinygrad.device import BufferSpec, LRUAllocator, Compiled, Compiler, CompileError
import moderngl as mgl
import numpy as np
from tinygrad.renderer.glsl import OpenGLRenderer

ctx = mgl.create_standalone_context()

class GLProgram:
  def __init__(self, device:GLDevice, name:str, lib:bytes):
    self.lib = lib
    self.cached_shader = None
  def __call__(self, *bufs:tuple[ctypes._CData, BufferSpec], global_size:tuple[int,int,int]=(1,1,1), local_size:Optional[tuple[int,int,int]]=None, vals:tuple[int, ...]=(), wait=False) -> Optional[float]:  # noqa: E501
    if not self.cached_shader:
      self.cached_shader = ctx.compute_shader(self.lib.decode()) 
    shader = self.cached_shader
    # shader = ctx.compute_shader(self.lib.decode()) 
    i = 0
    for buf in bufs:
      buf.bind_to_storage_buffer(i)
      i += 1
    if wait:
      query = ctx.query(time=True)
      with query:
        shader.run(group_x=global_size[0], group_y=global_size[1], group_z=global_size[2])
      return query.elapsed / 1e9
    shader.run(group_x=global_size[0], group_y=global_size[1], group_z=global_size[2])
    ctx.finish()

class GLAllocator(LRUAllocator['GLDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> tuple[ctypes._CData, BufferSpec]:
    return ctx.buffer(reserve=size)
  def _free(self, opaque:tuple[ctypes._CData, BufferSpec], options:BufferSpec):
    opaque.release()
  def _copyin(self, dest:tuple[ctypes._CData, BufferSpec], src:memoryview):
    dest.write(src)
  def _copyout(self, dest:memoryview, src:tuple[ctypes._CData, BufferSpec]):
    src.read_into(dest)

class GLDevice(Compiled):
  def __init__(self, device:str=""):
    renderer = OpenGLRenderer()
    super().__init__(device, GLAllocator(self), renderer, Compiler(self), functools.partial(GLProgram, self))
  def synchronize(self):
    print("SYNC")
    ctx.finish()

OPENGLDevice = GLDevice # for legacy reasons
