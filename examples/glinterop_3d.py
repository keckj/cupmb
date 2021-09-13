import os, sys, ctypes

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "DEBUG"

from cupmb import ParallelMarchingBlocks, BufferType, Backend

import numpy as np
import pyrr
import glfw
from OpenGL.GL import *  # noqa F403
import OpenGL.GL.shaders

VERTEX_SHADER = """
#version 330

in vec3 position;
out vec3 newColor;

uniform mat4 transform;

void main() {
    gl_Position = transform * vec4(position, 1.0f);
}
"""


FRAGMENT_SHADER = """
#version 330

out vec4 outColor;

void main() {
    outColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}
"""

fdef = """
template <typename T, typename I, I Nx, I Ny, I Nz>
__device__ __forceinline__ T scalar(
        const T& x,  const T& y,  const T& z,
        const I& ix, const I& iy, const I& iz) {
    return (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) + (z-0.5)*(z-0.5) - 0.5*0.5;
}
"""


def setup_buffers():
    import pycuda
    import pycuda.gl.autoinit

    max_vertices = 5 * 1024 * 1024
    # 120 MiB
    max_indices = 5 * 1024 * 1024
    # 120 MiB

    ftype = np.float32
    itype = np.uint32

    vertex_bytes = 3 * max_vertices * ftype(0).itemsize
    index_bytes = 3 * max_indices * itype(0).itemsize

    flag = pycuda.gl.graphics_map_flags.WRITE_DISCARD

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertex_bytes, None, GL_DYNAMIC_DRAW)
    shared_vertex_buffer = pycuda.gl.RegisteredBuffer(int(VBO), flag)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_bytes, None, GL_DYNAMIC_DRAW)
    shared_index_buffer = pycuda.gl.RegisteredBuffer(int(EBO), flag)

    pmb = ParallelMarchingBlocks(
        3, fdef, ftype=ftype, itype=itype, backend=Backend.PYCUDA
    )

    def as_pycuda_buffer(mapped_buffer, backend=pmb.backend):
        return backend.Buffer(*mapped_buffer.device_ptr_and_size())
        # mem = cp.cuda.memory.UnownedMemory(int_ptr, size, mapped_buffer)
        # ptr = cp.cuda.memory.MemoryPointer(mem, 0)

    vertex_mapping = shared_vertex_buffer.map()
    index_mapping = shared_index_buffer.map()
    mapped_vertex_buffer = as_pycuda_buffer(vertex_mapping)
    mapped_index_buffer = as_pycuda_buffer(index_mapping)
    assert mapped_vertex_buffer.size == vertex_bytes
    assert mapped_index_buffer.size == index_bytes

    pmb.vertex_buffer = mapped_vertex_buffer
    pmb.vertex_index_buffer = mapped_index_buffer
    pmb.lock_buffer(BufferType.VERTICES)
    pmb.lock_buffer(BufferType.VERTEX_INDICES)
    pmb.setup()

    assert pmb.allocated_vertex_count == max_vertices, (
        pmb.allocated_vertex_count,
        max_vertices,
    )
    assert pmb.allocated_vertex_index_count == max_indices, (
        pmb.allocated_vertex_index_count,
        max_indices,
    )
    vertex_mapping.unmap()
    index_mapping.unmap()

    def update_buffers(
        shared_index_buffer=shared_index_buffer,
        shared_vertex_buffer=shared_vertex_buffer,
        as_pycuda_buffer=as_pycuda_buffer,
        pmb=pmb,
    ):
        vertex_mapping = shared_vertex_buffer.map()
        index_mapping = shared_index_buffer.map()
        mapped_vertex_buffer = as_pycuda_buffer(vertex_mapping)
        mapped_index_buffer = as_pycuda_buffer(index_mapping)
        pmb.replace_buffer(BufferType.VERTICES, mapped_vertex_buffer)
        pmb.replace_buffer(BufferType.VERTEX_INDICES, mapped_index_buffer)
        (_, I) = pmb()
        Ntriangles = 0 if (I is None) else I.shape[0]
        vertex_mapping.unmap()
        index_mapping.unmap()
        return Ntriangles

    return update_buffers


def main():
    if not glfw.init():
        return

    window = glfw.create_window(
        800, 800, "Cupmb Cuda/OpenGL interop example", None, None
    )

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
    )

    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    glUseProgram(shader)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    update_buffers = setup_buffers()

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        t = glfw.get_time()
        rot_x = pyrr.Matrix44.from_x_rotation(0.1 * t)
        rot_y = pyrr.Matrix44.from_y_rotation(t)

        transformLoc = glGetUniformLocation(shader, "transform")
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)

        Ntriangles = update_buffers()
        glDrawElements(GL_TRIANGLES, int(Ntriangles), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    sys.exit(main())
