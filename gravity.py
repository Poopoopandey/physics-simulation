import glfw
from OpenGL.GL import *
import numpy as np
import math
import ctypes
import time


VERTEX_SHADER = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out float lightIntensity;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    vec3 worldPos = (model * vec4(aPos, 1.0)).xyz;
    vec3 normal = normalize(aPos);
    vec3 dirToCenter = normalize(-worldPos);
    lightIntensity = max(dot(normal, dirToCenter), 0.15);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in float lightIntensity;
out vec4 FragColor;
uniform vec4 objectColor;
uniform bool isGrid;
uniform bool GLOW;

void main() {
    if (isGrid)
        FragColor = objectColor;
    else if (GLOW)
        FragColor = vec4(objectColor.rgb * 100000.0, objectColor.a);
    else {
        float fade = smoothstep(0.0, 10.0, lightIntensity * 10.0);
        FragColor = vec4(objectColor.rgb * fade, objectColor.a);
    }
}
"""


def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def create_shader_program():
    vs = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fs = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(program))
    return program

def spherical_to_cartesian(r, theta, phi):
    return np.array([
        r * math.sin(theta) * math.cos(phi),
        r * math.cos(theta),
        r * math.sin(theta) * math.sin(phi)
    ], dtype=np.float32)

class Object3D:
    def __init__(self, position, velocity, mass, density=3344, color=(1,0,0,1), glow=False):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.mass = mass
        self.density = density
        self.color = color
        self.glow = glow

        self.radius = ((3 * mass / density) / (4 * math.pi)) ** (1/3) / 30000
        self.vertices = self.generate_sphere()

        self.vao, self.vbo = self.create_buffers()

    def generate_sphere(self):
        vertices = []
        stacks, sectors = 10, 10
        for i in range(stacks):
            t1 = i / stacks * math.pi
            t2 = (i + 1) / stacks * math.pi
            for j in range(sectors):
                p1 = j / sectors * 2 * math.pi
                p2 = (j + 1) / sectors * 2 * math.pi

                v1 = spherical_to_cartesian(self.radius, t1, p1)
                v2 = spherical_to_cartesian(self.radius, t1, p2)
                v3 = spherical_to_cartesian(self.radius, t2, p1)
                v4 = spherical_to_cartesian(self.radius, t2, p2)

                vertices += v1.tolist() + v2.tolist() + v3.tolist()
                vertices += v2.tolist() + v4.tolist() + v3.tolist()

        return np.array(vertices, dtype=np.float32)

    def create_buffers(self):
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        return vao, vbo

    def update(self, dt):
        self.position += self.velocity * dt / 94


camera_pos = np.array([0, 1000, 5000], dtype=np.float32)
camera_front = np.array([0, 0, -1], dtype=np.float32)
camera_up = np.array([0, 1, 0], dtype=np.float32)

def main():
    glfw.init()
    window = glfw.create_window(800, 600, "Python Gravity Sim", None, None)
    glfw.make_context_current(window)

    glEnable(GL_DEPTH_TEST)
    program = create_shader_program()
    glUseProgram(program)

    objs = [
        Object3D([-5000, 650, -350], [0, 0, 1500], 5.97e22, color=(0,1,1,1)),
        Object3D([5000, 650, -350], [0, 0, -1500], 5.97e22, color=(0,1,1,1)),
        Object3D([0, 0, -350], [0,0,0], 1.99e25, color=(1,0.9,0.2,1), glow=True)
    ]

    last_time = time.time()

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        now = time.time()
        dt = now - last_time
        last_time = now

        for obj in objs:
            obj.update(dt)
            model = np.eye(4, dtype=np.float32)
            model[:3, 3] = obj.position

            glUniformMatrix4fv(glGetUniformLocation(program, "model"), 1, GL_FALSE, model)
            glUniform4f(glGetUniformLocation(program, "objectColor"), *obj.color)
            glUniform1i(glGetUniformLocation(program, "GLOW"), obj.glow)

            glBindVertexArray(obj.vao)
            glDrawArrays(GL_TRIANGLES, 0, len(obj.vertices)//3)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()

