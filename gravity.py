import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import ctypes

# =========================================================
# SHADERS (simple + correct)
# =========================================================

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
uniform vec4 color;

void main() {
    FragColor = color;
}
"""

# =========================================================
# MATH
# =========================================================

class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o): return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o): return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def __mul__(self, s): return Vec3(self.x*s, self.y*s, self.z*s)
    def __truediv__(self, s): return Vec3(self.x/s, self.y/s, self.z/s)

    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        l = self.length()
        return self / l if l > 0 else Vec3()

# =========================================================
# PHYSICS CONSTANTS (REALISTIC & STABLE)
# =========================================================

G = 1.0                 # Normalized gravity constant
SOFTENING = 200.0       # Prevents singularities
DT = 1.0 / 240.0        # FIXED timestep (key for accuracy)

# =========================================================
# BODY CLASS
# =========================================================

class Body:
    def __init__(self, pos, vel, mass, radius, color, fixed=False):
        self.pos = pos
        self.vel = vel
        self.acc = Vec3()
        self.mass = mass
        self.radius = radius
        self.color = color
        self.fixed = fixed
        self.trail = []

        self.vao, self.vbo, self.count = self._make_sphere()

    def _make_sphere(self):
        verts = []
        stacks, sectors = 16, 16

        for i in range(stacks):
            t1 = i * math.pi / stacks
            t2 = (i+1) * math.pi / stacks
            for j in range(sectors):
                p1 = j * 2*math.pi / sectors
                p2 = (j+1) * 2*math.pi / sectors

                def v(t, p):
                    return [
                        self.radius * math.sin(t)*math.cos(p),
                        self.radius * math.cos(t),
                        self.radius * math.sin(t)*math.sin(p)
                    ]

                verts += v(t1,p1)+v(t2,p1)+v(t2,p2)
                verts += v(t1,p1)+v(t2,p2)+v(t1,p2)

        verts = np.array(verts, dtype=np.float32)
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)

        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,12,ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        return vao, vbo, len(verts)//3

# =========================================================
# PHYSICS CORE (THIS IS THE IMPORTANT PART)
# =========================================================

def compute_acceleration(bodies):
    for b in bodies:
        b.acc = Vec3()

    for i, a in enumerate(bodies):
        if a.fixed: continue
        for j, b in enumerate(bodies):
            if i == j: continue

            r = b.pos - a.pos
            d2 = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING*SOFTENING
            inv_r3 = 1.0 / (d2 * math.sqrt(d2))
            a.acc += r * (G * b.mass * inv_r3)

def step_physics(bodies):
    # Velocity Verlet (symplectic, energy-stable)

    compute_acceleration(bodies)

    for b in bodies:
        if not b.fixed:
            b.vel += b.acc * (DT * 0.5)

    for b in bodies:
        if not b.fixed:
            b.pos += b.vel * DT
            b.trail.append(Vec3(b.pos.x, b.pos.y, b.pos.z))
            if len(b.trail) > 300:
                b.trail.pop(0)

    compute_acceleration(bodies)

    for b in bodies:
        if not b.fixed:
            b.vel += b.acc * (DT * 0.5)

# =========================================================
# CAMERA
# =========================================================

def look_at(eye, center, up):
    f = (center-eye).normalize()
    s = Vec3(
        f.y*up.z - f.z*up.y,
        f.z*up.x - f.x*up.z,
        f.x*up.y - f.y*up.x
    ).normalize()
    u = Vec3(
        s.y*f.z - s.z*f.y,
        s.z*f.x - s.x*f.z,
        s.x*f.y - s.y*f.x
    )

    M = np.identity(4, dtype=np.float32)
    M[0][:3] = [s.x, u.x, -f.x]
    M[1][:3] = [s.y, u.y, -f.y]
    M[2][:3] = [s.z, u.z, -f.z]
    M[3][0] = -s.x*eye.x - s.y*eye.y - s.z*eye.z
    M[3][1] = -u.x*eye.x - u.y*eye.y - u.z*eye.z
    M[3][2] = f.x*eye.x + f.y*eye.y + f.z*eye.z
    return M

# =========================================================
# MAIN
# =========================================================

def main():
    glfw.init()
    window = glfw.create_window(1200, 800, "Physically Correct Gravity", None, None)
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)

    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    # Camera
    cam_pos = Vec3(0, 6000, 16000)
    cam_front = Vec3(0, -0.3, -1).normalize()
    cam_up = Vec3(0,1,0)

    # Bodies
    bodies = []

    star = Body(Vec3(0,0,0), Vec3(), 5000, 1000, (1,1,0,1), fixed=True)
    bodies.append(star)

    def circular_v(r):
        return math.sqrt(G * star.mass / r)

    bodies.append(Body(Vec3(5000,0,0), Vec3(0,0,circular_v(5000)), 20, 250, (0.4,0.6,1,1)))
    bodies.append(Body(Vec3(-8000,0,0), Vec3(0,0,-circular_v(8000)), 40, 350, (1,0.4,0.3,1)))
    bodies.append(Body(Vec3(0,0,11000), Vec3(circular_v(11000),0,0), 80, 450, (0.9,0.8,0.4,1)))

    projection = np.identity(4, dtype=np.float32)
    f = 1/math.tan(math.radians(45)/2)
    projection[0][0]=f*(800/1200)
    projection[1][1]=f
    projection[2][2]=-1
    projection[2][3]=-1
    projection[3][2]=-0.2

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        step_physics(bodies)

        glUseProgram(shader)

        view = look_at(cam_pos, cam_pos+cam_front, cam_up)
        glUniformMatrix4fv(glGetUniformLocation(shader,"view"),1,GL_FALSE,view)
        glUniformMatrix4fv(glGetUniformLocation(shader,"projection"),1,GL_FALSE,projection)

        for b in bodies:
            model = np.identity(4, dtype=np.float32)
            model[3][:3] = [b.pos.x, b.pos.y, b.pos.z]
            glUniformMatrix4fv(glGetUniformLocation(shader,"model"),1,GL_FALSE,model)
            glUniform4f(glGetUniformLocation(shader,"color"), *b.color)
            glBindVertexArray(b.vao)
            glDrawArrays(GL_TRIANGLES,0,b.count)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
