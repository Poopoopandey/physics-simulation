import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import ctypes
from collections import deque

vertex_shader_source = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 FragPos;
out vec3 Normal;
void main() {
    FragPos = aPos;
    Normal = normalize(aPos);
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
in vec3 FragPos;
in vec3 Normal;
out vec4 FragColor;
uniform vec4 objectColor;
uniform bool isGrid;
uniform bool isTrail;
uniform bool glow;
void main() {
    if (isGrid || isTrail) {
        FragColor = objectColor;
    } else if (glow) {
        FragColor = vec4(objectColor.rgb * 2.0, objectColor.a);
    } else {
        vec3 lightDir = normalize(vec3(0.5, 1.0, 0.5));
        float diff = max(dot(Normal, lightDir), 0.2);
        vec3 ambient = objectColor.rgb * 0.3;
        vec3 diffuse = objectColor.rgb * diff * 0.7;
        FragColor = vec4(ambient + diffuse, objectColor.a);
    }
}
"""

class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        length = self.length()
        if length > 0:
            return Vec3(self.x / length, self.y / length, self.z / length)
        return Vec3(0, 0, 0)
    
    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def copy(self):
        return Vec3(self.x, self.y, self.z)

class Vec4:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.r = self.x = float(x)
        self.g = self.y = float(y)
        self.b = self.z = float(z)
        self.a = self.w = float(w)

def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(fovy / 2.0)
    mat = np.identity(4, dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2.0 * far * near) / (near - far)
    mat[3, 2] = -1.0
    mat[3, 3] = 0.0
    return mat

def look_at(eye, center, up):
    f = (center - eye).normalize()
    s = f.cross(up).normalize()
    u = s.cross(f)
    
    result = np.identity(4, dtype=np.float32)
    result[0][0] = s.x
    result[1][0] = s.y
    result[2][0] = s.z
    result[0][1] = u.x
    result[1][1] = u.y
    result[2][1] = u.z
    result[0][2] = -f.x
    result[1][2] = -f.y
    result[2][2] = -f.z
    result[3][0] = -s.dot(eye)
    result[3][1] = -u.dot(eye)
    result[3][2] = f.dot(eye)
    return result

def translate(mat, vec):
    result = mat.copy()
    result[3][0] = vec.x
    result[3][1] = vec.y
    result[3][2] = vec.z
    return result

# Global variables
running = True
pause = False
show_trails = True
camera_pos = Vec3(0.0, 5000.0, 20000.0)
camera_front = Vec3(0.0, 0.0, -1.0)
camera_up = Vec3(0.0, 1.0, 0.0)
last_x, last_y = 400.0, 300.0
yaw, pitch = -90.0, -15.0
delta_time, last_frame = 0.0, 0.0
time_scale = 1.0

# Physics constants (scaled for simulation)
G = 1.0  # Simplified gravitational constant
objs = []
grid_vao = None

class Object:
    def __init__(self, position, velocity, mass, radius, color, glow=False, name=""):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.radius = radius
        self.color = color
        self.glow = glow
        self.name = name
        self.trail = deque(maxlen=100)  # Store last 100 positions
        
        self.vao = None
        self.vbo = None
        self.vertex_count = 0
        
        vertices = self.draw_sphere()
        self.vertex_count = len(vertices)
        self.create_vbo_vao(vertices)
        print(f"Created {name}: mass={mass:.2e}, radius={radius}, pos={position}")
    
    def draw_sphere(self):
        vertices = []
        stacks, sectors = 20, 20
        
        for i in range(stacks + 1):
            theta1 = (i / stacks) * math.pi
            theta2 = ((i + 1) / stacks) * math.pi
            
            for j in range(sectors):
                phi1 = (j / sectors) * 2 * math.pi
                phi2 = ((j + 1) / sectors) * 2 * math.pi
                
                v1 = Vec3(
                    self.radius * math.sin(theta1) * math.cos(phi1),
                    self.radius * math.cos(theta1),
                    self.radius * math.sin(theta1) * math.sin(phi1)
                )
                v2 = Vec3(
                    self.radius * math.sin(theta1) * math.cos(phi2),
                    self.radius * math.cos(theta1),
                    self.radius * math.sin(theta1) * math.sin(phi2)
                )
                v3 = Vec3(
                    self.radius * math.sin(theta2) * math.cos(phi1),
                    self.radius * math.cos(theta2),
                    self.radius * math.sin(theta2) * math.sin(phi1)
                )
                v4 = Vec3(
                    self.radius * math.sin(theta2) * math.cos(phi2),
                    self.radius * math.cos(theta2),
                    self.radius * math.sin(theta2) * math.sin(phi2)
                )
                
                vertices.extend([v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z])
                vertices.extend([v2.x, v2.y, v2.z, v4.x, v4.y, v4.z, v3.x, v3.y, v3.z])
        
        return vertices
    
    def create_vbo_vao(self, vertices):
        vertices_array = np.array(vertices, dtype=np.float32)
        
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices_array.nbytes, vertices_array, GL_STATIC_DRAW)
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)
    
    def update(self, dt):
        # Update position
        self.position = self.position + self.velocity * dt
        
        # Add to trail every few frames
        if len(self.trail) == 0 or (self.position - self.trail[-1]).length() > self.radius * 0.5:
            self.trail.append(self.position.copy())
    
    def apply_force(self, force_vec):
        # F = ma, so a = F/m
        acceleration = force_vec * (1.0 / self.mass)
        self.velocity = self.velocity + acceleration
    
    def cleanup(self):
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])


def create_grid(size=20000, divisions=20):
    vertices = []
    step = size / divisions
    half = size / 2
    
    # Grid on XZ plane
    for i in range(divisions + 1):
        # Lines parallel to X axis
        z = -half + i * step
        vertices.extend([-half, 0, z, half, 0, z])
        # Lines parallel to Z axis
        x = -half + i * step
        vertices.extend([x, 0, -half, x, 0, half])
    
    vertices_array = np.array(vertices, dtype=np.float32)
    
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices_array.nbytes, vertices_array, GL_STATIC_DRAW)
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)
    
    return vao, len(vertices) // 3


def draw_trail(shader, obj):
    if len(obj.trail) < 2:
        return
    
    vertices = []
    for pos in obj.trail:
        vertices.extend([pos.x, pos.y, pos.z])
    
    vertices_array = np.array(vertices, dtype=np.float32)
    
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices_array.nbytes, vertices_array, GL_DYNAMIC_DRAW)
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    
    model = np.identity(4, dtype=np.float32)
    model_loc = glGetUniformLocation(shader, "model")
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    
    trail_color = Vec4(obj.color.r * 0.5, obj.color.g * 0.5, obj.color.b * 0.5, 0.6)
    color_loc = glGetUniformLocation(shader, "objectColor")
    glUniform4f(color_loc, trail_color.r, trail_color.g, trail_color.b, trail_color.a)
    
    glUniform1i(glGetUniformLocation(shader, "isTrail"), 1)
    glUniform1i(glGetUniformLocation(shader, "glow"), 0)
    
    glLineWidth(2.0)
    glDrawArrays(GL_LINE_STRIP, 0, len(obj.trail))
    
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])


def main():
    global camera_pos, camera_front, camera_up, delta_time, last_frame, objs, pause, running
    global yaw, pitch, last_x, last_y, time_scale, show_trails, grid_vao
    
    if not glfw.init():
        print("Failed to initialize GLFW")
        return
    
    window = glfw.create_window(1200, 800, "3D Gravity Simulation - Enhanced", None, None)
    if not window:
        print("Failed to create window")
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.02, 0.02, 0.05, 1.0)
    glLineWidth(1.5)
    
    shader = compileProgram(
        compileShader(vertex_shader_source, GL_VERTEX_SHADER),
        compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    )
    
    glUseProgram(shader)
    
    # Setup projection
    projection = perspective(math.radians(45.0), 1200.0/800.0, 1.0, 100000.0)
    proj_loc = glGetUniformLocation(shader, "projection")
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    
    # Create grid
    grid_vao, grid_vertex_count = create_grid(25000, 25)
    
    # Create solar system-like setup
    objs = [
        # Central star (massive, glowing)
        Object(Vec3(0, 0, 0), Vec3(0, 0, 0), 10000, 1500, 
               Vec4(1.0, 0.9, 0.3, 1.0), glow=True, name="Star"),
        
        # Inner planet (fast orbit)
        Object(Vec3(3500, 0, 0), Vec3(0, 0, 32), 10, 300,
               Vec4(0.9, 0.3, 0.2, 1.0), name="Mercury"),
        
        # Earth-like planet
        Object(Vec3(-6000, 0, 0), Vec3(0, 0, -25), 50, 500,
               Vec4(0.2, 0.5, 0.9, 1.0), name="Earth"),
        
        # Large outer planet with moon
        Object(Vec3(0, 0, 10000), Vec3(18, 0, 0), 200, 800,
               Vec4(0.8, 0.6, 0.3, 1.0), name="Jupiter"),
        
        # Moon of outer planet
        Object(Vec3(0, 0, 11500), Vec3(32, 0, 0), 5, 200,
               Vec4(0.7, 0.7, 0.7, 1.0), name="Moon"),
        
        # Distant planet
        Object(Vec3(15000, 0, 0), Vec3(0, 0, 15), 100, 600,
               Vec4(0.5, 0.8, 0.9, 1.0), name="Neptune"),
    ]
    
    print(f"\n✓ Created {len(objs)} celestial bodies")
    print(f"✓ Camera at {camera_pos}")
    print("\n╔══════════════════ CONTROLS ══════════════════╗")
    print("║ W/S/A/D/Space/Shift - Move camera           ║")
    print("║ Mouse              - Look around             ║")
    print("║ K                  - Pause/Resume            ║")
    print("║ T                  - Toggle trails           ║")
    print("║ +/-                - Speed up/slow down time ║")
    print("║ R                  - Reset camera            ║")
    print("║ Q/ESC              - Quit                    ║")
    print("╚══════════════════════════════════════════════╝\n")
    
    # Keyboard callback
    def key_callback(window, key, scancode, action, mods):
        global camera_pos, pause, running, time_scale, show_trails, camera_front, yaw, pitch
        
        if action == glfw.PRESS:
            if key == glfw.KEY_K:
                pause = not pause
                print(f"{'⏸ PAUSED' if pause else '▶ RUNNING'}")
            elif key == glfw.KEY_T:
                show_trails = not show_trails
                print(f"Trails: {'ON' if show_trails else 'OFF'}")
            elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
                time_scale *= 1.5
                print(f"Time scale: {time_scale:.2f}x")
            elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
                time_scale /= 1.5
                print(f"Time scale: {time_scale:.2f}x")
            elif key == glfw.KEY_R:
                camera_pos = Vec3(0.0, 5000.0, 20000.0)
                yaw, pitch = -90.0, -15.0
                camera_front = Vec3(0.0, 0.0, -1.0)
                print("Camera reset")
            elif key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
                running = False
        
        # Continuous movement
        speed = 500.0 * delta_time
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            camera_pos = camera_pos + camera_front * speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            camera_pos = camera_pos - camera_front * speed
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            camera_pos = camera_pos - camera_front.cross(camera_up).normalize() * speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            camera_pos = camera_pos + camera_front.cross(camera_up).normalize() * speed
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            camera_pos = camera_pos + camera_up * speed
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            camera_pos = camera_pos - camera_up * speed
    
    # Mouse callback
    def mouse_callback(window, xpos, ypos):
        global last_x, last_y, yaw, pitch, camera_front
        
        xoffset = (xpos - last_x) * 0.1
        yoffset = (last_y - ypos) * 0.1
        last_x, last_y = xpos, ypos
        
        yaw += xoffset
        pitch = max(min(pitch + yoffset, 89.0), -89.0)
        
        camera_front = Vec3(
            math.cos(math.radians(yaw)) * math.cos(math.radians(pitch)),
            math.sin(math.radians(pitch)),
            math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
        ).normalize()
    
    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    
    view_loc = glGetUniformLocation(shader, "view")
    model_loc = glGetUniformLocation(shader, "model")
    color_loc = glGetUniformLocation(shader, "objectColor")
    
    frame_count = 0
    while not glfw.window_should_close(window) and running:
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Update camera
        view = look_at(camera_pos, camera_pos + camera_front, camera_up)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        
        # Physics update
        if not pause:
            dt = delta_time * time_scale
            
            # Calculate forces between all objects
            for i, obj1 in enumerate(objs):
                total_force = Vec3(0, 0, 0)
                
                for j, obj2 in enumerate(objs):
                    if i != j:
                        # Vector from obj1 to obj2
                        r_vec = obj2.position - obj1.position
                        distance = r_vec.length()
                        
                        if distance > (obj1.radius + obj2.radius) * 0.1:  # Avoid singularity
                            # Gravitational force: F = G * m1 * m2 / r^2
                            force_magnitude = G * obj1.mass * obj2.mass / (distance * distance)
                            force_direction = r_vec.normalize()
                            force = force_direction * force_magnitude
                            total_force = total_force + force
                
                # Apply total force
                obj1.apply_force(total_force * dt)
            
            # Update positions
            for obj in objs:
                obj.update(dt)
        
        # Draw grid
        glUniform1i(glGetUniformLocation(shader, "isGrid"), 1)
        glUniform1i(glGetUniformLocation(shader, "isTrail"), 0)
        glUniform1i(glGetUniformLocation(shader, "glow"), 0)
        glUniform4f(color_loc, 0.15, 0.15, 0.2, 0.3)
        
        model = np.identity(4, dtype=np.float32)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        
        glBindVertexArray(grid_vao)
        glDrawArrays(GL_LINES, 0, grid_vertex_count)
        
        # Draw trails
        if show_trails:
            for obj in objs:
                if not obj.glow:  # Don't draw trail for star
                    draw_trail(shader, obj)
        
        # Draw objects
        glUniform1i(glGetUniformLocation(shader, "isGrid"), 0)
        glUniform1i(glGetUniformLocation(shader, "isTrail"), 0)
        
        for obj in objs:
            model = translate(np.identity(4, dtype=np.float32), obj.position)
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
            glUniform4f(color_loc, obj.color.r, obj.color.g, obj.color.b, obj.color.a)
            glUniform1i(glGetUniformLocation(shader, "glow"), 1 if obj.glow else 0)
            
            glBindVertexArray(obj.vao)
            glDrawArrays(GL_TRIANGLES, 0, obj.vertex_count // 3)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        frame_count += 1
        if frame_count == 120:
            print(f"Running at {1.0/delta_time:.0f} FPS")
    
    for obj in objs:
        obj.cleanup()
    
    glDeleteVertexArrays(1, [grid_vao])
    glfw.terminate()


if __name__ == "__main__":
    main()