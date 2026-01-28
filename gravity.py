import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import ctypes

vertex_shader_source = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 FragPos;
out vec3 Normal;
out float depth;
void main() {
    FragPos = aPos;
    Normal = normalize(aPos);
    depth = aPos.y;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in float depth;
out vec4 FragColor;
uniform vec4 objectColor;
uniform bool isGrid;
uniform bool glow;
void main() {
    if (isGrid) {
        float depthColor = clamp(-depth / 1500.0, 0.0, 1.0);
        vec3 baseColor = mix(vec3(0.1, 0.15, 0.3), vec3(0.2, 0.4, 0.8), depthColor);
        FragColor = vec4(baseColor, 0.85);
    } else if (glow) {
        FragColor = vec4(objectColor.rgb * 2.5, objectColor.a);
    } else {
        vec3 lightDir = normalize(vec3(0.0, 0.0, 0.0) - FragPos);
        float diff = max(dot(Normal, lightDir), 0.0);
        vec3 ambient = objectColor.rgb * 0.15;
        vec3 diffuse = objectColor.rgb * diff * 0.85;
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

# Globals
running = True
pause = False
camera_pos = Vec3(0.0, 10000.0, 16000.0)
camera_front = Vec3(0.0, -0.35, -1.0).normalize()
camera_up = Vec3(0.0, 1.0, 0.0)
last_x, last_y = 700.0, 450.0
yaw, pitch = -90.0, -22.0
delta_time, last_frame = 0.0, 0.0
time_scale = 1.0

G = 50.0
epsilon = 500.0
objs = []
grid_heights = {}

class Object:
    def __init__(self, position, velocity, mass, radius, color, glow=False, name="", fixed=False):
        self.position = position
        self.velocity = velocity
        self.acc = Vec3(0, 0, 0)
        self.mass = mass
        self.radius = radius
        self.color = color
        self.glow = glow
        self.name = name
        self.fixed = fixed
        
        self.vao = None
        self.vbo = None
        self.vertex_count = 0
        
        vertices = self.draw_sphere()
        self.vertex_count = len(vertices)
        self.create_vbo_vao(vertices)
    
    def draw_sphere(self):
        vertices = []
        stacks, sectors = 16, 16
        
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
    
    def cleanup(self):
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])


def calculate_warp(x, z, objects):
    key = (int(x/100), int(z/100))
    if key in grid_heights:
        return grid_heights[key]
    
    warp = 0.0
    for obj in objects:
        dx = obj.position.x - x
        dz = obj.position.z - z
        r2 = dx*dx + dz*dz + epsilon*epsilon
        r = math.sqrt(r2)
        warp_strength = (obj.mass * 20.0) / r
        warp += warp_strength
    
    result = -warp
    grid_heights[key] = result
    return result


def create_warped_grid(size, divisions, objects):
    global grid_heights
    grid_heights = {}
    
    vertices = []
    step = size / divisions
    half = size / 2
    
    grid_points = []
    for i in range(divisions + 1):
        row = []
        for j in range(divisions + 1):
            x = -half + j * step
            z = -half + i * step
            y = calculate_warp(x, z, objects)
            row.append([x, y, z])
        grid_points.append(row)
    
    for i in range(divisions + 1):
        for j in range(divisions):
            vertices.extend(grid_points[i][j])
            vertices.extend(grid_points[i][j + 1])
    
    for i in range(divisions):
        for j in range(divisions + 1):
            vertices.extend(grid_points[i][j])
            vertices.extend(grid_points[i + 1][j])
    
    return np.array(vertices, dtype=np.float32)


def main():
    global camera_pos, camera_front, camera_up, delta_time, last_frame, objs, pause, running
    global yaw, pitch, last_x, last_y, time_scale
    
    if not glfw.init():
        return
    
    window = glfw.create_window(1400, 900, "DEBUG: Testing Movement", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.0, 0.0, 0.03, 1.0)
    
    shader = compileProgram(
        compileShader(vertex_shader_source, GL_VERTEX_SHADER),
        compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    )
    
    glUseProgram(shader)
    
    projection = perspective(math.radians(45.0), 1400.0/900.0, 1.0, 150000.0)
    proj_loc = glGetUniformLocation(shader, "projection")
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    
    print("\n" + "="*75)
    print("DEBUG MODE - Testing Physics".center(75))
    print("="*75)
    
    # Create objects
    star = Object(Vec3(0, 0, 0), Vec3(0, 0, 0), 3000, 1200, 
                  Vec4(1.0, 0.95, 0.35, 1.0), glow=True, name="Star", fixed=True)
    objs.append(star)
    
    # Single test planet with known velocity
    r = 5000
    v = math.sqrt(G * star.mass / r)
    print(f"\nTest planet:")
    print(f"  Distance: {r}")
    print(f"  Required velocity: {v:.2f}")
    
    planet = Object(Vec3(r, 0, 0), Vec3(0, 0, v), 25, 320,
                    Vec4(0.4, 0.6, 1.0, 1.0), name="Test Planet", fixed=False)
    objs.append(planet)
    
    print(f"  Initial pos: ({planet.position.x:.1f}, {planet.position.y:.1f}, {planet.position.z:.1f})")
    print(f"  Initial vel: ({planet.velocity.x:.2f}, {planet.velocity.y:.2f}, {planet.velocity.z:.2f})")
    print(f"\nStarting simulation...")
    print("Watch the terminal - it will print planet position every 30 frames")
    print("="*75 + "\n")
    
    # Grid
    grid_size = 22000
    grid_div = 45
    grid_vao = glGenVertexArrays(1)
    grid_vbo = glGenBuffers(1)
    
    def update_grid():
        vertices = create_warped_grid(grid_size, grid_div, objs)
        glBindVertexArray(grid_vao)
        glBindBuffer(GL_ARRAY_BUFFER, grid_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)
        return len(vertices) // 3
    
    grid_vertex_count = update_grid()
    
    def key_callback(window, key, scancode, action, mods):
        global camera_pos, pause, running, time_scale, camera_front, yaw, pitch
        
        if action == glfw.PRESS:
            if key == glfw.KEY_K:
                pause = not pause
                print(f"\n{'PAUSED' if pause else 'RUNNING'}\n")
            elif key == glfw.KEY_Q:
                running = False
        
        speed = 700.0
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            camera_pos = camera_pos + camera_front * speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            camera_pos = camera_pos - camera_front * speed
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            camera_pos = camera_pos - camera_front.cross(camera_up).normalize() * speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            camera_pos = camera_pos + camera_front.cross(camera_up).normalize() * speed
    
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
    
    frame = 0
    
    while not glfw.window_should_close(window) and running:
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        view = look_at(camera_pos, camera_pos + camera_front, camera_up)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        
        # PHYSICS
        if not pause:
            dt = min(delta_time, 1.0/120.0) * time_scale
            
            # Print debug info first frame
            if frame == 0:
                print(f"Frame 0: dt = {dt:.6f}")
            
            # Step 1: Compute accelerations
            for i, obj1 in enumerate(objs):
                if obj1.fixed:
                    continue
                
                acc = Vec3(0, 0, 0)
                for j, obj2 in enumerate(objs):
                    if i == j:
                        continue
                    
                    dx = obj2.position.x - obj1.position.x
                    dy = obj2.position.y - obj1.position.y
                    dz = obj2.position.z - obj1.position.z
                    r2 = dx*dx + dy*dy + dz*dz + epsilon*epsilon
                    r = math.sqrt(r2)
                    a = G * obj2.mass / r2
                    
                    acc.x += a * dx / r
                    acc.y += a * dy / r
                    acc.z += a * dz / r
                
                obj1.acc = acc
                
                # Debug first frame
                if frame == 0:
                    print(f"  {obj1.name} acceleration: ({acc.x:.4f}, {acc.y:.4f}, {acc.z:.4f})")
            
            # Step 2: Half-step velocity
            for obj in objs:
                if not obj.fixed:
                    obj.velocity.x += obj.acc.x * dt * 0.5
                    obj.velocity.y += obj.acc.y * dt * 0.5
                    obj.velocity.z += obj.acc.z * dt * 0.5
            
            # Step 3: Full-step position
            for obj in objs:
                if not obj.fixed:
                    obj.position.x += obj.velocity.x * dt
                    obj.position.y += obj.velocity.y * dt
                    obj.position.z += obj.velocity.z * dt
            
            # Step 4: Second half-step velocity
            for obj in objs:
                if not obj.fixed:
                    obj.velocity.x += obj.acc.x * dt * 0.5
                    obj.velocity.y += obj.acc.y * dt * 0.5
                    obj.velocity.z += obj.acc.z * dt * 0.5
            
            # Debug output every 30 frames
            if frame % 30 == 0 and frame > 0:
                for obj in objs:
                    if not obj.fixed:
                        print(f"Frame {frame}: {obj.name} pos=({obj.position.x:.1f}, {obj.position.y:.1f}, {obj.position.z:.1f}), "
                              f"vel=({obj.velocity.x:.2f}, {obj.velocity.y:.2f}, {obj.velocity.z:.2f})")
            
            # Update grid
            if frame % 2 == 0:
                grid_vertex_count = update_grid()
        
        # Draw grid
        glUniform1i(glGetUniformLocation(shader, "isGrid"), 1)
        glUniform1i(glGetUniformLocation(shader, "glow"), 0)
        glUniform4f(color_loc, 0.3, 0.4, 0.7, 0.8)
        
        model = np.identity(4, dtype=np.float32)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        
        glLineWidth(1.8)
        glBindVertexArray(grid_vao)
        glDrawArrays(GL_LINES, 0, grid_vertex_count)
        
        # Draw objects
        glUniform1i(glGetUniformLocation(shader, "isGrid"), 0)
        
        for obj in objs:
            model = translate(np.identity(4, dtype=np.float32), obj.position)
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
            glUniform4f(color_loc, obj.color.r, obj.color.g, obj.color.b, obj.color.a)
            glUniform1i(glGetUniformLocation(shader, "glow"), 1 if obj.glow else 0)
            
            glBindVertexArray(obj.vao)
            glDrawArrays(GL_TRIANGLES, 0, obj.vertex_count // 3)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        frame += 1
    
    for obj in objs:
        obj.cleanup()
    
    glDeleteVertexArrays(1, [grid_vao])
    glDeleteBuffers(1, [grid_vbo])
    glfw.terminate()


if __name__ == "__main__":
    main()