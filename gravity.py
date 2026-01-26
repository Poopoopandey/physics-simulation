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
        // Enhanced lighting for grid based on depth
        float depthColor = clamp(-depth / 1500.0, 0.0, 1.0);
        vec3 baseColor = mix(vec3(0.1, 0.15, 0.3), vec3(0.2, 0.4, 0.8), depthColor);
        
        // Add subtle lighting
        vec3 lightDir = normalize(vec3(0.3, 1.0, 0.2));
        vec3 normal = normalize(vec3(0, 1, 0)); // Grid normal points up
        float diffuse = max(dot(normal, lightDir), 0.0) * 0.3 + 0.7;
        
        FragColor = vec4(baseColor * diffuse, 0.85);
    } else if (glow) {
        // Bright glow for star
        FragColor = vec4(objectColor.rgb * 2.5, objectColor.a);
    } else {
        // Realistic lighting for planets
        vec3 lightDir = normalize(vec3(0.0, 0.0, 0.0) - FragPos); // Light from center
        float diff = max(dot(Normal, lightDir), 0.0);
        
        vec3 ambient = objectColor.rgb * 0.15;
        vec3 diffuse = objectColor.rgb * diff * 0.85;
        
        // Add rim lighting
        vec3 viewDir = normalize(-FragPos);
        float rim = 1.0 - max(dot(viewDir, Normal), 0.0);
        vec3 rimLight = objectColor.rgb * rim * 0.2;
        
        FragColor = vec4(ambient + diffuse + rimLight, objectColor.a);
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

# Physics constants - more balanced
G = 50.0
epsilon = 500.0  # Gravitational softening
objs = []

# Store grid heights for smoother rendering
grid_heights = {}

class Object:
    def __init__(self, position, velocity, mass, radius, color, glow=False, name=""):
        self.position = position
        self.velocity = velocity
        self.acc = Vec3(0, 0, 0)  # ACCELERATION STORAGE
        self.mass = mass
        self.radius = radius
        self.color = color
        self.glow = glow
        self.name = name
        
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


def calculate_circular_velocity(distance, central_mass):
    """Calculate velocity for stable circular orbit"""
    return math.sqrt(G * central_mass / distance)


def calculate_warp(x, z, objects):
    """Calculate spacetime curvature with caching"""
    key = (int(x/100), int(z/100))  # Cache by grid cell
    
    if key in grid_heights:
        return grid_heights[key]
    
    warp = 0.0
    for obj in objects:
        dx = obj.position.x - x
        dz = obj.position.z - z
        r2 = dx*dx + dz*dz + epsilon*epsilon  # With softening
        r = math.sqrt(r2)
        
        # Gravitational potential visualization
        warp_strength = (obj.mass * 20.0) / r
        warp += warp_strength
    
    result = -warp
    grid_heights[key] = result
    return result


def create_warped_grid(size, divisions, objects):
    global grid_heights
    grid_heights = {}  # Clear cache
    
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
    
    # Horizontal lines
    for i in range(divisions + 1):
        for j in range(divisions):
            vertices.extend(grid_points[i][j])
            vertices.extend(grid_points[i][j + 1])
    
    # Vertical lines
    for i in range(divisions):
        for j in range(divisions + 1):
            vertices.extend(grid_points[i][j])
            vertices.extend(grid_points[i + 1][j])
    
    return np.array(vertices, dtype=np.float32)


def calculate_total_energy(objects):
    """Calculate total energy (kinetic + potential) for verification"""
    E = 0.0
    
    for i, a in enumerate(objects):
        if a.glow:
            continue
        
        # Kinetic energy: 0.5 * m * v²
        v2 = a.velocity.length()**2
        E += 0.5 * a.mass * v2
        
        # Potential energy: -G * m1 * m2 / r (count each pair once)
        for j, b in enumerate(objects):
            if i < j:
                r = (a.position - b.position).length()
                if r > 0:
                    E -= G * a.mass * b.mass / r
    
    return E


def main():
    global camera_pos, camera_front, camera_up, delta_time, last_frame, objs, pause, running
    global yaw, pitch, last_x, last_y, time_scale
    
    if not glfw.init():
        return
    
    window = glfw.create_window(1400, 900, "🌌 Stable N-Body Simulation with Spacetime Curvature", None, None)
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
    print("🌌 STABLE N-BODY GRAVITY SIMULATION 🌌".center(75))
    print("="*75)
    print("\n⚙️  Physics Configuration:")
    print(f"   G = {G}")
    print(f"   Softening ε = {epsilon}")
    print(f"   Integration: Leapfrog (symplectic)")
    print(f"   Timestep: min(dt, 1/120) × time_scale")
    
    # Central star (massive, stationary)
    star = Object(Vec3(0, 0, 0), Vec3(0, 0, 0), 3000, 1200, 
                  Vec4(1.0, 0.95, 0.35, 1.0), glow=True, name="Star")
    objs.append(star)
    
    # Planets with proper circular orbits
    r1 = 5000
    v1 = calculate_circular_velocity(r1, star.mass)
    planet1 = Object(Vec3(r1, 0, 0), Vec3(0, 0, v1), 25, 320,
                     Vec4(0.4, 0.6, 1.0, 1.0), name="Mercury")
    objs.append(planet1)
    
    r2 = 7500
    v2 = calculate_circular_velocity(r2, star.mass)
    planet2 = Object(Vec3(-r2, 0, 0), Vec3(0, 0, -v2), 40, 400,
                     Vec4(1.0, 0.5, 0.2, 1.0), name="Mars")
    objs.append(planet2)
    
    r3 = 10500
    v3 = calculate_circular_velocity(r3, star.mass)
    planet3 = Object(Vec3(0, 0, r3), Vec3(v3, 0, 0), 80, 550,
                     Vec4(0.9, 0.7, 0.35, 1.0), name="Jupiter")
    objs.append(planet3)
    
    print(f"\n🌟 Created {len(objs)} bodies:")
    for obj in objs:
        if obj.glow:
            print(f"   {obj.name}: mass={obj.mass}, radius={obj.radius}")
        else:
            dist = obj.position.length()
            vel = obj.velocity.length()
            print(f"   {obj.name}: r={dist:.0f}, v={vel:.2f}")
    
    # Calculate initial energy
    E0 = calculate_total_energy(objs)
    print(f"\n📊 Initial Total Energy: {E0:.2e}")
    
    print("\n" + "─"*75)
    print("⌨️  CONTROLS:")
    print("  W/A/S/D/Space/Shift → Fly camera")
    print("  Mouse              → Look around")  
    print("  K                  → Pause/Resume")
    print("  +/-                → Speed up/slow down")
    print("  E                  → Show energy (check stability)")
    print("  R                  → Reset camera")
    print("  Q                  → Quit")
    print("="*75 + "\n")
    
    # Grid setup
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
                print(f"\n{'⏸️  PAUSED' if pause else '▶️  RUNNING'}\n")
            elif key == glfw.KEY_E:
                E = calculate_total_energy(objs)
                dE = abs((E - E0) / E0) * 100 if E0 != 0 else 0
                print(f"\n📊 Energy: {E:.2e} (ΔE = {dE:.3f}%)\n")
            elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
                time_scale = min(5.0, time_scale * 1.3)
                print(f"⏩ Time: {time_scale:.2f}x")
            elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
                time_scale = max(0.1, time_scale / 1.3)
                print(f"⏪ Time: {time_scale:.2f}x")
            elif key == glfw.KEY_R:
                camera_pos = Vec3(0.0, 10000.0, 16000.0)
                yaw, pitch = -90.0, -22.0
                camera_front = Vec3(0.0, -0.35, -1.0).normalize()
                print("📷 Camera reset")
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
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            camera_pos = camera_pos + camera_up * speed
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            camera_pos = camera_pos - camera_up * speed
    
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
        
        # PHYSICS - Leapfrog integration with softening
        if not pause:
            dt = min(delta_time, 1.0/120.0) * time_scale
            
            # FIRST: Compute accelerations
            for i, obj1 in enumerate(objs):
                if obj1.glow:
                    continue
                
                acc = Vec3(0, 0, 0)
                for j, obj2 in enumerate(objs):
                    if i == j:
                        continue
                    
                    dx = obj2.position.x - obj1.position.x
                    dy = obj2.position.y - obj1.position.y
                    dz = obj2.position.z - obj1.position.z
                    r2 = dx*dx + dy*dy + dz*dz + epsilon*epsilon  # Softening
                    r = math.sqrt(r2)
                    a = G * obj2.mass / r2
                    
                    acc.x += a * dx / r
                    acc.y += a * dy / r
                    acc.z += a * dz / r
                
                obj1.acc = acc
            
            # SECOND: Update velocities (half step)
            for obj in objs:
                if not obj.glow:
                    obj.velocity.x += obj.acc.x * dt * 0.5
                    obj.velocity.y += obj.acc.y * dt * 0.5
                    obj.velocity.z += obj.acc.z * dt * 0.5
            
            # THIRD: Update positions
            for obj in objs:
                if not obj.glow:
                    obj.position.x += obj.velocity.x * dt
                    obj.position.y += obj.velocity.y * dt
                    obj.position.z += obj.velocity.z * dt
            
            # FOURTH: Update velocities again (second half)
            for obj in objs:
                if not obj.glow:
                    obj.velocity.x += obj.acc.x * dt * 0.5
                    obj.velocity.y += obj.acc.y * dt * 0.5
                    obj.velocity.z += obj.acc.z * dt * 0.5
            
            # Update grid every 2 frames
            if frame % 2 == 0:
                grid_vertex_count = update_grid()
        
        # DRAW WARPED GRID
        glUniform1i(glGetUniformLocation(shader, "isGrid"), 1)
        glUniform1i(glGetUniformLocation(shader, "glow"), 0)
        glUniform4f(color_loc, 0.3, 0.4, 0.7, 0.8)
        
        model = np.identity(4, dtype=np.float32)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        
        glLineWidth(1.8)
        glBindVertexArray(grid_vao)
        glDrawArrays(GL_LINES, 0, grid_vertex_count)
        
        # DRAW OBJECTS
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
        if frame == 120:
            E = calculate_total_energy(objs)
            dE = abs((E - E0) / E0) * 100 if E0 != 0 else 0
            print(f"⚡ {1.0/delta_time:.0f} FPS | Energy drift: {dE:.3f}%\n")
    
    for obj in objs:
        obj.cleanup()
    
    glDeleteVertexArrays(1, [grid_vao])
    glDeleteBuffers(1, [grid_vbo])
    glfw.terminate()


if __name__ == "__main__":
    main()