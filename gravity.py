import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import ctypes

# Shader sources
vertex_shader_source = """
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

fragment_shader_source = """
#version 330 core
in float lightIntensity;
out vec4 FragColor;
uniform vec4 objectColor;
uniform bool isGrid;
uniform bool GLOW;
void main() {
    if (isGrid) {
        FragColor = objectColor;
    } else if(GLOW){
        FragColor = vec4(objectColor.rgb * 100000, objectColor.a);
    } else {
        float fade = smoothstep(0.0, 10.0, lightIntensity*10);
        FragColor = vec4(objectColor.rgb * fade, objectColor.a);
    }
}
"""

# Simple vector/matrix classes
class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        if isinstance(x, (list, tuple)):
            self.x, self.y, self.z = float(x[0]), float(x[1]), float(x[2])
        else:
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
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def __repr__(self):
        return f"Vec3({self.x}, {self.y}, {self.z})"

class Vec4:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.r = self.x = float(x)
        self.g = self.y = float(y)
        self.b = self.z = float(z)
        self.a = self.w = float(w)

def perspective(fovy, aspect, near, far):
    """Create perspective projection matrix"""
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
    """Create view matrix"""
    f = (center - eye).normalize()
    s = f.cross(up).normalize()
    u = s.cross(f)
    
    mat = np.identity(4, dtype=np.float32)
    mat[0, 0] = s.x
    mat[1, 0] = s.y
    mat[2, 0] = s.z
    mat[0, 1] = u.x
    mat[1, 1] = u.y
    mat[2, 1] = u.z
    mat[0, 2] = -f.x
    mat[1, 2] = -f.y
    mat[2, 2] = -f.z
    mat[3, 0] = -s.dot(eye)
    mat[3, 1] = -u.dot(eye)
    mat[3, 2] = f.dot(eye)
    return mat

def translate(mat, vec):
    """Create translation matrix"""
    result = mat.copy()
    result[3, 0] = vec.x
    result[3, 1] = vec.y
    result[3, 2] = vec.z
    return result

def radians(degrees):
    """Convert degrees to radians"""
    return degrees * math.pi / 180.0

# Global variables
running = True
pause = False
camera_pos = Vec3(0.0, 1000.0, 5000.0)
camera_front = Vec3(0.0, 0.0, -1.0)
camera_up = Vec3(0.0, 1.0, 0.0)
last_x, last_y = 400.0, 300.0
yaw, pitch = -90.0, 0.0
delta_time, last_frame = 0.0, 0.0

G = 6.6743e-11
c = 299792458.0
init_mass = 10**22
size_ratio = 30000.0

objs = []
grid_vao, grid_vbo = None, None


class Object:
    def __init__(self, init_position, init_velocity, mass, density=3344, 
                 color=Vec4(1.0, 0.0, 0.0, 1.0), glow=False):
        self.position = Vec3(*init_position) if isinstance(init_position, (list, tuple)) else init_position
        self.velocity = Vec3(*init_velocity) if isinstance(init_velocity, (list, tuple)) else init_velocity
        self.mass = mass
        self.density = density
        self.radius = ((3 * self.mass / self.density) / (4 * math.pi)) ** (1.0/3.0) / size_ratio
        self.color = color
        self.glow = glow
        
        self.initializing = False
        self.launched = False
        self.target = False
        self.last_pos = self.position
        
        self.vao = None
        self.vbo = None
        self.vertex_count = 0
        
        vertices = self.draw()
        self.vertex_count = len(vertices)
        self.create_vbo_vao(vertices)
    
    def draw(self):
        vertices = []
        stacks = 10
        sectors = 10
        
        for i in range(stacks + 1):
            theta1 = (i / stacks) * math.pi
            theta2 = ((i + 1) / stacks) * math.pi
            
            for j in range(sectors):
                phi1 = (j / sectors) * 2 * math.pi
                phi2 = ((j + 1) / sectors) * 2 * math.pi
                
                v1 = self.spherical_to_cartesian(self.radius, theta1, phi1)
                v2 = self.spherical_to_cartesian(self.radius, theta1, phi2)
                v3 = self.spherical_to_cartesian(self.radius, theta2, phi1)
                v4 = self.spherical_to_cartesian(self.radius, theta2, phi2)
                
                vertices.extend([v1.x, v1.y, v1.z])
                vertices.extend([v2.x, v2.y, v2.z])
                vertices.extend([v3.x, v3.y, v3.z])
                
                vertices.extend([v2.x, v2.y, v2.z])
                vertices.extend([v4.x, v4.y, v4.z])
                vertices.extend([v3.x, v3.y, v3.z])
        
        return vertices
    
    def spherical_to_cartesian(self, r, theta, phi):
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.cos(theta)
        z = r * math.sin(theta) * math.sin(phi)
        return Vec3(x, y, z)
    
    def create_vbo_vao(self, vertices):
        vertices_array = np.array(vertices, dtype=np.float32)
        
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices_array.nbytes, vertices_array, GL_STATIC_DRAW)
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)
    
    def update_pos(self):
        self.position.x += self.velocity.x / 94
        self.position.y += self.velocity.y / 94
        self.position.z += self.velocity.z / 94
        self.radius = ((3 * self.mass / self.density) / (4 * math.pi)) ** (1.0/3.0) / size_ratio
    
    def update_vertices(self):
        vertices = self.draw()
        vertices_array = np.array(vertices, dtype=np.float32)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices_array.nbytes, vertices_array, GL_STATIC_DRAW)
    
    def get_pos(self):
        return self.position
    
    def accelerate(self, x, y, z):
        self.velocity.x += x / 96
        self.velocity.y += y / 96
        self.velocity.z += z / 96
    
    def check_collision(self, other):
        dx = other.position.x - self.position.x
        dy = other.position.y - self.position.y
        dz = other.position.z - self.position.z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if other.radius + self.radius > distance:
            return -0.2
        return 1.0
    
    def cleanup(self):
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])


def start_glu():
    if not glfw.init():
        print("Failed to initialize GLFW")
        return None
    
    window = glfw.create_window(800, 600, "3D Gravity Simulation", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return None
    
    glfw.make_context_current(window)
    
    glEnable(GL_DEPTH_TEST)
    glViewport(0, 0, 800, 600)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.05, 0.05, 0.08, 1.0)
    
    return window


def create_shader_program(vertex_source, fragment_source):
    vertex_shader = compileShader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_source, GL_FRAGMENT_SHADER)
    shader_program = compileProgram(vertex_shader, fragment_shader)
    return shader_program


def create_vbo_vao(vertices):
    vertices_array = np.array(vertices, dtype=np.float32)
    
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices_array.nbytes, vertices_array, GL_STATIC_DRAW)
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)
    
    return vao, vbo


def update_cam(shader_program, camera_pos):
    glUseProgram(shader_program)
    view = look_at(camera_pos, camera_pos + camera_front, camera_up)
    view_loc = glGetUniformLocation(shader_program, "view")
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)


def key_callback(window, key, scancode, action, mods):
    global camera_pos, pause, running
    
    camera_speed = 10000.0 * delta_time
    shift_pressed = (mods & glfw.MOD_SHIFT) != 0
    
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camera_pos = camera_pos + camera_front * camera_speed
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camera_pos = camera_pos - camera_front * camera_speed
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        camera_pos = camera_pos - camera_front.cross(camera_up).normalize() * camera_speed
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        camera_pos = camera_pos + camera_front.cross(camera_up).normalize() * camera_speed
    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
        camera_pos = camera_pos + camera_up * camera_speed
    if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
        camera_pos = camera_pos - camera_up * camera_speed
    
    if key == glfw.KEY_K and action == glfw.PRESS:
        pause = not pause
    
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        running = False
        glfw.set_window_should_close(window, True)
    
    if objs and objs[-1].initializing:
        if key == glfw.KEY_UP and (action == glfw.PRESS or action == glfw.REPEAT):
            if not shift_pressed:
                objs[-1].position.y += objs[-1].radius * 0.2
            else:
                objs[-1].position.z += objs[-1].radius * 0.2
        if key == glfw.KEY_DOWN and (action == glfw.PRESS or action == glfw.REPEAT):
            if not shift_pressed:
                objs[-1].position.y -= objs[-1].radius * 0.2
            else:
                objs[-1].position.z -= objs[-1].radius * 0.2
        if key == glfw.KEY_RIGHT and (action == glfw.PRESS or action == glfw.REPEAT):
            objs[-1].position.x += objs[-1].radius * 0.2
        if key == glfw.KEY_LEFT and (action == glfw.PRESS or action == glfw.REPEAT):
            objs[-1].position.x -= objs[-1].radius * 0.2


def mouse_callback(window, xpos, ypos):
    global last_x, last_y, yaw, pitch, camera_front
    
    xoffset = xpos - last_x
    yoffset = last_y - ypos
    last_x = xpos
    last_y = ypos
    
    sensitivity = 0.1
    xoffset *= sensitivity
    yoffset *= sensitivity
    
    yaw += xoffset
    pitch += yoffset
    
    if pitch > 89.0:
        pitch = 89.0
    if pitch < -89.0:
        pitch = -89.0
    
    front_x = math.cos(radians(yaw)) * math.cos(radians(pitch))
    front_y = math.sin(radians(pitch))
    front_z = math.sin(radians(yaw)) * math.cos(radians(pitch))
    camera_front = Vec3(front_x, front_y, front_z).normalize()


def mouse_button_callback(window, button, action, mods):
    global objs
    
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            objs.append(Object(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), init_mass))
            objs[-1].initializing = True
        if action == glfw.RELEASE:
            objs[-1].initializing = False
            objs[-1].launched = True
    
    if objs and button == glfw.MOUSE_BUTTON_RIGHT and objs[-1].initializing:
        if action == glfw.PRESS or action == glfw.REPEAT:
            objs[-1].mass *= 1.2
            print(f"MASS: {objs[-1].mass}")


def scroll_callback(window, xoffset, yoffset):
    global camera_pos
    
    camera_speed = 250000.0 * delta_time
    if yoffset > 0:
        camera_pos = camera_pos + camera_front * camera_speed
    elif yoffset < 0:
        camera_pos = camera_pos - camera_front * camera_speed


def draw_grid(shader_program, grid_vao, vertex_count):
    glUseProgram(shader_program)
    model = np.identity(4, dtype=np.float32)
    model_loc = glGetUniformLocation(shader_program, "model")
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    
    glBindVertexArray(grid_vao)
    glPointSize(5.0)
    glDrawArrays(GL_LINES, 0, vertex_count // 3)
    glBindVertexArray(0)


def create_grid_vertices(size, divisions, objs):
    vertices = []
    step = size / divisions
    half_size = size / 2.0
    
    for y_step in range(3, 4):
        y = -half_size * 0.3 + y_step * step
        for z_step in range(divisions + 1):
            z = -half_size + z_step * step
            for x_step in range(divisions):
                x_start = -half_size + x_step * step
                x_end = x_start + step
                vertices.extend([x_start, y, z, x_end, y, z])
    
    for x_step in range(divisions + 1):
        x = -half_size + x_step * step
        for y_step in range(3, 4):
            y = -half_size * 0.3 + y_step * step
            for z_step in range(divisions):
                z_start = -half_size + z_step * step
                z_end = z_start + step
                vertices.extend([x, y, z_start, x, y, z_end])
    
    return vertices


def main():
    global camera_pos, camera_front, delta_time, last_frame, objs, grid_vao, grid_vbo, pause
    
    window = start_glu()
    if not window:
        return
    
    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
    
    model_loc = glGetUniformLocation(shader_program, "model")
    object_color_loc = glGetUniformLocation(shader_program, "objectColor")
    glUseProgram(shader_program)
    
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    
    projection = perspective(radians(45.0), 800.0 / 600.0, 0.1, 750000.0)
    projection_loc = glGetUniformLocation(shader_program, "projection")
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)
    
    camera_pos = Vec3(0.0, 1000.0, 5000.0)
    camera_front = Vec3(0.0, 0.0, -1.0)
    
    objs = [
        Object(Vec3(-5000, 650, -350), Vec3(0, 0, 1500), 5.97219*10**22, 5515, 
               Vec4(0.0, 1.0, 1.0, 1.0)),
        Object(Vec3(5000, 650, -350), Vec3(0, 0, -1500), 5.97219*10**22, 5515, 
               Vec4(0.0, 1.0, 1.0, 1.0)),
        Object(Vec3(0, 0, -350), Vec3(0, 0, 0), 1.989*10**25, 5515, 
               Vec4(1.0, 0.929, 0.176, 1.0), True),
    ]
    
    print(f"Initialized {len(objs)} objects")
    
    grid_vertices = create_grid_vertices(20000.0, 25, objs)
    grid_vao, grid_vbo = create_vbo_vao(grid_vertices)
    
    frame_count = 0
    while not glfw.window_should_close(window) and running:
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        update_cam(shader_program, camera_pos)
        
        if objs and objs[-1].initializing:
            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
                objs[-1].mass *= 1.0 + 1.0 * delta_time
                objs[-1].radius = ((3 * objs[-1].mass / objs[-1].density) / 
                                  (4 * math.pi)) ** (1.0/3.0) / size_ratio
                objs[-1].update_vertices()
        
        glUseProgram(shader_program)
        glUniform4f(object_color_loc, 1.0, 1.0, 1.0, 0.25)
        glUniform1i(glGetUniformLocation(shader_program, "isGrid"), 1)
        glUniform1i(glGetUniformLocation(shader_program, "GLOW"), 0)
        
        grid_array = np.array(grid_vertices, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, grid_vbo)
        glBufferData(GL_ARRAY_BUFFER, grid_array.nbytes, grid_array, GL_DYNAMIC_DRAW)
        draw_grid(shader_program, grid_vao, len(grid_vertices))
        
        for obj in objs:
            glUniform4f(object_color_loc, obj.color.r, obj.color.g, obj.color.b, obj.color.a)
            
            for obj2 in objs:
                if obj2 is not obj and not obj.initializing and not obj2.initializing:
                    dx = obj2.get_pos().x - obj.get_pos().x
                    dy = obj2.get_pos().y - obj.get_pos().y
                    dz = obj2.get_pos().z - obj.get_pos().z
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    if distance > 0:
                        direction = [dx / distance, dy / distance, dz / distance]
                        distance *= 1000
                        g_force = (G * obj.mass * obj2.mass) / (distance * distance)
                        
                        acc1 = g_force / obj.mass
                        acc = [direction[0] * acc1, direction[1] * acc1, direction[2] * acc1]
                        
                        if not pause:
                            obj.accelerate(acc[0], acc[1], acc[2])
                        
                        obj.velocity = obj.velocity * obj.check_collision(obj2)
            
            if obj.initializing:
                obj.radius = ((3 * obj.mass / obj.density) / (4 * math.pi)) ** (1.0/3.0) / 1000000
                obj.update_vertices()
            
            if not pause:
                obj.update_pos()
            
            model = np.identity(4, dtype=np.float32)
            model = translate(model, obj.position)
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
            glUniform1i(glGetUniformLocation(shader_program, "isGrid"), 0)
            
            if obj.glow:
                glUniform1i(glGetUniformLocation(shader_program, "GLOW"), 1)
            else:
                glUniform1i(glGetUniformLocation(shader_program, "GLOW"), 0)
            
            glBindVertexArray(obj.vao)
            glDrawArrays(GL_TRIANGLES, 0, obj.vertex_count // 3)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        frame_count += 1
        if frame_count == 60:
            print(f"Simulation running, FPS: {1.0/delta_time:.1f}")
    
    for obj in objs:
        obj.cleanup()
    
    glDeleteVertexArrays(1, [grid_vao])
    glDeleteBuffers(1, [grid_vbo])
    glDeleteProgram(shader_program)
    glfw.terminate()


if __name__ == "__main__":
    main()

