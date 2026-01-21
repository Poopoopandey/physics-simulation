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
void main() {
    if (isGrid) {
        FragColor = objectColor;
    } else {
        vec3 lightDir = normalize(vec3(0, 1, 0.5));
        float diff = max(dot(Normal, lightDir), 0.3);
        FragColor = vec4(objectColor.rgb * diff, objectColor.a);
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

# Global variables
running = True
pause = False
camera_pos = Vec3(0.0, 0.0, 15000.0)
camera_front = Vec3(0.0, 0.0, -1.0)
camera_up = Vec3(0.0, 1.0, 0.0)
last_x, last_y = 400.0, 300.0
yaw, pitch = -90.0, 0.0
delta_time, last_frame = 0.0, 0.0

G = 6.6743e-11
init_mass = 10**22
objs = []

class Object:
    def __init__(self, position, velocity, mass, radius, color):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.radius = radius  # Direct radius in world units
        self.color = color
        self.initializing = False
        
        self.vao = None
        self.vbo = None
        self.vertex_count = 0
        
        vertices = self.draw()
        self.vertex_count = len(vertices)
        self.create_vbo_vao(vertices)
        print(f"Created {color} sphere at {position} with radius {radius}")
    
    def draw(self):
        vertices = []
        stacks, sectors = 20, 20
        
        for i in range(stacks + 1):
            theta1 = (i / stacks) * math.pi
            theta2 = ((i + 1) / stacks) * math.pi
            
            for j in range(sectors):
                phi1 = (j / sectors) * 2 * math.pi
                phi2 = ((j + 1) / sectors) * 2 * math.pi
                
                # Vertices at radius r
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
    
    def update_pos(self):
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y
        self.position.z += self.velocity.z
    
    def accelerate(self, ax, ay, az):
        self.velocity.x += ax
        self.velocity.y += ay
        self.velocity.z += az
    
    def cleanup(self):
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])


def main():
    global camera_pos, camera_front, delta_time, last_frame, objs, pause, running
    
    if not glfw.init():
        print("Failed to initialize GLFW")
        return
    
    window = glfw.create_window(800, 600, "3D Gravity Simulation - WORKING", None, None)
    if not window:
        print("Failed to create window")
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.05, 0.1, 1.0)
    
    shader = compileProgram(
        compileShader(vertex_shader_source, GL_VERTEX_SHADER),
        compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    )
    
    glUseProgram(shader)
    
    # Setup projection
    projection = perspective(math.radians(45.0), 800.0/600.0, 1.0, 100000.0)
    proj_loc = glGetUniformLocation(shader, "projection")
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    
    # Create BIG visible objects
    objs = [
        Object(Vec3(-3000, 0, 0), Vec3(0, 0, 15), 10**24, 800, Vec4(0, 1, 1, 1)),   # Cyan left
        Object(Vec3(3000, 0, 0), Vec3(0, 0, -15), 10**24, 800, Vec4(0, 1, 1, 1)),   # Cyan right  
        Object(Vec3(0, 0, 0), Vec3(0, 0, 0), 10**27, 1200, Vec4(1, 0.9, 0.2, 1)),  # Yellow center
    ]
    
    print(f"\nCamera at {camera_pos}")
    print("You should see 2 cyan and 1 yellow sphere!")
    print("\nControls: W/S/A/D/Space/Shift - move, Mouse - look, K - pause, Q - quit\n")
    
    # Keyboard callback
    def key_callback(window, key, scancode, action, mods):
        global camera_pos, pause, running
        speed = 500.0
        
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
        if key == glfw.KEY_K and action == glfw.PRESS:
            pause = not pause
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            running = False
    
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
    
    while not glfw.window_should_close(window) and running:
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Update camera
        view = look_at(camera_pos, camera_pos + camera_front, camera_up)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        
        # Gravity
        if not pause:
            for obj in objs:
                for obj2 in objs:
                    if obj is not obj2:
                        dx = obj2.position.x - obj.position.x
                        dy = obj2.position.y - obj.position.y
                        dz = obj2.position.z - obj.position.z
                        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                        
                        if dist > 0:
                            dist_m = dist * 1000
                            force = G * obj.mass * obj2.mass / (dist_m * dist_m)
                            acc = force / obj.mass / 1000  # scale down
                            
                            obj.accelerate(
                                dx / dist * acc,
                                dy / dist * acc,
                                dz / dist * acc
                            )
                
                obj.update_pos()
        
        # Draw objects
        glUniform1i(glGetUniformLocation(shader, "isGrid"), 0)
        for obj in objs:
            model = translate(np.identity(4, dtype=np.float32), obj.position)
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
            glUniform4f(color_loc, obj.color.r, obj.color.g, obj.color.b, obj.color.a)
            
            glBindVertexArray(obj.vao)
            glDrawArrays(GL_TRIANGLES, 0, obj.vertex_count // 3)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    for obj in objs:
        obj.cleanup()
    
    glfw.terminate()


if __name__ == "__main__":
    main()