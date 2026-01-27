import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import ctypes

# ============================
# SHADERS (unchanged visually)
# ============================

vertex_shader_source = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 Normal;
void main() {
    Normal = normalize(aPos);
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
in vec3 Normal;
out vec4 FragColor;
uniform vec4 objectColor;
uniform bool glow;
void main() {
    if (glow) {
        FragColor = vec4(objectColor.rgb * 2.5, objectColor.a);
    } else {
        vec3 lightDir = normalize(vec3(0.4, 1.0, 0.6));
        float diff = max(dot(Normal, lightDir), 0.2);
        FragColor = vec4(objectColor.rgb * diff, objectColor.a);
    }
}
"""

# ============================
# BASIC MATH
# ============================

class Vec3:
    def __init__(self,x=0,y=0,z=0):
        self.x,self.y,self.z = float(x),float(y),float(z)
    def __add__(self,o): return Vec3(self.x+o.x,self.y+o.y,self.z+o.z)
    def __sub__(self,o): return Vec3(self.x-o.x,self.y-o.y,self.z-o.z)
    def __mul__(self,s): return Vec3(self.x*s,self.y*s,self.z*s)
    def length(self): return math.sqrt(self.x**2+self.y**2+self.z**2)

def perspective(fovy, aspect, near, far):
    f = 1.0/math.tan(fovy/2)
    m = np.zeros((4,4),np.float32)
    m[0,0]=f/aspect; m[1,1]=f
    m[2,2]=(far+near)/(near-far)
    m[2,3]=(2*far*near)/(near-far)
    m[3,2]=-1
    return m

def look_at(eye,center,up):
    f=(center-eye); fl=f.length(); f=Vec3(f.x/fl,f.y/fl,f.z/fl)
    s=Vec3(f.y*up.z-f.z*up.y,f.z*up.x-f.x*up.z,f.x*up.y-f.y*up.x)
    sl=s.length(); s=Vec3(s.x/sl,s.y/sl,s.z/sl)
    u=Vec3(s.y*f.z-s.z*f.y,s.z*f.x-s.x*f.z,s.x*f.y-s.y*f.x)
    m=np.identity(4,np.float32)
    m[0,:3]=[s.x,u.x,-f.x]
    m[1,:3]=[s.y,u.y,-f.y]
    m[2,:3]=[s.z,u.z,-f.z]
    m[3,0]=- (s.x*eye.x+s.y*eye.y+s.z*eye.z)
    m[3,1]=- (u.x*eye.x+u.y*eye.y+u.z*eye.z)
    m[3,2]=  (f.x*eye.x+f.y*eye.y+f.z*eye.z)
    return m

# ============================
# PHYSICS PARAMETERS
# ============================

G = 1.0            # normalized gravitational constant
EPS = 50.0         # softening length
DT = 0.005         # FIXED timestep (critical)

# ============================
# OBJECT
# ============================

class Body:
    def __init__(self,pos,vel,mass,radius,color,glow=False,fixed=False):
        self.pos=pos
        self.vel=vel
        self.acc=Vec3()
        self.mass=mass
        self.radius=radius
        self.color=color
        self.glow=glow
        self.fixed=fixed

        self.vao=glGenVertexArrays(1)
        self.vbo=glGenBuffers(1)
        verts=[]
        for i in range(12):
            for j in range(12):
                verts+=[0,0,0]  # cheap sphere placeholder
        verts=np.array(verts,np.float32)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER,self.vbo)
        glBufferData(GL_ARRAY_BUFFER,verts.nbytes,verts,GL_STATIC_DRAW)
        glVertexAttribPointer(0,3,GL_FLOAT,False,0,None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

# ============================
# FORCE COMPUTATION
# ============================

def compute_accelerations(bodies):
    for a in bodies:
        a.acc=Vec3()
    for i,a in enumerate(bodies):
        for j,b in enumerate(bodies):
            if i==j: continue
            dx=b.pos.x-a.pos.x
            dy=b.pos.y-a.pos.y
            dz=b.pos.z-a.pos.z
            r2=dx*dx+dy*dy+dz*dz+EPS*EPS
            invr=1/math.sqrt(r2)
            f=G*b.mass/r2
            a.acc.x+=f*dx*invr
            a.acc.y+=f*dy*invr
            a.acc.z+=f*dz*invr

# ============================
# MAIN
# ============================

def main():
    glfw.init()
    win=glfw.create_window(1200,800,"Accurate N-Body Simulation",None,None)
    glfw.make_context_current(win)
    glEnable(GL_DEPTH_TEST)

    shader=compileProgram(
        compileShader(vertex_shader_source,GL_VERTEX_SHADER),
        compileShader(fragment_shader_source,GL_FRAGMENT_SHADER)
    )
    glUseProgram(shader)

    proj=perspective(math.radians(45),1.5,1,50000)
    glUniformMatrix4fv(glGetUniformLocation(shader,"projection"),1,False,proj)

    # Bodies
    star=Body(Vec3(0,0,0),Vec3(0,0,0),1000,500,[1,1,0,1],True,True)
    r=6000
    v=math.sqrt(G*star.mass/r)
    p1=Body(Vec3(r,0,0),Vec3(0,0,v),1,100,[0.4,0.6,1,1])
    bodies=[star,p1]

    # INITIAL ACCELERATION
    compute_accelerations(bodies)

    cam=Vec3(0,8000,12000)
    up=Vec3(0,1,0)

    while not glfw.window_should_close(win):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # ===== LEAPFROG (CORRECT) =====
        for b in bodies:
            if not b.fixed:
                b.vel.x+=b.acc.x*DT*0.5
                b.vel.y+=b.acc.y*DT*0.5
                b.vel.z+=b.acc.z*DT*0.5

        for b in bodies:
            if not b.fixed:
                b.pos.x+=b.vel.x*DT
                b.pos.y+=b.vel.y*DT
                b.pos.z+=b.vel.z*DT

        compute_accelerations(bodies)

        for b in bodies:
            if not b.fixed:
                b.vel.x+=b.acc.x*DT*0.5
                b.vel.y+=b.acc.y*DT*0.5
                b.vel.z+=b.acc.z*DT*0.5

        # ==============================

        view=look_at(cam,Vec3(),up)
        glUniformMatrix4fv(glGetUniformLocation(shader,"view"),1,False,view)

        for b in bodies:
            model=np.identity(4,np.float32)
            model[3,:3]=[b.pos.x,b.pos.y,b.pos.z]
            glUniformMatrix4fv(glGetUniformLocation(shader,"model"),1,False,model)
            glUniform4f(glGetUniformLocation(shader,"objectColor"),*b.color)
            glUniform1i(glGetUniformLocation(shader,"glow"),b.glow)
            glBindVertexArray(b.vao)
            glDrawArrays(GL_TRIANGLES,0,36)

        glfw.swap_buffers(win)
        glfw.poll_events()

    glfw.terminate()

if __name__=="__main__":
    main()
