//=============================================================================================
// Framework for the ray tracing homework
// ---------------------------------------------------------------------------------------------
// Name    : 
// Neptun : 
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

struct vec3 {
	float x, y, z;

	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }

	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }

	vec3 operator+(const vec3& v) const {
		return vec3(x + v.x, y + v.y, z + v.z);
	}
	vec3 operator-(const vec3& v) const {
		return vec3(x - v.x, y - v.y, z - v.z);
	}
	vec3 operator*(const vec3& v) const {
		return vec3(x * v.x, y * v.y, z * v.z);
	}
	vec3 operator-() const {
		return vec3(-x, -y, -z);
	}
	vec3 normalize() const {
		return (*this) * (1 / (Length() + 0.000001));
	}
	float Length() const { return sqrtf(x * x + y * y + z * z); }

	operator float*() { return &x; }
};

float dot(const vec3& v1, const vec3& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}



void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

	out vec2 texcoord;

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates

	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";


struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}
};

class Vector {
public:
	const float x;
	const float y;
	const float z;
	const float w;
	Vector(float x, float y, float z, float w = 1) :x(x / w), y(y / w), z(z / w), w(1) {}
	Vector(const Vector& v2):x(v2.x),y(v2.y),z(v2.z),w(1) {}
	Vector(vec4 vector):x(vector.v[0]), y(vector.v[1]), z(vector.v[2]), w(vector.v[3]) {}
	const Vector operator+ (const Vector& v2) const {
		return Vector(x + v2.x, y + v2.y, z + v2.z, 1);
	}
	const Vector operator- (const Vector& v2) const {
		return Vector(x - v2.x, y - v2.y, z - v2.z, 1);
	}
	const Vector operator* (const float& f) const {
		return Vector(x * f, y * f, z * f, 1);
	}
	const Vector operator/ (const float& f) const {
		return Vector(x, y, z, f);
	}
	const Vector operator% (const Vector& v2) const {
		return Vector(y*v2.z- z*v2.y, z*v2.x - x*v2.z, x*v2.y - y*v2.x, 1);
	}
	const float operator* (const Vector& v2) const {
		return x*v2.x + y*v2.y + z*v2.z;
	}

	const vec4 operator()() {
		return vec4(x, y, z, w);
	}

	const float length() const {
		return sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
	}

	const Vector normalize() const {
		return Vector(x, y, z, length());
	}
};

// handle of the shader program
unsigned int shaderProgram;

class FullScreenTexturedQuad {
	unsigned int vao, textureId;	// vertex array object id and texture id
public:
	void Create(vec3 image[windowWidth * windowHeight]) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

								// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		static float vertexCoords[] = { -1, -1, 1, -1, -1, 1,
			1, -1, 1, 1, -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
																							   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

																	  // Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
		}
		glDrawArrays(GL_TRIANGLES, 0, 6);	// draw two triangles forming a quad
	}
};

struct Ray {
	const Vector position;
	const Vector orientation;

	Ray(Vector position, Vector orientation):position(position),orientation(orientation) {}
};

class Light {
public:
	const vec3 color;
	Light(const vec3& color):color(color) {}
	virtual Vector getDirection(const Vector& to) const = 0;
	virtual vec3 getIntensity(const Vector& to) const = 0;
};

class DotLight : public Light {
	const Vector position;
	const float range;
public:
	DotLight(const Vector& position, const float& range = 1000.0, const vec3& color = vec3(1,1,1)):position(position), range(range), Light(color) {}
	Vector getDirection(const Vector& to) const {
		return (position - to).normalize();
	}
	vec3 getIntensity(const Vector& to) const {
		return color*max((pow(range, 2)-pow((position - to).length(),2))/pow(range, 2),0);
	}
};

struct Hit;
class RoughMaterial {
	const vec3 color;
	const float shininess;
public:
	RoughMaterial(const vec3& color, const float& shininess) :color(color), shininess(shininess) {
		int i = 0;
	}
	const vec3 getColor(const Vector& position, const Vector& normal, const Vector& rayDir, const Light& light) const {
		Vector lightDir = light.getDirection(position);
		vec3 lightIntensity = light.getIntensity(position);

		float spekular = pow(max((lightDir + rayDir).normalize()*normal, 0.0), shininess);

		// a spekuláris intenzitása nem függ a távolságtól ???
		return this->color*lightIntensity*(max((lightDir*normal), 0)) + (lightIntensity*spekular);
	}
};

struct Hit {
	const Vector position;
	const Vector normal;
	const Vector rayDir;
	const RoughMaterial& roughMaterial;
	const float t;
	Hit(
		float t, const Vector position = vec4(), const Vector normal = vec4(), const Vector rayDir = vec4(), 
		const RoughMaterial& roughMaterial = RoughMaterial(vec3(), 0)
	) :position(position),normal(normal.normalize()),rayDir(rayDir.normalize()),t(t), roughMaterial(roughMaterial) {
		if (t > 0) {
			int i = 0;
		}
	}

	const vec3 getColor(const std::vector<Light*>& lights) const {
		if (t > 0) {
			vec3 color(0,0,0); // fény nélkül minden fekete
			for (std::vector<Light*>::const_iterator light = lights.begin(); light != lights.end(); light++) {
				color = color + roughMaterial.getColor(position, normal, rayDir, *(*light));
			}
			return color;
		}
		else {
			return vec3(0, 0, 0);
		}
	}
};


class Intersectable {
protected: // A leszármazottnak látni kell!!!!!
	const RoughMaterial roughMaterial;
public:
	Intersectable(const RoughMaterial& roughMaterial):roughMaterial(roughMaterial) {}
	virtual const Hit intersect(const Ray& ray) const = 0;
};

class Sphere : public Intersectable {
	const Vector position;
	const float r;
public:
	Sphere(Vector position, float r, const RoughMaterial& roughMaterial):position(position),r(r),Intersectable(roughMaterial) {}

	const Hit intersect(const Ray& ray) const {
		const Vector v = ray.orientation;
		const Vector eye = ray.position;
		const Vector c = position;
		const float R = r;

		// forrás: diasor
		// a1*t^2+a2*t+a3=0
		float a1 = v*v;
		float a2 = 2 * ((eye - c)*v);
		float a3 = ((eye - c)*(eye - c)) - pow(R, 2);
		if (pow(a2, 2) - 4 * a1*a3 > 0) {
			float t1 = (-a2 + sqrt(pow(a2, 2) - 4 * a1*a3)) / (2 * a1);
			float t2 = (-a2 - sqrt(pow(a2, 2) - 4 * a1*a3)) / (2 * a1);
			float t;
			if (t1 > 0 && t2 > 0) {
				t = min(t1, t2);
			}
			else {
				t = max(t1, t2);
			}
			return Hit(t, eye + v*t, eye + v*t - c, v*(-t), roughMaterial);
		}
		else {
			return Hit(-1);
		}
	}
};

class Camera {
	const Vector position;
	const Vector lookat;
	const Vector up;
	const Vector right;
	// The virtual world: single quad
	FullScreenTexturedQuad fullScreenTexturedQuad;

	vec3* background;	// The image, which stores the ray tracing result
public:
	Camera(Vector position, Vector lookat, Vector up, Vector right):position(position),lookat(lookat),up(up),right(right) {
		background = new vec3[windowWidth * windowHeight];
	}

	void render(const std::vector<Intersectable*>& objects, const std::vector<Light*>& lights) {
		// Ray tracing fills the image called background
		for (int x = 0; x < windowWidth; x++) {
			for (int y = 0; y < windowHeight; y++) {
				Vector _lookat = lookat;
				Vector _right = right*(x - windowWidth / 2.0) / (windowWidth / 2.0);
				Vector _up = up*(y - windowHeight / 2.0) / (windowHeight / 2.0);
				Ray ray(position, (
					_lookat +
					_right +
					_up
					).normalize());

				std::vector<Hit> hits;
				for (std::vector<Intersectable*>::const_iterator object = objects.begin(); object != objects.end(); object++) {
					hits.push_back((*object)->intersect(ray));
				}

				const Hit bestHit = getBestHit(hits);
				background[y * windowWidth + x] = bestHit.getColor(lights);
			}
		}
		fullScreenTexturedQuad.Create(background);
	}

	const Hit& getBestHit(const std::vector<Hit>& hits) const {
		const Hit* bestHit = &hits[0];
		for (std::vector<Hit>::const_iterator hit = hits.begin(); hit != hits.end(); hit++) {
			if (bestHit->t == -1 && (*hit).t > 0) {
				int i = 0;
			}
			if ((*hit).t > 0 && ((*hit).t < bestHit->t || bestHit->t < 0)) {
				bestHit = &(*hit);
			}
		}
		return *bestHit;
	}

	void draw() {
		fullScreenTexturedQuad.Draw();
	}

	~Camera() {
		delete[] background;
	}
};

Camera* camera;
std::vector<Intersectable*> objects;
std::vector<Light*> lights;

												// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);

	camera = new Camera(
		Vector(0,0,-500),
		Vector(0,0,1),
		Vector(0,1,0),
		Vector(1, 0, 0)
	);
	objects.push_back(new Sphere(Vector(0, 0, -150), 100, RoughMaterial(vec3(1,0,0),50)));
	objects.push_back(new Sphere(Vector(150, 0, 0), 100, RoughMaterial(vec3(0, 1, 0), 200)));
	lights.push_back(new DotLight(
		Vector(0, -300, 0)
	));

	lights.push_back(new DotLight(
		Vector(500, 500, -500)
	));
	lights.push_back(new DotLight(
		Vector(500, 500, -250)
	));
	camera->render(objects, lights);
	glutPostRedisplay();
}

void onExit() {
	glDeleteProgram(shaderProgram);
	delete camera;
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	camera->draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
