
/*
 * FLAME GPU v 1.5.X for CUDA 9
 * Copyright University of Sheffield.
 * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence
 * on www.flamegpu.com website.
 *
 */


// includes, project
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>

#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
	    
#include "header.h"
#include "visualisation.h"

#define FOVY 45.0

// bo variables
GLuint sphereVerts;
GLuint sphereNormals;

//Simulation output buffers/textures

cudaGraphicsResource_t Person_default_cgr;
GLuint Person_default_tbo;
GLuint Person_default_displacementTex;

cudaGraphicsResource_t Person_s2_cgr;
GLuint Person_s2_tbo;
GLuint Person_s2_displacementTex;

cudaGraphicsResource_t TBAssignment_tbdefault_cgr;
GLuint TBAssignment_tbdefault_tbo;
GLuint TBAssignment_tbdefault_displacementTex;

cudaGraphicsResource_t Household_hhdefault_cgr;
GLuint Household_hhdefault_tbo;
GLuint Household_hhdefault_displacementTex;

cudaGraphicsResource_t HouseholdMembership_hhmembershipdefault_cgr;
GLuint HouseholdMembership_hhmembershipdefault_tbo;
GLuint HouseholdMembership_hhmembershipdefault_displacementTex;

cudaGraphicsResource_t Church_chudefault_cgr;
GLuint Church_chudefault_tbo;
GLuint Church_chudefault_displacementTex;

cudaGraphicsResource_t ChurchMembership_chumembershipdefault_cgr;
GLuint ChurchMembership_chumembershipdefault_tbo;
GLuint ChurchMembership_chumembershipdefault_displacementTex;

cudaGraphicsResource_t Transport_trdefault_cgr;
GLuint Transport_trdefault_tbo;
GLuint Transport_trdefault_displacementTex;

cudaGraphicsResource_t TransportMembership_trmembershipdefault_cgr;
GLuint TransportMembership_trmembershipdefault_tbo;
GLuint TransportMembership_trmembershipdefault_displacementTex;

cudaGraphicsResource_t Clinic_cldefault_cgr;
GLuint Clinic_cldefault_tbo;
GLuint Clinic_cldefault_displacementTex;

cudaGraphicsResource_t Workplace_wpdefault_cgr;
GLuint Workplace_wpdefault_tbo;
GLuint Workplace_wpdefault_displacementTex;

cudaGraphicsResource_t WorkplaceMembership_wpmembershipdefault_cgr;
GLuint WorkplaceMembership_wpmembershipdefault_tbo;
GLuint WorkplaceMembership_wpmembershipdefault_displacementTex;


// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -VIEW_DISTANCE;

// keyboard controls
#if defined(PAUSE_ON_START)
bool paused = true;
#else
bool paused = false;
#endif

// vertex Shader
GLuint vertexShader;
GLuint fragmentShader;
GLuint shaderProgram;
GLuint vs_displacementMap;
GLuint vs_mapIndex;



//timer
cudaEvent_t start, stop;
const int display_rate = 50;
int frame_count;
float frame_time = 0.0;

#ifdef SIMULATION_DELAY
//delay
int delay_count = 0;
#endif

// prototypes
int initGL();
void initShader();
void createVBO( GLuint* vbo, GLuint size);
void deleteVBO( GLuint* vbo);
void createTBO( cudaGraphicsResource_t* cudaResource, GLuint* tbo, GLuint* tex, GLuint size);
void deleteTBO( cudaGraphicsResource_t* cudaResource, GLuint* tbo);
void setVertexBufferData();
void reshape(int width, int height);
void display();
void keyboard( unsigned char key, int x, int y);
void special(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void runCuda();
void checkGLError();

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
   
}

const char vertexShaderSource[] = 
{  
	"#extension GL_EXT_gpu_shader4 : enable										\n"
	"uniform samplerBuffer displacementMap;										\n"
	"attribute in float mapIndex;												\n"
	"varying vec3 normal, lightDir;												\n"
	"varying vec4 colour;														\n"
    "void main()																\n"
    "{																			\n"
	"	vec4 position = gl_Vertex;											    \n"
	"	vec4 lookup = texelFetchBuffer(displacementMap, (int)mapIndex);		    \n"
    "	if (lookup.w > 7.5)	                								\n"
	"		colour = vec4(0.518, 0.353, 0.02, 0.0);						    	\n"
    "	else if (lookup.w > 6.5)	               								\n"
	"		colour = vec4(1.0, 1.0, 1.0, 0.0);								    \n"
    "	else if (lookup.w > 5.5)	                							\n"
	"		colour = vec4(1.0, 0.0, 1.0, 0.0);								    \n"
	"	else if (lookup.w > 4.5)	                							\n"
	"		colour = vec4(0.0, 1.0, 1.0, 0.0);								    \n"
    "	else if (lookup.w > 3.5)	                							\n"
	"		colour = vec4(1.0, 1.0, 0.0, 0.0);								    \n"
	"	else if (lookup.w > 2.5)	                							\n"
	"		colour = vec4(0.0, 0.0, 1.0, 0.0);								    \n"
	"	else if (lookup.w > 1.5)	                							\n"
	"		colour = vec4(0.0, 1.0, 0.0, 0.0);								    \n"
    "	else if (lookup.w > 0.5)	                							\n"
	"		colour = vec4(1.0, 0.0, 0.0, 0.0);								    \n"
    "	else                      	                							\n"
	"		colour = vec4(0.0, 0.0, 0.0, 0.0);								    \n"
	"																    		\n"
	"	lookup.w = 1.0;												    		\n"
	"	position += lookup;											    		\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * position;		    		\n"
	"																			\n"
	"	vec3 mvVertex = vec3(gl_ModelViewMatrix * position);			    	\n"
	"	lightDir = vec3(gl_LightSource[0].position.xyz - mvVertex);				\n"
	"	normal = gl_NormalMatrix * gl_Normal;									\n"
    "}																			\n"
};

const char fragmentShaderSource[] = 
{  
	"varying vec3 normal, lightDir;												\n"
	"varying vec4 colour;														\n"
	"void main (void)															\n"
	"{																			\n"
	"	// Defining The Material Colors											\n"
	"	vec4 AmbientColor = vec4(0.25, 0.0, 0.0, 1.0);							\n"
	"	vec4 DiffuseColor = colour;					            		    	\n"
	"																			\n"
	"	// Scaling The Input Vector To Length 1									\n"
	"	vec3 n_normal = normalize(normal);							        	\n"
	"	vec3 n_lightDir = normalize(lightDir);	                                \n"
	"																			\n"
	"	// Calculating The Diffuse Term And Clamping It To [0;1]				\n"
	"	float DiffuseTerm = clamp(dot(n_normal, n_lightDir), 0.0, 1.0);\n"
	"																			\n"
	"	// Calculating The Final Color											\n"
	"	gl_FragColor = AmbientColor + DiffuseColor * DiffuseTerm;				\n"
	"																			\n"
	"}																			\n"
};

//GPU Kernels

__global__ void output_Person_agent_to_VBO(xmachine_memory_Person_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
    vbo[index].x = 0.0;
    vbo[index].y = 0.0;
    vbo[index].z = 0.0;
    vbo[index].w = 1.0;
}

__global__ void output_TBAssignment_agent_to_VBO(xmachine_memory_TBAssignment_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
    vbo[index].x = 0.0;
    vbo[index].y = 0.0;
    vbo[index].z = 0.0;
    vbo[index].w = 1.0;
}

__global__ void output_Household_agent_to_VBO(xmachine_memory_Household_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
    vbo[index].x = 0.0;
    vbo[index].y = 0.0;
    vbo[index].z = 0.0;
    vbo[index].w = 1.0;
}

__global__ void output_HouseholdMembership_agent_to_VBO(xmachine_memory_HouseholdMembership_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
    vbo[index].x = 0.0;
    vbo[index].y = 0.0;
    vbo[index].z = 0.0;
    vbo[index].w = 1.0;
}

__global__ void output_Church_agent_to_VBO(xmachine_memory_Church_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
    vbo[index].x = 0.0;
    vbo[index].y = 0.0;
    vbo[index].z = 0.0;
    vbo[index].w = 1.0;
}

__global__ void output_ChurchMembership_agent_to_VBO(xmachine_memory_ChurchMembership_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
    vbo[index].x = 0.0;
    vbo[index].y = 0.0;
    vbo[index].z = 0.0;
    vbo[index].w = 1.0;
}

__global__ void output_Transport_agent_to_VBO(xmachine_memory_Transport_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
    vbo[index].x = 0.0;
    vbo[index].y = 0.0;
    vbo[index].z = 0.0;
    vbo[index].w = 1.0;
}

__global__ void output_TransportMembership_agent_to_VBO(xmachine_memory_TransportMembership_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
    vbo[index].x = 0.0;
    vbo[index].y = 0.0;
    vbo[index].z = 0.0;
    vbo[index].w = 1.0;
}

__global__ void output_Clinic_agent_to_VBO(xmachine_memory_Clinic_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
    vbo[index].x = 0.0;
    vbo[index].y = 0.0;
    vbo[index].z = 0.0;
    vbo[index].w = 1.0;
}

__global__ void output_Workplace_agent_to_VBO(xmachine_memory_Workplace_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
    vbo[index].x = 0.0;
    vbo[index].y = 0.0;
    vbo[index].z = 0.0;
    vbo[index].w = 1.0;
}

__global__ void output_WorkplaceMembership_agent_to_VBO(xmachine_memory_WorkplaceMembership_list* agents, glm::vec4* vbo, glm::vec3 centralise){

	//global thread index
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	vbo[index].x = 0.0;
	vbo[index].y = 0.0;
	vbo[index].z = 0.0;
	
    vbo[index].x = 0.0;
    vbo[index].y = 0.0;
    vbo[index].z = 0.0;
    vbo[index].w = 1.0;
}


void initVisualisation()
{
	// Create GL context
	int   argc   = 1;
        char glutString[] = "GLUT application"; 
	char *argv[] = {glutString, NULL};
	//char *argv[] = {"GLUT application", NULL};
	glutInit( &argc, argv);
	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize( WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow( "FLAME GPU Visualiser");

	// initialize GL
	if( !initGL()) {
			return;
	}
	initShader();

	// register callbacks
	glutReshapeFunc( reshape);
	glutDisplayFunc( display);
	glutKeyboardFunc( keyboard);
	glutSpecialFunc( special);
	glutMouseFunc( mouse);
	glutMotionFunc( motion);
    
	// create VBO's
	createVBO( &sphereVerts, SPHERE_SLICES* (SPHERE_STACKS+1) * sizeof(glm::vec3));
	createVBO( &sphereNormals, SPHERE_SLICES* (SPHERE_STACKS+1) * sizeof (glm::vec3));
	setVertexBufferData();

	// create TBO
	createTBO(&Person_default_cgr, &Person_default_tbo, &Person_default_displacementTex, xmachine_memory_Person_MAX * sizeof( glm::vec4));
	
	createTBO(&Person_s2_cgr, &Person_s2_tbo, &Person_s2_displacementTex, xmachine_memory_Person_MAX * sizeof( glm::vec4));
	
	createTBO(&TBAssignment_tbdefault_cgr, &TBAssignment_tbdefault_tbo, &TBAssignment_tbdefault_displacementTex, xmachine_memory_TBAssignment_MAX * sizeof( glm::vec4));
	
	createTBO(&Household_hhdefault_cgr, &Household_hhdefault_tbo, &Household_hhdefault_displacementTex, xmachine_memory_Household_MAX * sizeof( glm::vec4));
	
	createTBO(&HouseholdMembership_hhmembershipdefault_cgr, &HouseholdMembership_hhmembershipdefault_tbo, &HouseholdMembership_hhmembershipdefault_displacementTex, xmachine_memory_HouseholdMembership_MAX * sizeof( glm::vec4));
	
	createTBO(&Church_chudefault_cgr, &Church_chudefault_tbo, &Church_chudefault_displacementTex, xmachine_memory_Church_MAX * sizeof( glm::vec4));
	
	createTBO(&ChurchMembership_chumembershipdefault_cgr, &ChurchMembership_chumembershipdefault_tbo, &ChurchMembership_chumembershipdefault_displacementTex, xmachine_memory_ChurchMembership_MAX * sizeof( glm::vec4));
	
	createTBO(&Transport_trdefault_cgr, &Transport_trdefault_tbo, &Transport_trdefault_displacementTex, xmachine_memory_Transport_MAX * sizeof( glm::vec4));
	
	createTBO(&TransportMembership_trmembershipdefault_cgr, &TransportMembership_trmembershipdefault_tbo, &TransportMembership_trmembershipdefault_displacementTex, xmachine_memory_TransportMembership_MAX * sizeof( glm::vec4));
	
	createTBO(&Clinic_cldefault_cgr, &Clinic_cldefault_tbo, &Clinic_cldefault_displacementTex, xmachine_memory_Clinic_MAX * sizeof( glm::vec4));
	
	createTBO(&Workplace_wpdefault_cgr, &Workplace_wpdefault_tbo, &Workplace_wpdefault_displacementTex, xmachine_memory_Workplace_MAX * sizeof( glm::vec4));
	
	createTBO(&WorkplaceMembership_wpmembershipdefault_cgr, &WorkplaceMembership_wpmembershipdefault_tbo, &WorkplaceMembership_wpmembershipdefault_displacementTex, xmachine_memory_WorkplaceMembership_MAX * sizeof( glm::vec4));
	

	//set shader uniforms
	glUseProgram(shaderProgram);

	//create a events for timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

void runVisualisation(){
	// start rendering mainloop
	glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
	if(!paused){
#ifdef SIMULATION_DELAY
	delay_count++;
	if (delay_count == SIMULATION_DELAY){
		delay_count = 0;
		singleIteration();
	}
#else
	singleIteration();
#endif
	}

	//kernals sizes
	int threads_per_tile = 256;
	int tile_size;
	dim3 grid;
	dim3 threads;
	glm::vec3 centralise;

	//pointer
	glm::vec4 *dptr;

	
	if (get_agent_Person_default_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &Person_default_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, Person_default_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_Person_default_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_Person_agent_to_VBO<<< grid, threads>>>(get_device_Person_default_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &Person_default_cgr));
	}
	
	if (get_agent_Person_s2_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &Person_s2_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, Person_s2_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_Person_s2_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_Person_agent_to_VBO<<< grid, threads>>>(get_device_Person_s2_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &Person_s2_cgr));
	}
	
	if (get_agent_TBAssignment_tbdefault_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &TBAssignment_tbdefault_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, TBAssignment_tbdefault_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_TBAssignment_tbdefault_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_TBAssignment_agent_to_VBO<<< grid, threads>>>(get_device_TBAssignment_tbdefault_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &TBAssignment_tbdefault_cgr));
	}
	
	if (get_agent_Household_hhdefault_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &Household_hhdefault_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, Household_hhdefault_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_Household_hhdefault_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_Household_agent_to_VBO<<< grid, threads>>>(get_device_Household_hhdefault_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &Household_hhdefault_cgr));
	}
	
	if (get_agent_HouseholdMembership_hhmembershipdefault_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &HouseholdMembership_hhmembershipdefault_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, HouseholdMembership_hhmembershipdefault_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_HouseholdMembership_hhmembershipdefault_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_HouseholdMembership_agent_to_VBO<<< grid, threads>>>(get_device_HouseholdMembership_hhmembershipdefault_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &HouseholdMembership_hhmembershipdefault_cgr));
	}
	
	if (get_agent_Church_chudefault_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &Church_chudefault_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, Church_chudefault_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_Church_chudefault_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_Church_agent_to_VBO<<< grid, threads>>>(get_device_Church_chudefault_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &Church_chudefault_cgr));
	}
	
	if (get_agent_ChurchMembership_chumembershipdefault_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &ChurchMembership_chumembershipdefault_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, ChurchMembership_chumembershipdefault_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_ChurchMembership_chumembershipdefault_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_ChurchMembership_agent_to_VBO<<< grid, threads>>>(get_device_ChurchMembership_chumembershipdefault_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &ChurchMembership_chumembershipdefault_cgr));
	}
	
	if (get_agent_Transport_trdefault_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &Transport_trdefault_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, Transport_trdefault_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_Transport_trdefault_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_Transport_agent_to_VBO<<< grid, threads>>>(get_device_Transport_trdefault_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &Transport_trdefault_cgr));
	}
	
	if (get_agent_TransportMembership_trmembershipdefault_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &TransportMembership_trmembershipdefault_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, TransportMembership_trmembershipdefault_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_TransportMembership_trmembershipdefault_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_TransportMembership_agent_to_VBO<<< grid, threads>>>(get_device_TransportMembership_trmembershipdefault_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &TransportMembership_trmembershipdefault_cgr));
	}
	
	if (get_agent_Clinic_cldefault_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &Clinic_cldefault_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, Clinic_cldefault_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_Clinic_cldefault_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_Clinic_agent_to_VBO<<< grid, threads>>>(get_device_Clinic_cldefault_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &Clinic_cldefault_cgr));
	}
	
	if (get_agent_Workplace_wpdefault_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &Workplace_wpdefault_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, Workplace_wpdefault_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_Workplace_wpdefault_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_Workplace_agent_to_VBO<<< grid, threads>>>(get_device_Workplace_wpdefault_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &Workplace_wpdefault_cgr));
	}
	
	if (get_agent_WorkplaceMembership_wpmembershipdefault_count() > 0)
	{
		// map OpenGL buffer object for writing from CUDA
        size_t accessibleBufferSize = 0;
        gpuErrchk(cudaGraphicsMapResources(1, &WorkplaceMembership_wpmembershipdefault_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer( (void**)&dptr, &accessibleBufferSize, WorkplaceMembership_wpmembershipdefault_cgr));
		//cuda block size
		tile_size = (int) ceil((float)get_agent_WorkplaceMembership_wpmembershipdefault_count()/threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
        
        //continuous variables  
        centralise = getMaximumBounds() + getMinimumBounds();
        centralise /= 2;
        
		output_WorkplaceMembership_agent_to_VBO<<< grid, threads>>>(get_device_WorkplaceMembership_wpmembershipdefault_agents(), dptr, centralise);
		gpuErrchkLaunch();
		// unmap buffer object
        gpuErrchk(cudaGraphicsUnmapResources(1, &WorkplaceMembership_wpmembershipdefault_cgr));
	}
	
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
int initGL()
{
	// initialize necessary OpenGL extensions
	glewInit();
	if (! glewIsSupported( "GL_VERSION_2_0 " 
		"GL_ARB_pixel_buffer_object")) {
		fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
		fflush( stderr);
		return 1;
	}

	// default initialization
	glClearColor( 1.0, 1.0, 1.0, 1.0);
	glEnable( GL_DEPTH_TEST);

	reshape(WINDOW_WIDTH, WINDOW_HEIGHT);
	checkGLError();

	//lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	return 1;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GLSL Vertex Shader
////////////////////////////////////////////////////////////////////////////////
void initShader()
{
	const char* v = vertexShaderSource;
	const char* f = fragmentShaderSource;

	//vertex shader
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &v, 0);
	glCompileShader(vertexShader);

	//fragment shader
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &f, 0);
	glCompileShader(fragmentShader);

	//program
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// check for errors
	GLint status;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(vertexShader, 262144, &len, data); 
		printf("%s", data);
	}
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(fragmentShader, 262144, &len, data); 
		printf("%s", data);
	}
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Program Link Error\n");
	}

	// get shader variables
	vs_displacementMap = glGetUniformLocation(shaderProgram, "displacementMap");
	vs_mapIndex = glGetAttribLocation(shaderProgram, "mapIndex"); 
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, GLuint size)
{
	// create buffer object
	glGenBuffers( 1, vbo);
	glBindBuffer( GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData( GL_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

	glBindBuffer( GL_ARRAY_BUFFER, 0);

	checkGLError();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO( GLuint* vbo)
{
	glBindBuffer( 1, *vbo);
	glDeleteBuffers( 1, vbo);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Create TBO
////////////////////////////////////////////////////////////////////////////////
void createTBO(cudaGraphicsResource_t* cudaResource, GLuint* tbo, GLuint* tex, GLuint size)
{
	// create buffer object
	glGenBuffers( 1, tbo);
	glBindBuffer( GL_TEXTURE_BUFFER_EXT, *tbo);

	// initialize buffer object
	glBufferData( GL_TEXTURE_BUFFER_EXT, size, 0, GL_DYNAMIC_DRAW);

	//tex
	glGenTextures(1, tex);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, *tex);
	glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, *tbo); 
	glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);

    // register buffer object with CUDA
    gpuErrchk(cudaGraphicsGLRegisterBuffer(cudaResource, *tbo, cudaGraphicsMapFlagsWriteDiscard)); 

    checkGLError();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete TBO
////////////////////////////////////////////////////////////////////////////////
void deleteTBO(cudaGraphicsResource_t* cudaResource,  GLuint* tbo)
{
    gpuErrchk(cudaGraphicsUnregisterResource(*cudaResource));
    *cudaResource = 0;

    glBindBuffer( 1, *tbo);
    glDeleteBuffers( 1, tbo);

	*tbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Set Sphere Vertex Data
////////////////////////////////////////////////////////////////////////////////

static void setSphereVertex(glm::vec3* data, int slice, int stack) {
	float PI = 3.14159265358;
    
	double sl = 2*PI*slice/SPHERE_SLICES;
	double st = 2*PI*stack/SPHERE_STACKS;
 
	data->x = cos(st)*sin(sl) * SPHERE_RADIUS;
	data->y = sin(st)*sin(sl) * SPHERE_RADIUS;
	data->z = cos(sl) * SPHERE_RADIUS;
}


////////////////////////////////////////////////////////////////////////////////
//! Set Sphere Normal Data
////////////////////////////////////////////////////////////////////////////////

static void setSphereNormal(glm::vec3* data, int slice, int stack) {
	float PI = 3.14159265358;
    
	double sl = 2*PI*slice/SPHERE_SLICES;
	double st = 2*PI*stack/SPHERE_STACKS;
 
	data->x = cos(st)*sin(sl);
	data->y = sin(st)*sin(sl);
	data->z = cos(sl);
}


////////////////////////////////////////////////////////////////////////////////
//! Set Vertex Buffer Data
////////////////////////////////////////////////////////////////////////////////
void setVertexBufferData()
{
	int slice, stack;
	int i;

	// upload vertex points data
	glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
	glm::vec3* verts =( glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice=0; slice<SPHERE_SLICES/2; slice++) {
		for (stack=0; stack<=SPHERE_STACKS; stack++) {
			setSphereVertex(&verts[i++], slice, stack);
			setSphereVertex(&verts[i++], slice+1, stack);
		}
    }
	glUnmapBuffer(GL_ARRAY_BUFFER);

	// upload vertex normal data
	glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
	glm::vec3* normals =( glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice=0; slice<SPHERE_SLICES/2; slice++) {
		for (stack=0; stack<=SPHERE_STACKS; stack++) {
			setSphereNormal(&normals[i++], slice, stack);
			setSphereNormal(&normals[i++], slice+1, stack);
		}
    }
	glUnmapBuffer(GL_ARRAY_BUFFER);
}




////////////////////////////////////////////////////////////////////////////////
//! Reshape callback
////////////////////////////////////////////////////////////////////////////////

void reshape(int width, int height){
	// viewport
	glViewport( 0, 0, width, height);

	// projection
	glMatrixMode( GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(FOVY, (GLfloat)width / (GLfloat) height, NEAR_CLIP, FAR_CLIP);

	checkGLError();
}


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	float millis;
	
	//CUDA start Timing
	cudaEventRecord(start);

	// run CUDA kernel to generate vertex positions
	runCuda();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	//zoom
	glTranslatef(0.0, 0.0, translate_z); 
	//move
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 0.0, 1.0);


	//Set light position
	glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION);

	
	//Draw Person Agents in default state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, Person_default_displacementTex);
	//loop
	for (int i=0; i< get_agent_Person_default_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw Person Agents in s2 state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, Person_s2_displacementTex);
	//loop
	for (int i=0; i< get_agent_Person_s2_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw TBAssignment Agents in tbdefault state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, TBAssignment_tbdefault_displacementTex);
	//loop
	for (int i=0; i< get_agent_TBAssignment_tbdefault_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw Household Agents in hhdefault state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, Household_hhdefault_displacementTex);
	//loop
	for (int i=0; i< get_agent_Household_hhdefault_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw HouseholdMembership Agents in hhmembershipdefault state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, HouseholdMembership_hhmembershipdefault_displacementTex);
	//loop
	for (int i=0; i< get_agent_HouseholdMembership_hhmembershipdefault_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw Church Agents in chudefault state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, Church_chudefault_displacementTex);
	//loop
	for (int i=0; i< get_agent_Church_chudefault_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw ChurchMembership Agents in chumembershipdefault state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, ChurchMembership_chumembershipdefault_displacementTex);
	//loop
	for (int i=0; i< get_agent_ChurchMembership_chumembershipdefault_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw Transport Agents in trdefault state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, Transport_trdefault_displacementTex);
	//loop
	for (int i=0; i< get_agent_Transport_trdefault_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw TransportMembership Agents in trmembershipdefault state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, TransportMembership_trmembershipdefault_displacementTex);
	//loop
	for (int i=0; i< get_agent_TransportMembership_trmembershipdefault_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw Clinic Agents in cldefault state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, Clinic_cldefault_displacementTex);
	//loop
	for (int i=0; i< get_agent_Clinic_cldefault_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw Workplace Agents in wpdefault state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, Workplace_wpdefault_displacementTex);
	//loop
	for (int i=0; i< get_agent_Workplace_wpdefault_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	//Draw WorkplaceMembership Agents in wpmembershipdefault state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, WorkplaceMembership_wpmembershipdefault_displacementTex);
	//loop
	for (int i=0; i< get_agent_WorkplaceMembership_wpmembershipdefault_count(); i++){
		glVertexAttrib1f(vs_mapIndex, (float)i);
		
		//draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);

		glBindBuffer(GL_ARRAY_BUFFER, sphereNormals);
		glNormalPointer(GL_FLOAT, 0, 0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, SPHERE_SLICES * (SPHERE_STACKS+1));

		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	

	//CUDA stop timing
	cudaEventRecord(stop);
	glFlush();
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);
  frame_time += millis;

	if(frame_count == display_rate){
		char title [100];
		sprintf(title, "Execution & Rendering Total: %f (FPS), %f milliseconds per frame", display_rate/(frame_time/1000.0f), frame_time/display_rate);
		glutSetWindowTitle(title);

		//reset
		frame_count = 0;
    frame_time = 0.0;
	}else{
		frame_count++;
	}


	glutSwapBuffers();
	glutPostRedisplay();

}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
	switch( key) {
	// Space == 32
    case(32):
        paused = !paused;
        break;
    // Esc == 27
	case(27) :
		deleteVBO( &sphereVerts);
		deleteVBO( &sphereNormals);
		
		deleteTBO( &Person_default_cgr, &Person_default_tbo);
		
		deleteTBO( &Person_s2_cgr, &Person_s2_tbo);
		
		deleteTBO( &TBAssignment_tbdefault_cgr, &TBAssignment_tbdefault_tbo);
		
		deleteTBO( &Household_hhdefault_cgr, &Household_hhdefault_tbo);
		
		deleteTBO( &HouseholdMembership_hhmembershipdefault_cgr, &HouseholdMembership_hhmembershipdefault_tbo);
		
		deleteTBO( &Church_chudefault_cgr, &Church_chudefault_tbo);
		
		deleteTBO( &ChurchMembership_chumembershipdefault_cgr, &ChurchMembership_chumembershipdefault_tbo);
		
		deleteTBO( &Transport_trdefault_cgr, &Transport_trdefault_tbo);
		
		deleteTBO( &TransportMembership_trmembershipdefault_cgr, &TransportMembership_trmembershipdefault_tbo);
		
		deleteTBO( &Clinic_cldefault_cgr, &Clinic_cldefault_tbo);
		
		deleteTBO( &Workplace_wpdefault_cgr, &Workplace_wpdefault_tbo);
		
		deleteTBO( &WorkplaceMembership_wpmembershipdefault_cgr, &WorkplaceMembership_wpmembershipdefault_tbo);
		
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		exit(EXIT_SUCCESS);
	}
}




void special(int key, int x, int y){
    switch (key)
    {
    case(GLUT_KEY_RIGHT) :
        singleIteration();
        fflush(stdout);
        break;
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1<<button;
	} else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2;
		rotate_y += dx * 0.2;
	} else if (mouse_buttons & 4) {
		translate_z += dy * VIEW_DISTANCE * 0.001;
	}

  mouse_old_x = x;
  mouse_old_y = y;
}

void checkGLError(){
  int Error;
  if((Error = glGetError()) != GL_NO_ERROR)
  {
    const char* Message = (const char*)gluErrorString(Error);
    fprintf(stderr, "OpenGL Error : %s\n", Message);
  }
}
