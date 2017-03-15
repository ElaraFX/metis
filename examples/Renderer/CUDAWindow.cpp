#include "CUDAWindow.h"

#include <QMouseEvent>

#include <sstream>
#include <iostream>
#include "SceneLoader.h"
#include "Camera.h"
#include "Array.h"
#include "Scene.h"
#include "Util.h"
#include "BVH.h"
#include "CudaBVH.h"
#include "CudaRenderKernel.h"
#include "HDRloader.h"
#include "MouseKeyboardInput.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "linear_math.h"
#include <gl/GLU.h>

TriangleWindow::TriangleWindow(QWidget *parent)
    : QOpenGLWidget(parent)
    , m_frame(0)
{
    cudaNodePtr = NULL;
    cudaTriWoopPtr = NULL;
    cudaTriDebugPtr = NULL;
	cudaTriUVPtr = NULL;
    cudaTriNormalPtr = NULL;
    cudaTriIndicesPtr = NULL;
	cudaMaterialsPtr = NULL;
	cudaTexturePtr = NULL;
	cudaTextureData = NULL;

    cudaRendercam = NULL;
    hostRendercam = NULL;
    accumulatebuffer = NULL; // image buffer storing accumulated pixel samples
    finaloutputbuffer = NULL; // stores averaged pixel samples
	normalbuffer = NULL;
	materialbuffer = NULL;
    gpuHDRenv = NULL;
    cpuHDRenv = NULL;
    m_triNormals = NULL;
    gpuBVH = NULL;

    framenumber = 0;
    scalefactor = 1.2f;
    nocachedBVH = false;
	m_interval = 40;
	m_firsttime = true;

	m_windowSize = 15;
	m_variance_pos = 100;
	m_variance_col = 40;
	m_variance_dep = 100;

	mtlfile = "data/class2.mtl";
    scenefile = "data/class2.obj"; 
    HDRmapname = "data/Topanga_Forest_B_3k.hdr";
}
//! [1]

//! [3]
static const char *vertexShaderSource =
    "attribute highp vec4 posAttr;\n"
    "attribute lowp vec4 colAttr;\n"
    "varying lowp vec4 col;\n"
    "uniform highp mat4 matrix;\n"
    "void main() {\n"
    "   col = colAttr;\n"
    "   gl_Position = matrix * posAttr;\n"
    "}\n";

static const char *fragmentShaderSource =
    "varying lowp vec4 col;\n"
    "void main() {\n"
    "   gl_FragColor = col;\n"
    "}\n";
//! [3]

//! [4]
void TriangleWindow::initializeGL()
{
    initializeOpenGLFunctions();

    // create a CPU camera
    hostRendercam = new Camera();
    // initialise an interactive camera on the CPU side
    initCamera(width(), height());	
    interactiveCamera->buildRenderCamera(hostRendercam);


    std::cout << "create BVH file\n";
    // initialise all data needed to start rendering (BVH data, triangles, vertices)
    createBVH();

    initCUDAscenedata(); // copy scene data to the GPU, ready to be used by CUDA
    initHDR(); // initialise the HDR environment map

    createVBO(&m_vbo);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, width(), 0.0, height());
}
//! [4]

void TriangleWindow::ProfilerBegin() {
	QueryPerformanceFrequency(&m_tc);
	QueryPerformanceCounter(&m_t1);
}

void TriangleWindow::ProfilerEnd(long long int numRays) {
	double dff = m_tc.QuadPart;
	QueryPerformanceCounter(&m_t2);
	double s = (m_t2.QuadPart - m_t1.QuadPart) / dff;
	printf("render time is %.3fs\n", s);
	long long int rayPerSecond = numRays / s;
	printf("path per second %lld rays/s\n", rayPerSecond);
}

void TriangleWindow::slotWindowSizeChanged(int size){
	m_windowSize = size;
	buffer_reset = true;
}
void TriangleWindow::slotVariancPosChanged(double val){
	m_variance_pos = val;
	buffer_reset = true;
}

void TriangleWindow::slotVariancColChanged(double val){
	m_variance_col = val;
	buffer_reset = true;
}

void TriangleWindow::slotVariancDepChanged(double val){
	m_variance_dep = val;
	buffer_reset = true;
}

void TriangleWindow::createVBO(GLuint* vbo)
{
    //Create vertex buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    //Initialize VBO
    unsigned int size = width() * height() * sizeof(Vec3f);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //Register VBO with CUDA
    cudaGLRegisterBufferObject(*vbo);
}

// initialise HDR environment map
// from https://graphics.stanford.edu/wikis/cs148-11-summer/HDRIlluminator

void TriangleWindow::initHDR(){

    HDRImage HDRresult;
    const char* HDRfile = HDRmapname;

    if (HDRLoader::load(HDRfile, HDRresult)) 
        printf("HDR environment map loaded. Width: %d Height: %d\n", HDRresult.width, HDRresult.height);
    else{ 
        printf("HDR environment map not found\nAn HDR map is required as light source. Exiting now...\n"); 
        system("PAUSE");
        exit(0);
    }

    int HDRwidth = HDRresult.width;
    int HDRheight = HDRresult.height;
    cpuHDRenv = new Vec4f[HDRwidth * HDRheight];
    //_data = new RGBColor[width*height];

    for (int i = 0; i<HDRwidth; i++){
        for (int j = 0; j<HDRheight; j++){
            int idx = 3 * (HDRwidth*j + i);
            //int idx2 = width*(height-j-1)+i;
            int idx2 = HDRwidth*(j)+i;
            cpuHDRenv[idx2] = Vec4f(HDRresult.colors[idx], HDRresult.colors[idx + 1], HDRresult.colors[idx + 2], 0.0f);
        }
    }

    // copy HDR map to CUDA
    cudaMalloc(&gpuHDRenv, HDRwidth * HDRheight * sizeof(float4));
    cudaMemcpy(gpuHDRenv, cpuHDRenv, HDRwidth * HDRheight * sizeof(float4), cudaMemcpyHostToDevice);
}

void TriangleWindow::initCUDAscenedata()
{
    // allocate GPU memory for accumulation buffer
    cudaMalloc(&accumulatebuffer, width() * height() * sizeof(Vec3f));
    cudaMalloc(&normalbuffer, width() * height() * sizeof(Vec3f));
    cudaMalloc(&materialbuffer, width() * height() * sizeof(float));

    // allocate GPU memory for interactive camera
    cudaMalloc((void**)&cudaRendercam, sizeof(Camera));

    // allocate and copy scene databuffers to the GPU (BVH nodes, triangle vertices, triangle indices)
    cudaMalloc((void**)&cudaNodePtr, gpuBVH->getGpuNodesSize() * sizeof(float4));
    cudaMemcpy(cudaNodePtr, gpuBVH->getGpuNodes(), gpuBVH->getGpuNodesSize() * sizeof(float4), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cudaTriWoopPtr, gpuBVH->getGpuTriWoopSize() * sizeof(float4));
    cudaMemcpy(cudaTriWoopPtr, gpuBVH->getGpuTriWoop(), gpuBVH->getGpuTriWoopSize() * sizeof(float4), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cudaTriDebugPtr, gpuBVH->getDebugTriSize() * sizeof(float4));
    cudaMemcpy(cudaTriDebugPtr, gpuBVH->getDebugTri(), gpuBVH->getDebugTriSize() * sizeof(float4), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cudaTriUVPtr, gpuBVH->getUVTriSize() * sizeof(float4));
    cudaMemcpy(cudaTriUVPtr, gpuBVH->getUVTri(), gpuBVH->getUVTriSize() * sizeof(float4), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cudaTriIndicesPtr, gpuBVH->getGpuTriIndicesSize() * sizeof(S32));
    cudaMemcpy(cudaTriIndicesPtr, gpuBVH->getGpuTriIndices(), gpuBVH->getGpuTriIndicesSize() * sizeof(S32), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cudaMaterialsPtr, scene_info.materialNo * sizeof(MaterialCUDA));
    cudaMemcpy(cudaMaterialsPtr, scene_info.materials, scene_info.materialNo * sizeof(MaterialCUDA), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cudaTexturePtr, scene_info.textureNo * sizeof(TextureCUDA));
    cudaMemcpy(cudaTexturePtr, scene_info.textures, scene_info.textureNo * sizeof(TextureCUDA), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cudaTextureData, scene_info.textotalsize * sizeof(float4));
	for (int i = 0; i < scene_info.textureNo; i++)
	{
		cudaMemcpy(cudaTextureData + scene_info.textures[i].start_index, scene_info.textures[i].texels, scene_info.textures[i].height * scene_info.textures[i].width * sizeof(float4), cudaMemcpyHostToDevice);
	}

    std::cout << "Scene data copied to CUDA\n";
}

void TriangleWindow::createBVH(){
	load_material(mtlfile);
    load_object(scenefile);
    float maxi2 = processgeo();

    std::cout << "Scene geometry loaded and processed\n";

    // create arrays for the triangles and the vertices
    // Scene() constructor: Scene(const S32 numTris, const S32 numVerts, const Array<Triangle>& tris, const Array<Vec3f>& verts)

    Array<Scene::Triangle> tris;
    Array<Vec3f> verts;
    tris.clear();
    verts.clear();

    // convert Triangle to Scene::Triangle
    for (unsigned int i = 0; i < scene_info.trianglesNo; i++){
        Scene::Triangle newtri;
        newtri.verticeIndex = Vec3i(scene_info.triangles[i].v_idx1, scene_info.triangles[i].v_idx2, scene_info.triangles[i].v_idx3);
        newtri.normalIndex = Vec3i(scene_info.triangles[i].n_idx1, scene_info.triangles[i].n_idx2, scene_info.triangles[i].n_idx3);
		newtri.materialIndex = scene_info.triangles[i].m_idx;
        newtri.uvIndex = Vec3i(scene_info.triangles[i].uv_idx1, scene_info.triangles[i].uv_idx2, scene_info.triangles[i].uv_idx3);
        tris.add(newtri);
    }

    // fill up Array of vertices
    for (unsigned int i = 0; i < scene_info.verticesNo; i++) { 
        verts.add(Vec3f(scene_info.vertices[i].x, scene_info.vertices[i].y, scene_info.vertices[i].z)); 
    }

    std::cout << "Building a new scene\n";
    Scene* scene = new Scene(scene_info.trianglesNo, scene_info.verticesNo, tris, verts);

    std::cout << "Building BVH with spatial splits\n";
    // create a default platform
    Platform defaultplatform;
    BVH::BuildParams defaultparams;
    BVH::Stats stats;
    BVH myBVH(scene, defaultplatform, defaultparams);

    std::cout << "Building CudaBVH\n";
    // create CUDA friendly BVH datastructure
    gpuBVH = new CudaBVH(myBVH, BVHLayout_Compact);  // Fermi BVH layout = compact. BVH layout for Kepler kernel Compact2
    std::cout << "CudaBVH successfully created\n";
}

void TriangleWindow::deleteCudaAndCpuMemory()
{
    // free CUDA memory
    cudaFree(cudaNodePtr);
    cudaFree(cudaTriWoopPtr);
    cudaFree(cudaTriDebugPtr);
    cudaFree(cudaTriUVPtr);
    cudaFree(cudaTriNormalPtr);
    cudaFree(cudaTriIndicesPtr);
	cudaFree(cudaMaterialsPtr);
	cudaFree(cudaTexturePtr);
	cudaFree(cudaTextureData);
    cudaFree(cudaRendercam);
    cudaFree(accumulatebuffer);
    cudaFree(normalbuffer);
    cudaFree(materialbuffer);
    cudaFree(finaloutputbuffer);
    cudaFree(gpuHDRenv);

    delete hostRendercam;
    delete interactiveCamera;
    delete cpuHDRenv;
    delete gpuBVH;
}

//! [5]
void TriangleWindow::paintGL()
{
	
    const qreal retinaScale = devicePixelRatio();
    glViewport(0, 0, width() * retinaScale, height() * retinaScale);

    // if camera has moved, reset the accumulation buffer
    if (buffer_reset){
		cudaMemset(accumulatebuffer, 1, width() * height() * sizeof(Vec3f)); 
		cudaMemset(materialbuffer, 1, width() * height() * sizeof(float)); 
		cudaMemset(normalbuffer, 1, width() * height() * sizeof(Vec3f)); 
		framenumber = 0; 
	}

	if(m_firsttime || buffer_reset)
	{
		m_firsttime = false;
		ProfilerBegin();
	}

    buffer_reset = false;
    framenumber++;

    // build a new camera for each frame on the CPU
    interactiveCamera->buildRenderCamera(hostRendercam);

    // copy the CPU camera to a GPU camera
    cudaMemcpy(cudaRendercam, hostRendercam, sizeof(Camera), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    cudaGLMapBufferObject((void**)&finaloutputbuffer, m_vbo); // maps a buffer object for access by CUDA

    glClear(GL_COLOR_BUFFER_BIT); //clear all pixels

    // calculate a new seed for the random number generator, based on the framenumber
    unsigned int hashedframes = WangHash(framenumber);

    // gateway from host to CUDA, passes all data needed to render frame (triangles, BVH tree, camera) to CUDA for execution
    cudaRender(cudaNodePtr, cudaTriWoopPtr, cudaTriDebugPtr, cudaTriIndicesPtr,cudaMaterialsPtr, cudaTexturePtr, cudaTextureData, cudaTriUVPtr, finaloutputbuffer,
        accumulatebuffer, normalbuffer, materialbuffer, gpuHDRenv, 
		framenumber, hashedframes, gpuBVH->getGpuNodesSize(), gpuBVH->getLeafnodeCount(), gpuBVH->getTriCount(), cudaRendercam, width(), height(), m_windowSize, m_variance_pos, m_variance_col, m_variance_dep );

    cudaThreadSynchronize();
    cudaGLUnmapBufferObject(m_vbo);
    //glFlush();
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glVertexPointer(2, GL_FLOAT, 12, 0);
    glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_POINTS, 0, width() * height());
    glDisableClientState(GL_VERTEX_ARRAY);

    ++m_frame;
    update();

	if(framenumber % m_interval == 0){
		const int depth = 4;
		const int samp = 1;
		long long int numRays = (long long)width() * (long long)height() * (long long)(depth * samp * framenumber);
		ProfilerEnd(numRays);
	}
	
}
//! [5]

void TriangleWindow::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
	setFocus();
}

void TriangleWindow::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    if (event->buttons() & Qt::LeftButton) {
        if (dx != 0 || dy != 0)
        {
            interactiveCamera->changeYaw(-dx * 0.01);
            interactiveCamera->changePitch(dy * 0.01);
            buffer_reset = true;
        }
        //rotateBy(8 * dy, 8 * dx, 0);
    } else if (event->buttons() & Qt::RightButton) {
        if (dy != 0)
        {
            interactiveCamera->changeRadius(-dx * 0.01);
            buffer_reset = true;
        }
        //rotateBy(8 * dy, 0, 8 * dx);
    }
    lastPos = event->pos();
}

void TriangleWindow::mouseReleaseEvent(QMouseEvent * /* event */)
{
    //emit clicked();
}

void TriangleWindow::keyPressEvent(QKeyEvent *event)
{
	if(event->key() == Qt::Key_W) 
    {
		interactiveCamera->goForward(200);
		buffer_reset = true;
    }
	else if(event->key() == Qt::Key_A) 
    {
		interactiveCamera->strafe(-200);
		buffer_reset = true;
    }
	else if(event->key() == Qt::Key_S) 
    {
		interactiveCamera->goForward(-200);
		buffer_reset = true;
    }
	else if(event->key() == Qt::Key_D) 
    {
		interactiveCamera->strafe(200);
		buffer_reset = true;
    }
	else if(event->key() == Qt::Key_Q) 
    {
		interactiveCamera->changeAltitude(100);
		buffer_reset = true;
    }
	else if(event->key() == Qt::Key_E) 
    {
		interactiveCamera->changeAltitude(-100);
		buffer_reset = true;
    }
}

TriangleWindow::~TriangleWindow()
{
	deleteCudaAndCpuMemory();
}
