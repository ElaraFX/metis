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
    cpuNodePtr = NULL;
    cpuTriWoopPtr = NULL;
    cpuTriDebugPtr = NULL;
    cpuTriNormalPtr = NULL;
    cpuTriIndicesPtr = NULL;

    cudaNodePtr = NULL;
    cudaTriWoopPtr = NULL;
    cudaTriDebugPtr = NULL;
    cudaTriNormalPtr = NULL;
    cudaTriIndicesPtr = NULL;
	cudaMaterialsPtr = NULL;

    cudaRendercam = NULL;
    hostRendercam = NULL;
    accumulatebuffer = NULL; // image buffer storing accumulated pixel samples
    finaloutputbuffer = NULL; // stores averaged pixel samples
    gpuHDRenv = NULL;
    cpuHDRenv = NULL;
    m_triNormals = NULL;
    gpuBVH = NULL;

    framenumber = 0;
    nodeSize = 0;
    leafnode_count = 0;
    triangle_count = 0;
    triWoopSize = 0;
    triDebugSize = 0;
    triIndicesSize = 0;
    scalefactor = 1.2f;
    nocachedBVH = false;
	m_interval = 64;
	m_firsttime = true;

	m_windowSize = 10;
	m_variance = 1000;

	mtlfile = "data/teapot1.mtl";
    scenefile = "data/teapot1.obj"; 
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

    std::string BVHcacheFilename(scenefile);
    BVHcacheFilename += ".bvh";  

    FILE* BVHcachefile = fopen(BVHcacheFilename.c_str(), "rb");
    if (!BVHcachefile){ nocachedBVH = true; }

    //if (true){ // overrule cache
	if (nocachedBVH){
        std::cout << "No cached BVH file available\nCreating new BVH...\n";
        // initialise all data needed to start rendering (BVH data, triangles, vertices)
        createBVH();
        // store the BVH in a file
        writeBVHcachefile(BVHcachefile, BVHcacheFilename);
    }

    else { // cached BVH available
        std::cout << "Cached BVH available\nReading " << BVHcacheFilename << "...\n";
        loadBVHfromCache(BVHcachefile, BVHcacheFilename); 
    }

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
void TriangleWindow::slotVariancChanged(double val){
	m_variance = val;
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

void TriangleWindow::loadBVHfromCache(FILE* BVHcachefile, const std::string BVHcacheFilename)
{
    if (1 != fread(&nodeSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
    if (1 != fread(&triangle_count, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
    if (1 != fread(&leafnode_count, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
    if (1 != fread(&triWoopSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
    if (1 != fread(&triDebugSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
    if (1 != fread(&triIndicesSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&materialNo, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";

    std::cout << "Number of nodes: " << nodeSize << "\n";
    std::cout << "Number of triangles: " << triangle_count << "\n";
	std::cout << "Number of Materials: " << materialNo << "\n";
    std::cout << "Number of BVH leafnodes: " << leafnode_count << "\n";

    cpuNodePtr = (Vec4i*)malloc(nodeSize * sizeof(Vec4i));
    cpuTriWoopPtr = (Vec4i*)malloc(triWoopSize * sizeof(Vec4i));
    cpuTriDebugPtr = (Vec4i*)malloc(triDebugSize * sizeof(Vec4i));
    cpuTriIndicesPtr = (S32*)malloc(triIndicesSize * sizeof(S32));
	materials = (MaterialCUDA*)malloc(materialNo * sizeof(MaterialCUDA));

    if (nodeSize != fread(cpuNodePtr, sizeof(Vec4i), nodeSize, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
    if (triWoopSize != fread(cpuTriWoopPtr, sizeof(Vec4i), triWoopSize, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
    if (triDebugSize != fread(cpuTriDebugPtr, sizeof(Vec4i), triDebugSize, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
    if (triIndicesSize != fread(cpuTriIndicesPtr, sizeof(S32), triIndicesSize, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (materialNo != fread(materials, sizeof(MaterialCUDA), materialNo, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";

    fclose(BVHcachefile);
    std::cout << "Successfully loaded BVH from cache file!\n";
}

void TriangleWindow::writeBVHcachefile(FILE* BVHcachefile, const std::string BVHcacheFilename){

    BVHcachefile = fopen(BVHcacheFilename.c_str(), "wb");
    if (!BVHcachefile) std::cout << "Error opening BVH cache file!\n";
    if (1 != fwrite(&nodeSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
    if (1 != fwrite(&triangle_count, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
    if (1 != fwrite(&leafnode_count, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
    if (1 != fwrite(&triWoopSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
    if (1 != fwrite(&triDebugSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
    if (1 != fwrite(&triIndicesSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&materialNo, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";

    if (nodeSize != fwrite(cpuNodePtr, sizeof(Vec4i), nodeSize, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
    if (triWoopSize != fwrite(cpuTriWoopPtr, sizeof(Vec4i), triWoopSize, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
    if (triDebugSize != fwrite(cpuTriDebugPtr, sizeof(Vec4i), triDebugSize, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
    if (triIndicesSize != fwrite(cpuTriIndicesPtr, sizeof(S32), triIndicesSize, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (materialNo != fwrite(materials, sizeof(MaterialCUDA), materialNo, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";

    fclose(BVHcachefile);
    std::cout << "Successfully created BVH cache file!\n";
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

    // allocate GPU memory for interactive camera
    cudaMalloc((void**)&cudaRendercam, sizeof(Camera));

    // allocate and copy scene databuffers to the GPU (BVH nodes, triangle vertices, triangle indices)
    cudaMalloc((void**)&cudaNodePtr, nodeSize * sizeof(float4));
    cudaMemcpy(cudaNodePtr, cpuNodePtr, nodeSize * sizeof(float4), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cudaTriWoopPtr, triWoopSize * sizeof(float4));
    cudaMemcpy(cudaTriWoopPtr, cpuTriWoopPtr, triWoopSize * sizeof(float4), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cudaTriDebugPtr, triDebugSize * sizeof(float4));
    cudaMemcpy(cudaTriDebugPtr, cpuTriDebugPtr, triDebugSize * sizeof(float4), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cudaTriIndicesPtr, triIndicesSize * sizeof(S32));
    cudaMemcpy(cudaTriIndicesPtr, cpuTriIndicesPtr, triIndicesSize * sizeof(S32), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cudaMaterialsPtr, materialNo * sizeof(MaterialCUDA));
    cudaMemcpy(cudaMaterialsPtr, materials, materialNo * sizeof(MaterialCUDA), cudaMemcpyHostToDevice);

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
    for (unsigned int i = 0; i < trianglesNo; i++){
        Scene::Triangle newtri;
        newtri.verticeIndex = Vec3i(triangles[i].v_idx1, triangles[i].v_idx2, triangles[i].v_idx3);
        newtri.normalIndex = Vec3i(triangles[i].n_idx1, triangles[i].n_idx2, triangles[i].n_idx3);
		newtri.materialIndex = triangles[i].m_idx;
        tris.add(newtri);
    }

    // fill up Array of vertices
    for (unsigned int i = 0; i < verticesNo; i++) { 
        verts.add(Vec3f(vertices[i].x, vertices[i].y, vertices[i].z)); 
    }

    std::cout << "Building a new scene\n";
    Scene* scene = new Scene(trianglesNo, verticesNo, tris, verts);

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

    std::cout << "Hi Sam!  How you doin'?" << std::endl;

    cpuNodePtr = gpuBVH->getGpuNodes();
    cpuTriWoopPtr = gpuBVH->getGpuTriWoop();
    cpuTriDebugPtr = gpuBVH->getDebugTri();
    cpuTriIndicesPtr = gpuBVH->getGpuTriIndices();
    cpuTriNormalPtr = m_triNormals;

    nodeSize = gpuBVH->getGpuNodesSize();
    triWoopSize = gpuBVH->getGpuTriWoopSize();
    triDebugSize = gpuBVH->getDebugTriSize();
    triIndicesSize = gpuBVH->getGpuTriIndicesSize();
    leafnode_count = gpuBVH->getLeafnodeCount();
    triangle_count = gpuBVH->getTriCount();
}

void TriangleWindow::deleteCudaAndCpuMemory()
{
    // free CUDA memory
    cudaFree(cudaNodePtr);
    cudaFree(cudaTriWoopPtr);
    cudaFree(cudaTriDebugPtr);
    cudaFree(cudaTriNormalPtr);
    cudaFree(cudaTriIndicesPtr);
	cudaFree(cudaMaterialsPtr);
    cudaFree(cudaRendercam);
    cudaFree(accumulatebuffer);
    cudaFree(finaloutputbuffer);
    cudaFree(gpuHDRenv);

    // release CPU memory
    free(cpuNodePtr);
    free(cpuTriWoopPtr);
    free(cpuTriDebugPtr);
    free(cpuTriNormalPtr);
    free(cpuTriIndicesPtr);

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
    cudaRender(cudaNodePtr, cudaTriWoopPtr, cudaTriDebugPtr, cudaTriIndicesPtr,cudaMaterialsPtr, finaloutputbuffer,
        accumulatebuffer, gpuHDRenv, framenumber, hashedframes, nodeSize, leafnode_count, triangle_count, cudaRendercam, width(), height());

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