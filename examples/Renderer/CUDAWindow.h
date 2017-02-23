#include <QtGui/QGuiApplication>
#include <QtWidgets/QOpenGLWidget>
#include <QOpenGLFunctions>

#include <Windows.h>

#include "linear_math.h"
#include "Util.h"
#include "MMaterial.h"


class Camera;
class CudaBVH;

//! [1]
class TriangleWindow : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT
public:
    TriangleWindow(QWidget *parent = 0);
    void createVBO(GLuint* vbo);
    void initCUDAscenedata();
    void deleteCudaAndCpuMemory();
    void createBVH();
    void loadBVHfromCache(FILE* BVHcachefile, const std::string BVHcacheFilename);
    void writeBVHcachefile(FILE* BVHcachefile, const std::string BVHcacheFilename);
    void initHDR();

	void ProfilerBegin();
	void ProfilerEnd(long long int numRays);

protected:
    void initializeGL() Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
    //void resizeGL(int width, int height) Q_DECL_OVERRIDE;
    void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    void mouseReleaseEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
	void keyPressEvent(QKeyEvent *) Q_DECL_OVERRIDE;

public slots:
	void slotWindowSizeChanged(int size);
	void slotVariancChanged(double val);

private:
    GLuint m_posAttr;
    GLuint m_colAttr;
    GLuint m_matrixUniform;

    GLuint m_vbo;
    Vec4i* cpuNodePtr;
    Vec4i* cpuTriWoopPtr;
    Vec4i* cpuTriDebugPtr;
    Vec4f* cpuTriNormalPtr;
    S32*   cpuTriIndicesPtr;

    float4* cudaNodePtr;
    float4* cudaTriWoopPtr;
    float4* cudaTriDebugPtr;
    float4* cudaTriNormalPtr;
    S32*    cudaTriIndicesPtr;
	MaterialCUDA* cudaMaterialsPtr;

    Camera* cudaRendercam;
    Camera* hostRendercam;
    Vec3f* accumulatebuffer; // image buffer storing accumulated pixel samples
    Vec3f* normalbuffer;	// stores ray intersect normal per pixel samples
	Vec3f* finaloutputbuffer; // stores averaged pixel samples
    float* depthbuffer; // stores ray intersect depth per pixel samples
    float* eyecosdepthbuffer; // stores ray intersect depth with eye ray cos per pixel samples
    int* materialbuffer; // stores ray intersect material per pixel samples
    float4* gpuHDRenv;
    Vec4f* cpuHDRenv;
    Vec4f* m_triNormals;
    CudaBVH* gpuBVH;

    int framenumber;
    int nodeSize;
    int leafnode_count;
    int triangle_count;
    int triWoopSize;
    int triDebugSize;
    int triIndicesSize;
    float scalefactor;
    bool nocachedBVH;
	
    const char* mtlfile;
    const char* scenefile; 
    const char* HDRmapname;

    int				m_frame;
	int				m_interval;
	bool			m_firsttime;
	LARGE_INTEGER	m_t1, m_t2, m_tc;

    QPoint lastPos;

	int	m_windowSize;
	float m_variance;
};