#include <QtGui/QGuiApplication>
#include <QtWidgets/QOpenGLWidget>
#include <QOpenGLFunctions>

#include "linear_math.h"
#include "Util.h"

class Camera;
class CudaBVH;

//! [1]
class TriangleWindow : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    TriangleWindow(QWidget *parent = 0);
    void createVBO(GLuint* vbo);
    void initCUDAscenedata();
    void deleteCudaAndCpuMemory();
    void createBVH();
    void loadBVHfromCache(FILE* BVHcachefile, const std::string BVHcacheFilename);
    void writeBVHcachefile(FILE* BVHcachefile, const std::string BVHcacheFilename);
    void initHDR();


protected:
    void initializeGL() Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
    //void resizeGL(int width, int height) Q_DECL_OVERRIDE;
    void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    void mouseReleaseEvent(QMouseEvent *event) Q_DECL_OVERRIDE;

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

    Camera* cudaRendercam;
    Camera* hostRendercam;
    Vec3f* accumulatebuffer; // image buffer storing accumulated pixel samples
    Vec3f* finaloutputbuffer; // stores averaged pixel samples
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

    const char* scenefile; 
    const char* HDRmapname;

    int m_frame;

    QPoint lastPos;
};