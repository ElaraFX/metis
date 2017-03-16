#include <QtGui/QGuiApplication>
#include <QtWidgets/QOpenGLWidget>
#include <QOpenGLFunctions>

#include <Windows.h>

#include "linear_math.h"
#include "Util.h"
#include "MMaterial.h"
#include "RenderData.h"


struct Camera;
class CudaBVH;
struct TextureCUDA;

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
    //void loadBVHfromCache(FILE* BVHcachefile, const std::string BVHcacheFilename);
    //void writeBVHcachefile(FILE* BVHcachefile, const std::string BVHcacheFilename);
    void initHDR();

	void ProfilerBegin();
	void ProfilerEnd(long long int numRays);
	~TriangleWindow();

protected:
    void initializeGL() Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
    //void resizeGL(int width, int height) Q_DECL_OVERRIDE;
    void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    void mouseReleaseEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
	void keyPressEvent(QKeyEvent *);

public slots:
	void slotWindowSizeChanged(int size);
	void slotVariancPosChanged(double val);
	void slotVariancColChanged(double val);
	void slotVariancDepChanged(double val);

private:
    GLuint m_posAttr;
    GLuint m_colAttr;
    GLuint m_matrixUniform;

    GLuint m_vbo;

	gpuData* m_gpu_data;
	gpuData* m_host_gpu_data;
    Camera* hostRendercam;
	Vec3f* finaloutputbuffer; // stores averaged pixel samples
    float4* gpuHDRenv;
    Vec4f* cpuHDRenv;
    Vec4f* m_triNormals;
    CudaBVH* gpuBVH;

    int framenumber;
    int nodeSize;
	
    const char* mtlfile;
    const char* scenefile; 
    const char* HDRmapname;

	int				m_interval;
	bool			m_firsttime;
	LARGE_INTEGER	m_t1, m_t2, m_tc;

    QPoint lastPos;

	controlParam cp;
};