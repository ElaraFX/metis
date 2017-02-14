#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cfloat>
#include <string.h>
#include <assert.h>

#include "linear_math.h"
#include "Geometry.h"
#include "SceneLoader.h"


using std::string;

unsigned verticesNo = 0;
unsigned normalsNo = 0;
unsigned trianglesNo = 0;
Vertex* vertices = NULL;   // vertex list
Vec3f* normals = NULL;
Triangle* triangles = NULL;  // triangle list

 
struct face {                  
	std::vector<int> vertex;
	std::vector<int> texture;
	std::vector<int> normal;
};

std::vector<face> faces;

namespace enums {
	enum ColorComponent {
		Red = 0,
		Green = 1,
		Blue = 2
	};
}

using namespace enums;

// Rescale input objects to have this size...
const float MaxCoordAfterRescale = 1.2f;

// if some file cannot be found, panic and exit
void panic(const char *fmt, ...)
{
	static char message[131072];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(message, sizeof message, fmt, ap);
	printf(message); fflush(stdout);
	va_end(ap);

	exit(1);
}

struct TriangleMesh
{
	std::vector<Vec3f> verts;
    std::vector<Vec3f> nors;
	std::vector<Vec3i> faceVertIndexs;
    std::vector<Vec3i> faceNorIndexs;
	Vec3f bounding_box[2];   // mesh bounding box
};

void load_object(const char *filename)
{
	std::cout << "Loading object..." << std::endl;
	const char* edot = strrchr(filename, '.');
	if (edot) {
		edot++;
		////////////////////
		// basic OBJ loader
		////////////////////

		// the OBJ loader code based is an improved version of the obj code in  
		// http://www.keithlantz.net/2013/04/kd-tree-construction-using-the-surface-area-heuristic-stack-based-traversal-and-the-hyperplane-separation-theorem/
		// this code triangulates models with quads, n-gons and triangle fans

		if (!strcmp(edot, "obj")) {

			std::cout << "loading OBJ model: " << filename;
			std::string filenamestring = filename;
			std::ifstream in(filenamestring.c_str());
			std::cout << filenamestring << "\n";

			if (!in.good())
			{
				std::cout << "ERROR: loading obj:(" << filename << ") file not found or not good" << "\n";
				system("PAUSE");
				exit(0);
			}

			Vertex *pCurrentVertex = NULL;
            Vec3f *pCurrentNormal = NULL;
			Triangle *pCurrentTriangle = NULL;
			unsigned totalVertices, totalNormals, totalTriangles = 0;
			TriangleMesh mesh;
					
			std::ifstream ifs(filenamestring.c_str(), std::ifstream::in);

			if (!ifs.good())
			{
			std::cout << "ERROR: loading obj:(" << filename << ") file not found or not good" << "\n";
			system("PAUSE");
			exit(0);
			}


			std::string line, key;
			while (!ifs.eof() && std::getline(ifs, line)) {
			    key = "";
			    std::stringstream stringstream(line);
			    stringstream >> key >> std::ws;

			    // if (sscanf(buffer, "v %f %f %f", &f1, &f2, &f3) == 3){
			    // mesh.verts.push_back(Vec3f(f1, f2, f3));

			    if (key == "v") { // vertex	
				    float x, y, z;
				    while (!stringstream.eof()) {
					    stringstream >> x >> std::ws >> y >> std::ws >> z >> std::ws;
					    mesh.verts.push_back(Vec3f(x, y, z));
				    }
			    }
			    else if (key == "vp") { // parameter
				    float x;
				    // std::vector<float> tempparameters;
				    while (!stringstream.eof()) {
					    stringstream >> x >> std::ws;
					    // tempparameters.push_back(x);
				    }
				    //parameters.push_back(tempparameters);
			    }
			    else if (key == "vt") { // texture coordinate
				    float x;
				    // std::vector<float> temptexcoords;
				    while (!stringstream.eof()) {
					    stringstream >> x >> std::ws;
					    // temptexcoords.push_back(x);
				    }
				    //texcoords.push_back(temptexcoords);
			    }
			    else if (key == "vn") { // normal
				    float x, y, z;
				    // std::vector<float> tempnormals;
				    while (!stringstream.eof()) {
                        stringstream >> x >> std::ws >> y >> std::ws >> z >> std::ws;
                        mesh.nors.push_back(Vec3f(x, y, z));
					    //	tempnormals.push_back(x);
				    }
				    //tempnormal.normalize();
				    //normals.push_back(tempnormals);
			    }

			    else if (key == "f") { // face
				    face f;
				    int v, t, n;
				    while (!stringstream.eof()) {
					    stringstream >> v >> std::ws;
					    f.vertex.push_back(v); // v - 1
					    if (stringstream.peek() == '/') {
						    stringstream.get();
						    if (stringstream.peek() == '/') {
							    stringstream.get();
							    stringstream >> n >> std::ws;
							    f.normal.push_back(n - 1);
						    }
						    else {
							    stringstream >> t >> std::ws;
							    f.texture.push_back(t - 1);
							    if (stringstream.peek() == '/') {
								    stringstream.get();
								    stringstream >> n >> std::ws;
								    f.normal.push_back(n - 1);
							    }
						    }
					    }
				    }

			        int numtriangles = f.vertex.size() - 2; // 1 triangle if 3 vertices, 2 if 4 etc

			        for (int i = 0; i < numtriangles; i++){  // first vertex remains the same for all triangles in a triangle fan
			        mesh.faceVertIndexs.push_back(Vec3i(f.vertex[0], f.vertex[i + 1], f.vertex[i + 2]));
                    mesh.faceNorIndexs.push_back(Vec3i(f.normal[0], f.normal[i + 1], f.normal[i + 2]));
			    }

			    //while (stream >> v_extra) {
			    //	v2 = v3;
			    //	v3 = v_extra;
			    //	mesh.faces.push_back(Vec3i(v1, v2, v3));
			    //}
			    }
			    else {
			    }
					
			} // end of while loop

			totalVertices = mesh.verts.size();
            totalNormals = mesh.nors.size();
			totalTriangles = mesh.faceVertIndexs.size();

			vertices = (Vertex *)malloc(totalVertices*sizeof(Vertex));
			verticesNo = totalVertices;
			pCurrentVertex = vertices;

            normals = (Vec3f*)malloc(totalNormals*sizeof(Vec3f));
            normalsNo = totalNormals;
            pCurrentNormal = normals;

			triangles = (Triangle *)malloc(totalTriangles*sizeof(Triangle));
			trianglesNo = totalTriangles;
			pCurrentTriangle = triangles;

			std::cout << "total vertices: " << totalVertices << "\n";
            std::cout << "total normals: " << totalNormals << "\n";
			std::cout << "total triangles: " << totalTriangles << "\n";

			for (int i = 0; i < totalVertices; i++){
				Vec3f currentvert = mesh.verts[i];
				pCurrentVertex->x = currentvert.x;
				pCurrentVertex->y = currentvert.y;
				pCurrentVertex->z = currentvert.z;
				//pCurrentVertex->_normal.x = 0.f; // not used for now, normals are computed on-the-fly by GPU
				//pCurrentVertex->_normal.y = 0.f; // not used 
				//pCurrentVertex->_normal.z = 0.f; // not used 

				pCurrentVertex++;
			}

			std::cout << "Vertices loaded\n";

            for (int i = 0; i < totalNormals; i++){
                Vec3f currentnor = mesh.nors[i];
                pCurrentNormal->x = currentnor.x;
                pCurrentNormal->y = currentnor.y;
                pCurrentNormal->z = currentnor.z;

                pCurrentNormal++;
            }

            std::cout << "Normals loaded\n";

//			for (int i = 0; i < totalTriangles; i++)
			while(totalTriangles)
			{ 
				totalTriangles--;
				
				Vec3i currentfaceinds = mesh.faceVertIndexs[totalTriangles];
                Vec3i currentnorminds = mesh.faceNorIndexs[totalTriangles];

				pCurrentTriangle->v_idx1 = currentfaceinds.x - 1;
				pCurrentTriangle->v_idx2 = currentfaceinds.y - 1;
				pCurrentTriangle->v_idx3 = currentfaceinds.z - 1;

                pCurrentTriangle->n_idx1 = currentnorminds.x;
                pCurrentTriangle->n_idx2 = currentnorminds.y;
                pCurrentTriangle->n_idx3 = currentnorminds.z;

				Vertex *vertexA = &vertices[currentfaceinds.x - 1];
				Vertex *vertexB = &vertices[currentfaceinds.y - 1];
				Vertex *vertexC = &vertices[currentfaceinds.z - 1];


				pCurrentTriangle->_center = Vec3f(
					(vertexA->x + vertexB->x + vertexC->x) / 3.0f,
					(vertexA->y + vertexB->y + vertexC->y) / 3.0f,
					(vertexA->z + vertexB->z + vertexC->z) / 3.0f);

				pCurrentTriangle++;
			}
		}

		else
			panic("Unknown extension (only .ply and .obj accepted)");
	}
	else
		panic("No extension in filename (only .ply accepted)");

	std::cout << "Vertices:  " << verticesNo << std::endl;
    std::cout << "Normals:  " << normalsNo << std::endl;
	std::cout << "Triangles: " << trianglesNo << std::endl;

}

	/////////////////////////
	// SCENE GEOMETRY PROCESSING
	/////////////////////////

float processgeo(){

	// Center scene at world's center

	Vec3f minp(FLT_MAX, FLT_MAX, FLT_MAX);
	Vec3f maxp(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	// calculate min and max bounds of scene
	// loop over all triangles in scene, grow minp and maxp
	for (unsigned i = 0; i<trianglesNo; i++) {

		minp = min3f(minp, vertices[triangles[i].v_idx1]);
		minp = min3f(minp, vertices[triangles[i].v_idx2]);
		minp = min3f(minp, vertices[triangles[i].v_idx3]);

		maxp = max3f(maxp, vertices[triangles[i].v_idx1]);
		maxp = max3f(maxp, vertices[triangles[i].v_idx2]);
		maxp = max3f(maxp, vertices[triangles[i].v_idx3]);
	}

	// scene bounding box center before scaling and translating
	Vec3f origCenter = Vec3f(
		(maxp.x + minp.x) * 0.5,
		(maxp.y + minp.y) * 0.5,
		(maxp.z + minp.z) * 0.5);

	minp -= origCenter;
	maxp -= origCenter;

	// Scale scene so max(abs x,y,z coordinates) = MaxCoordAfterRescale

	float maxi = 0;
	maxi = std::max(maxi, (float)fabs(minp.x));
	maxi = std::max(maxi, (float)fabs(minp.y));
	maxi = std::max(maxi, (float)fabs(minp.z));
	maxi = std::max(maxi, (float)fabs(maxp.x));
	maxi = std::max(maxi, (float)fabs(maxp.y));
	maxi = std::max(maxi, (float)fabs(maxp.z));

	std::cout << "Scaling factor: " << (MaxCoordAfterRescale / maxi) << "\n";
	std::cout << "Center origin: " << origCenter.x << " " << origCenter.y << " " << origCenter.z << "\n";

	std::cout << "\nCentering and scaling vertices..." << std::endl;
	for (unsigned i = 0; i<verticesNo; i++) {
		vertices[i] -= origCenter;
		//vertices[i].y += origCenter.y;
		//vertices[i] *= (MaxCoordAfterRescale / maxi);
		vertices[i] *= 20; // 0.25
	}

	return MaxCoordAfterRescale;
}