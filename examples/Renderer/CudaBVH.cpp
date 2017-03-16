#include "CudaBVH.h"
#include "SceneLoader.h"  // required for triangles and vertices


static int woopcount = 0;  // counts Woopified triangles

CudaBVH::CudaBVH(const BVH& bvh, BVHLayout layout)
	: m_layout(layout)
{
	FW_ASSERT(layout >= 0 && layout < BVHLayout_Max);
	m_gpuNodes = NULL;
	m_gpuTriNormal = NULL;
	m_debugTri = NULL;
	m_UVTri = NULL;
	m_gpuTriIndices = NULL;

	if (layout == BVHLayout_Compact)
	{
		createCompact(bvh, 1);
		return;
	}

	if (layout == BVHLayout_Compact2)
	{
		createCompact(bvh, 16);
		return;
	}
}

CudaBVH::~CudaBVH(void)
{
	free(m_gpuNodes);
    free(m_gpuTriNormal);
    free(m_debugTri);
    free(m_UVTri);
    free(m_gpuTriIndices);
}

namespace detail
{
struct StackEntry
{
    const BVHNode*  node;
    S32             idx;

    StackEntry(const BVHNode* n = NULL, int i = 0) : node(n), idx(i) {}
};
}

void CudaBVH::createCompact(const BVH& bvh, int nodeOffsetSizeDiv)
{
    using namespace detail; // for StackEntry

	int leafcount = 0; // counts leafnodes

	// construct and initialize data arrays which will be copied to CudaBVH buffers (last part of this function). 

	Array<Vec4i> nodeData(NULL, 4); 
	Array<Vec4i> triNormalData;
	Array<Vec4i> triDebugData; // array for regular (non-woop) triangles
	Array<Vec4f> triUVData;
	Array<S32> triIndexData;

	// construct a stack (array of stack entries) to help in filling the data arrays
	Array<StackEntry> stack(StackEntry(bvh.getRoot(), 0)); // initialise stack to rootnode

	while (stack.getSize()) // while stack is not empty
	{
		StackEntry e = stack.removeLast(); // pop the stack
		FW_ASSERT(e.node->getNumChildNodes() == 2);
		const AABB* cbox[2];   
		int cidx[2]; // stores indices to both children

		// Process children.

		// for each child in entry e
		for (int i = 0; i < 2; i++)
		{
			const BVHNode* child = e.node->getChildNode(i); // current childnode
			cbox[i] = &child->m_bounds; // current child's AABB

			////////////////////////////
			/// INNER NODE
			//////////////////////////////

			// Inner node => push to stack.

			if (!child->isLeaf()) // no leaf, thus an inner node
			{   // compute childindex
				cidx[i] = nodeData.getNumBytes() / nodeOffsetSizeDiv; // nodeOffsetSizeDiv is 1 for Fermi kernel, 16 for Kepler kernel		
				
				// push the current child on the stack
				stack.add(StackEntry(child, nodeData.getSize()));   
				nodeData.add(NULL, 4); /// adds 4 * Vec4i per inner node or 4 * 16 bytes/Vec4i = 64 bytes of empty data per inner node
				continue; // process remaining childnode (if any)
			}

			//////////////////////
			/// LEAF NODE
			/////////////////////

			// Leaf => append triangles.

			const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child);
			
			// index of a leafnode is a negative number, hence the ~
			cidx[i] = ~triNormalData.getSize();  // leafs must be stored as negative (bitwise complement) in order to be recognised by pathtracer as a leaf
		
			// for each triangle in leaf, range of triangle index j from m_lo to m_hi 
			for (int j = leaf->m_lo; j < leaf->m_hi; j++) 
			{
				// transform the triangle's vertices to Woop triangle (simple transform to right angled triangle, see paper by Sven Woop)
				woopifyTri(bvh, j);  /// j is de triangle index in triIndex array
				
				if (m_woop[0].x == 0.0f) m_woop[0].x = 0.0f;  // avoid degenerate coordinates
				// add the transformed woop triangle to triWoopData
				
				triNormalData.add((Vec4i*)m_normaltri, 3);  
				
				triDebugData.add((Vec4i*)m_debugtri, 3);  
				
				triUVData.add((Vec4f*)m_uvtri, 3);  

				// add tri index for current triangle to triIndexData	
				triIndexData.add(bvh.getTriIndices()[j]); 
				triIndexData.add(m_materialIndex); // zero padding because CUDA kernel uses same index for vertex array (3 vertices per triangle)
				triIndexData.add(0); // and array of triangle indices
			}

			// Leaf node terminator to indicate end of leaf, stores hexadecimal value 0x80000000 (= 2147483648 in decimal)
			triNormalData.add(0x80000000); // leafnode terminator code indicates the last triangle of the leaf node
			triDebugData.add(0x80000000); 
			triUVData.add(0x80000000);
			
			// add extra zero to triangle indices array to indicate end of leaf
			triIndexData.add(0);  // terminates triIndexdata for current leaf

			leafcount++;
		}

		// Write entry for current node.  
		/// 4 Vec4i per node (according to compact bvh node layout)
		Vec4i* dst = nodeData.getPtr(e.idx);
		///std::cout << "e.idx: " << e.idx << " cidx[0]: " << cidx[0] << " cidx[1]: " << cidx[1] << "\n";
		dst[0] = Vec4i(floatToBits(cbox[0]->min().x), floatToBits(cbox[0]->max().x), floatToBits(cbox[0]->min().y), floatToBits(cbox[0]->max().y));
		dst[1] = Vec4i(floatToBits(cbox[1]->min().x), floatToBits(cbox[1]->max().x), floatToBits(cbox[1]->min().y), floatToBits(cbox[1]->max().y));
		dst[2] = Vec4i(floatToBits(cbox[0]->min().z), floatToBits(cbox[0]->max().z), floatToBits(cbox[1]->min().z), floatToBits(cbox[1]->max().z));
		dst[3] = Vec4i(cidx[0], cidx[1], 0, 0);

	} // end of while loop, will iteratively empty the stack


	m_leafnodecount = leafcount;
	m_tricount = woopcount;

	// Write data arrays to arrays of CudaBVH

	m_gpuNodes = (Vec4i*) malloc(nodeData.getNumBytes());
	m_gpuNodesSize = nodeData.getSize();
	
	for (int i = 0; i < nodeData.getSize(); i++){	
		m_gpuNodes[i].x = nodeData.get(i).x;
		m_gpuNodes[i].y = nodeData.get(i).y;
		m_gpuNodes[i].z = nodeData.get(i).z;
		m_gpuNodes[i].w = nodeData.get(i).w; // child indices
	}	

	m_gpuTriNormal = (Vec4i*) malloc(triNormalData.getSize() * sizeof(Vec4i));
	m_gpuTriNormalSize = triNormalData.getSize();

	for (int i = 0; i < triNormalData.getSize(); i++){
		m_gpuTriNormal[i].x = triNormalData.get(i).x;
		m_gpuTriNormal[i].y = triNormalData.get(i).y;
		m_gpuTriNormal[i].z = triNormalData.get(i).z;
		m_gpuTriNormal[i].w = triNormalData.get(i).w;
	}

	m_debugTri = (Vec4i*)malloc(triDebugData.getSize() * sizeof(Vec4i));
	m_debugTriSize = triDebugData.getSize();

	for (int i = 0; i < triDebugData.getSize(); i++){
		m_debugTri[i].x = triDebugData.get(i).x;
		m_debugTri[i].y = triDebugData.get(i).y;
		m_debugTri[i].z = triDebugData.get(i).z;
		m_debugTri[i].w = triDebugData.get(i).w; 
	}

	m_UVTri = (Vec4f*)malloc(triDebugData.getSize() * sizeof(Vec4f));
	m_UVTriSize = triUVData.getSize();

	for (int i = 0; i < triUVData.getSize(); i++){
		m_UVTri[i].x = triUVData.get(i).x;
		m_UVTri[i].y = triUVData.get(i).y;
		m_UVTri[i].z = triUVData.get(i).z;
		m_UVTri[i].w = triUVData.get(i).w; 
	}

	m_gpuTriIndices = (S32*) malloc(triIndexData.getSize() * sizeof(S32));
	m_gpuTriIndicesSize = triIndexData.getSize();

	for (int i = 0; i < triIndexData.getSize(); i++){
		m_gpuTriIndices[i] = triIndexData.get(i);
	}
}

//------------------------------------------------------------------------

void CudaBVH::woopifyTri(const BVH& bvh, int triIdx)
{	
	woopcount++;

	// fetch the 3 vertex indices of this triangle
	const Vec3i& vtxInds = bvh.getScene()->getTriangle(bvh.getTriIndices()[triIdx]).verticeIndex;
    const Vec3i& norInds = bvh.getScene()->getTriangle(bvh.getTriIndices()[triIdx]).normalIndex;
    const Vec3i& uvInds = bvh.getScene()->getTriangle(bvh.getTriIndices()[triIdx]).uvIndex;
	const int& matInds = bvh.getScene()->getTriangle(bvh.getTriIndices()[triIdx]).materialIndex;

	const Vec3f& v0 = Vec3f(scene_info.vertices[vtxInds._v[0]].x, scene_info.vertices[vtxInds._v[0]].y, scene_info.vertices[vtxInds._v[0]].z); // vtx xyz pos voor eerste triangle vtx
	const Vec3f& v1 = Vec3f(scene_info.vertices[vtxInds._v[1]].x, scene_info.vertices[vtxInds._v[1]].y, scene_info.vertices[vtxInds._v[1]].z); // vtx xyz pos voor tweede triangle vtx
	const Vec3f& v2 = Vec3f(scene_info.vertices[vtxInds._v[2]].x, scene_info.vertices[vtxInds._v[2]].y, scene_info.vertices[vtxInds._v[2]].z); // vtx xyz pos voor derde triangle vtx

    const Vec3f& n0 = Vec3f(scene_info.normals[norInds._v[0]].x, scene_info.normals[norInds._v[0]].y, scene_info.normals[norInds._v[0]].z); // vtx xyz pos voor eerste triangle vtx
    const Vec3f& n1 = Vec3f(scene_info.normals[norInds._v[1]].x, scene_info.normals[norInds._v[1]].y, scene_info.normals[norInds._v[1]].z); // vtx xyz pos voor tweede triangle vtx
    const Vec3f& n2 = Vec3f(scene_info.normals[norInds._v[2]].x, scene_info.normals[norInds._v[2]].y, scene_info.normals[norInds._v[2]].z); // vtx xyz pos voor derde triangle vtx
	
    const Vec3f& uv0 = Vec3f(scene_info.uvs[uvInds._v[0]].x, scene_info.uvs[uvInds._v[0]].y, scene_info.uvs[uvInds._v[0]].z); // vtx xyz pos voor eerste triangle vtx
    const Vec3f& uv1 = Vec3f(scene_info.uvs[uvInds._v[1]].x, scene_info.uvs[uvInds._v[1]].y, scene_info.uvs[uvInds._v[1]].z); // vtx xyz pos voor tweede triangle vtx
    const Vec3f& uv2 = Vec3f(scene_info.uvs[uvInds._v[2]].x, scene_info.uvs[uvInds._v[2]].y, scene_info.uvs[uvInds._v[2]].z); // vtx xyz pos voor derde triangle vtx

	// regular triangles (for debugging only)
	m_debugtri[0] = Vec4f(v0.x, v0.y, v0.z, 0.0f);
	m_debugtri[1] = Vec4f(v1.x, v1.y, v1.z, 0.0f);
	m_debugtri[2] = Vec4f(v2.x, v2.y, v2.z, 0.0f);

    m_normaltri[0] = Vec4f(n0.x, n0.y, n0.z, 0.0f);
    m_normaltri[1] = Vec4f(n1.x, n1.y, n1.z, 0.0f);
    m_normaltri[2] = Vec4f(n2.x, n2.y, n2.z, 0.0f);

	m_uvtri[0] = Vec4f(uv0.x, uv0.y, uv0.z, 0.0f);
	m_uvtri[1] = Vec4f(uv1.x, uv1.y, uv1.z, 0.0f);
	m_uvtri[2] = Vec4f(uv2.x, uv2.y, uv2.z, 0.0f);

	m_materialIndex = matInds;

	Mat4f mtx;
	// compute edges and transform them with a matrix 
	mtx.setCol(0, Vec4f(v0 - v2, 0.0f)); // sets matrix column 0 equal to a Vec4f(Vec3f, 0.0f )
	mtx.setCol(1, Vec4f(v1 - v2, 0.0f));
	mtx.setCol(2, Vec4f(cross(v0 - v2, v1 - v2), 0.0f));
	mtx.setCol(3, Vec4f(v2, 1.0f));
	mtx = invert(mtx);   

	/// m_woop[3] stores 3 transformed triangle edges
	m_woop[0] = Vec4f(mtx(2, 0), mtx(2, 1), mtx(2, 2), -mtx(2, 3)); // elements of 3rd row of inverted matrix
 	m_woop[1] = mtx.getRow(0); 
	m_woop[2] = mtx.getRow(1); 
}

//------------------------------------------------------------------------
