#include <cstdio>

#include "BVH.h"
#include "SplitBVHBuilder.h"


BVH::BVH(Scene* scene, const Platform& platform, const BuildParams& params)
{
	FW_ASSERT(scene);
	m_scene = scene;
	m_platform = platform;

	if (params.enablePrints)
		printf("BVH builder: %d tris, %d vertices\n", scene->getNumTriangles(), scene->getNumVertices());

	// SplitBVHBuilder() builds the actual BVH
	m_root = SplitBVHBuilder(*this, params).run();

	if (params.enablePrints)
		printf("BVH: Scene bounds: (%.1f,%.1f,%.1f) - (%.1f,%.1f,%.1f)\n", m_root->m_bounds.min().x, m_root->m_bounds.min().y, m_root->m_bounds.min().z,
		m_root->m_bounds.max().x, m_root->m_bounds.max().y, m_root->m_bounds.max().z);

	float sah = 0.f;
	m_root->computeSubtreeProbabilities(m_platform, 1.f, sah);
	if (params.enablePrints)
		printf("top-down sah: %.2f\n", sah);

	if (params.stats)
	{
		params.stats->SAHCost = sah;
		params.stats->branchingFactor = 2;
		params.stats->numLeafNodes = m_root->getSubtreeSize(BVH_STAT_LEAF_COUNT);
		params.stats->numInnerNodes = m_root->getSubtreeSize(BVH_STAT_INNER_COUNT);
		params.stats->numTris = m_root->getSubtreeSize(BVH_STAT_TRIANGLE_COUNT);
		params.stats->numChildNodes = m_root->getSubtreeSize(BVH_STAT_CHILDNODE_COUNT);
	}
}
