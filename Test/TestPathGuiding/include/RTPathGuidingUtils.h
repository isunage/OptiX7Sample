#ifndef RT_PATH_GUIDING_UTILS_H
#define RT_PATH_GUIDING_UTILS_H
#include "../cuda/PathGuiding.h"
#include <vector>
#include <iterator>
#include <stack>
#include <algorithm>
#include <functional>
#include <RTLib/CUDA.h>
#include <RTLib/VectorFunction.h>
namespace test {
	using  RTDTreeNode = ::DTreeNode;
	class  RTDTree {
	public:
		RTDTree()noexcept {
			m_MaxDepth = 0;
			m_Nodes.emplace_back();
			m_Nodes.front().SetSumAll(0.0f);
			m_Sum = 0.0f;
			m_StatisticalWeight = 0.0f;
		}
		void Reset(const RTDTree& prvDTree, int newMaxDepth, float subDivTh) {
			m_Sum = 0.0f;
			m_StatisticalWeight = 0.0f;
			m_MaxDepth = 0;
			m_Nodes.clear();
			m_Nodes.emplace_back();
			struct StackNode {
				size_t         dstNodeIdx;
				size_t         srcNodeIdx;
				const RTDTree* srcDTree;
				int            depth;
			};
			std::stack<StackNode> stackNodes = {};
			stackNodes.push({ 0,0,&prvDTree,1 });
			const auto total = prvDTree.m_Sum;
			while (!stackNodes.empty())
			{
				StackNode sNode = stackNodes.top();
				stackNodes.pop();

				m_MaxDepth = std::max(m_MaxDepth, sNode.depth);

				for (int i = 0; i < 4; ++i) {
					const auto& srcNode = sNode.srcDTree->Node(sNode.srcNodeIdx);
					const auto fraction = total > 0.0f ? (srcNode.GetSum(i) / total) : std::pow(0.25f, sNode.depth);
					if (sNode.depth < newMaxDepth && fraction > subDivTh) {
						if (!srcNode.IsLeaf(i)) {
							//�O��DTree����ɂ���
							stackNodes.push({ m_Nodes.size(), srcNode.GetChild(i),&prvDTree,sNode.depth + 1 });
						}
						else {
							//Leaf
							//�������g�ɂ��ĕ������邩����
							stackNodes.push({ m_Nodes.size(), m_Nodes.size(), this,sNode.depth + 1 });
						}
						m_Nodes[sNode.dstNodeIdx].SetChild(i, static_cast<unsigned short>(m_Nodes.size()));
						m_Nodes.emplace_back();
						m_Nodes.back().SetSumAll(srcNode.GetSum(i) / 4.0f);
						if (m_Nodes.size() > std::numeric_limits<uint16_t>::max())
						{
							stackNodes = {};
							break;
						}
					}
				}
			}
			for (auto& node : m_Nodes)
			{
				node.SetSumAll(0.0f);
			}
		}
		void Build() {
			auto& root = m_Nodes.front();
			root.Build(m_Nodes);
			float sum = 0.0f;
			for (int i = 0; i < 4; ++i) {
				sum += root.sums[i];
			}
			m_Sum = sum;
		}
		auto GetSum()const noexcept -> float {
			return m_Sum;
		}
		void SetSum(float val)noexcept {
			m_Sum = val;
		}
		void SetStatisticalWeight(float val)noexcept {
			m_StatisticalWeight = val;
		}
		auto GetStatisticalWeight()const noexcept -> float {
			return m_StatisticalWeight;
		}
		auto GetApproxMemoryFootPrint()const noexcept -> size_t {
			return sizeof(DTreeNode) * m_Nodes.size() + sizeof(DTree);
		}
		auto GetNumNodes()const noexcept -> size_t {
			return m_Nodes.size();
		}
		auto GetDepth()const noexcept -> int {
			return m_MaxDepth;
		}
		auto  GetMean()const noexcept -> float {
			if (m_StatisticalWeight <= 0.0f) { return 0.0f; }
			const float factor = 1.0f / (4.0f * RTLIB_M_PI * m_StatisticalWeight);
			return factor * m_Sum;
		}
		auto Node(size_t idx)const noexcept -> const RTDTreeNode& {
			return m_Nodes[idx];
		}
		auto Node(size_t idx) noexcept -> RTDTreeNode& {
			return m_Nodes[idx];
		}
	private:
		std::vector<RTDTreeNode> m_Nodes;
		float                    m_Sum;
		float                    m_StatisticalWeight;
		int                      m_MaxDepth;
	};
	//���������DTreeWrapper�̔z���ʓr�����Ƃ��ēn�����ƂŊ��S�ɕ����ł���
	struct RTSTreeNode {
		RTSTreeNode()noexcept : dTree(), isLeaf{ true }, axis{ 0 }, children{}{
			
		}
		auto GetChildIdx(float3& p)const noexcept -> int {
			float* p_A = reinterpret_cast<float*>(&p);
			if (p_A[axis] < 0.5f) {
				p_A[axis] *= 2.0f;
				return 0;
			}
			else {
				p_A[axis] = 2.0f* p_A[axis]-1.0f;
				return 1;
			}
		}
		auto GetNodeIdx(float3& p)const noexcept -> unsigned int {
			return children[GetChildIdx(p)];
		}
		auto GetDTree(float3 p, float3& size, const std::vector<RTSTreeNode>& nodes)const noexcept -> const RTDTree* {
			const RTSTreeNode* cur = this;
			int   ndx   = cur->GetNodeIdx(p);
			int   depth = 1;
			while (depth < PATH_GUIDING_MAX_DEPTH) {
				if (cur->isLeaf) {
					return &cur->dTree;
				}
				reinterpret_cast<float*>(&size)[axis] /= 2.0f;
				cur = &nodes[ndx];
				ndx = cur->GetNodeIdx(p);
				depth++;
			}
			return nullptr;
		}
		auto GetDTree()const noexcept -> const RTDTree* {
			return &dTree;
		}
		auto GetDepth(const std::vector<RTSTreeNode>& nodes)const-> int {
			int result = 1;
			if (isLeaf) {
				return 1;
			}
			for (int i = 0; i < 2; ++i) {
				result = std::max(result, 1+nodes[children[i]].GetDepth(nodes));
			}
			return result;
		}
		RTDTree                  dTree;
		bool                     isLeaf;
		unsigned char            axis;
		unsigned short           padding;    //2^32���ő�
		unsigned int             children[2];//2^32
	};
	class  RTSTree {
	public:
		RTSTree(const float3& aabbMin, const float3& aabbMax)noexcept {
			this->Clear();
			auto size    = aabbMax - aabbMin;
			auto maxSize = std::max(std::max(size.x, size.y), size.z);
			m_AabbMin = aabbMin;
			m_AabbMax = aabbMin + make_float3(maxSize);
		}
		void Clear()noexcept {
			m_Nodes.clear();
			m_Nodes.emplace_back();
		}
		void SubDivideAll() {
			int nNodes = m_Nodes.size();
			for (size_t i = 0; i < nNodes; ++i)
			{
				if (m_Nodes[i].isLeaf) {
					SubDivide(i, m_Nodes);
				}
			}
		}
		void SubDivide(int nodeIdx, std::vector<RTSTreeNode>& nodes)
		{
			size_t curNodeIdx = nodes.size();
			nodes.resize(curNodeIdx + 2);
			auto& cur = nodes[nodeIdx];
			for (int i = 0; i < 2; ++i)
			{
				uint32_t idx      = curNodeIdx + i;
				cur.children[i]   = idx;
				nodes[idx].axis   = (cur.axis + 1) % 3;
				nodes[idx].isLeaf = true;
				nodes[idx].dTree  = cur.dTree;
				nodes[idx].dTree.SetStatisticalWeight(cur.dTree.GetStatisticalWeight()/2.0f);
			}
			cur.isLeaf = false;
			cur.dTree  = {};
		}
		auto GetDTree(float3 p, float3& size)const noexcept -> const RTDTree* {
			size = m_AabbMax - m_AabbMin;
			p    = p - m_AabbMin;
			p   /= size;
			return m_Nodes[0].GetDTree(p, size, m_Nodes);
		}
		auto GetDTree(const float3& p)const noexcept ->const RTDTree* {
			float3 size;
			return GetDTree(p, size);
		}
		auto GetDepth()const -> int {
			return m_Nodes[0].GetDepth(m_Nodes);
		}
		auto Node(size_t idx)const noexcept -> const RTSTreeNode& {
			return m_Nodes[idx];
		}
		auto Node(size_t idx) noexcept -> RTSTreeNode& {
			return m_Nodes[idx];
		}
		auto GetNumNodes()const noexcept -> size_t {
			return m_Nodes.size();
		}
		bool ShallSplit(const RTSTreeNode& node, int depth, size_t samplesRequired)const noexcept
		{
			std::cout << node.dTree.GetStatisticalWeight() << "vs " << samplesRequired << std::endl;
			return m_Nodes.size() < std::numeric_limits<uint32_t>::max() - 1 && node.dTree.GetStatisticalWeight() > samplesRequired;
		}
		void Refine(size_t sTreeTh, int maxMB) {
			if (maxMB >= 0) {
				size_t approxMemoryFootPrint = 0;
				for (const auto& node : m_Nodes)
				{
					approxMemoryFootPrint += node.GetDTree()->GetApproxMemoryFootPrint();
				}
				if (approxMemoryFootPrint / 1000000 >= maxMB) {
					return;
				}
			}

			struct StackNode {
				size_t index;
				int    depth;
			};
			printf("Refine!\n");
			std::stack<StackNode> nodeIndices = {};
			nodeIndices.push({ 0, 1 });
			while (!nodeIndices.empty())
			{
				StackNode sNode = nodeIndices.top();
				nodeIndices.pop();

				if (m_Nodes[sNode.index].isLeaf) {
					if (ShallSplit(m_Nodes[sNode.index], sNode.depth, sTreeTh)) {
						SubDivide((int)sNode.index, m_Nodes);
					}
				}

				if (!m_Nodes[sNode.index].isLeaf) {
					const auto& node = m_Nodes[sNode.index];
					for (int i = 0; i < 2; ++i) {
						nodeIndices.push({ node.children[i],sNode.depth + 1 });
					}
				}
			}
		}
		auto GetAabbMin()const -> float3 {
			return m_AabbMin;
		}
		auto GetAabbMax()const -> float3 {
			return m_AabbMax;
		}
	private:
		std::vector<RTSTreeNode> m_Nodes;
		float3                   m_AabbMin;
		float3                   m_AabbMax;
	};
	class  RTSTreeWrapper {
	public:
		RTSTreeWrapper(const float3& aabbMin, const float3& aabbMax)noexcept :m_CpuSTree{ aabbMin,aabbMax } {}
		void Upload()noexcept {
			const size_t gpuSTreeNodeCnt = m_CpuSTree.GetNumNodes();
			size_t gpuDTreeCnt     = 0;
			size_t gpuDTreeNodeCnt = 0;
			for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
				if (m_CpuSTree.Node(i).isLeaf) {
					gpuDTreeCnt++;
					gpuDTreeNodeCnt += m_CpuSTree.Node(i).dTree.GetNumNodes();
				}
			}
			//CPU Upload Memory
			std::vector<STreeNode>    sTreeNodes(gpuSTreeNodeCnt);
			std::vector<DTreeWrapper> dTreeWrappers(gpuDTreeCnt);
			std::vector<DTreeNode>    dTreeNodes(gpuDTreeNodeCnt);
			//GPU Upload Memory
			m_GpuSTreeNodes.resize(sTreeNodes.size());
			m_GpuDTreeWrappers.resize(dTreeWrappers.size());
			m_GpuDTreeNodesBuilding.resize(dTreeNodes.size());
			m_GpuDTreeNodesSampling.resize(dTreeNodes.size());
			{
				size_t dTreeIndex      = 0;
				size_t dTreeNodeOffset = 0;
				for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
					sTreeNodes[i].axis        = m_CpuSTree.Node(i).axis;
					sTreeNodes[i].children[0] = m_CpuSTree.Node(i).children[0];
					sTreeNodes[i].children[1] = m_CpuSTree.Node(i).children[1];
					if (m_CpuSTree.Node(i).isLeaf) {
						//DTREE
						sTreeNodes[i].dTree   = m_GpuDTreeWrappers.getDevicePtr() + dTreeIndex;
						//BUILDING
						dTreeWrappers[dTreeIndex].building.sum               = m_CpuSTree.Node(i).dTree.GetSum();
						dTreeWrappers[dTreeIndex].building.statisticalWeight = m_CpuSTree.Node(i).dTree.GetStatisticalWeight();
						dTreeWrappers[dTreeIndex].building.nodes             = m_GpuDTreeNodesBuilding.getDevicePtr() + dTreeNodeOffset;
						//SAMPLING
						dTreeWrappers[dTreeIndex].sampling.sum               = m_CpuSTree.Node(i).dTree.GetSum();
						dTreeWrappers[dTreeIndex].sampling.statisticalWeight = m_CpuSTree.Node(i).dTree.GetStatisticalWeight();
						dTreeWrappers[dTreeIndex].sampling.nodes             = m_GpuDTreeNodesSampling.getDevicePtr() + dTreeNodeOffset;
						//NODES
						for (size_t j = 0; j < m_CpuSTree.Node(i).dTree.GetNumNodes(); ++j) {
							//SUMS
							dTreeNodes[dTreeNodeOffset + j] = m_CpuSTree.Node(i).dTree.Node(j);
						}
						dTreeNodeOffset += m_CpuSTree.Node(i).dTree.GetNumNodes();
						dTreeIndex++;
					}
					else {
						sTreeNodes[i].dTree   = nullptr;
					}
				}
			}
			//Upload
			m_GpuSTreeNodes.upload(sTreeNodes);
			m_GpuDTreeWrappers.upload(dTreeWrappers);
			m_GpuDTreeNodesBuilding.upload(dTreeNodes);
			m_GpuDTreeNodesSampling.upload(dTreeNodes);
		}
		void Download() noexcept{
			const size_t gpuSTreeNodeCnt = m_CpuSTree.GetNumNodes();
			size_t gpuDTreeCnt     = 0;
			size_t gpuDTreeNodeCnt = 0;
			for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
				if (m_CpuSTree.Node(i).isLeaf) {
					gpuDTreeCnt++;
					gpuDTreeNodeCnt += m_CpuSTree.Node(i).dTree.GetNumNodes();
				}
			}
			std::vector<DTreeWrapper> dTreeWrappers(gpuDTreeCnt);
			std::vector<DTreeNode>    dTreeNodes(gpuDTreeNodeCnt);
			m_GpuDTreeWrappers.download(dTreeWrappers);
			m_GpuDTreeNodesBuilding.download(dTreeNodes);
			{
				size_t cpuDTreeIndex      = 0;
				size_t cpuDTreeNodeOffset = 0;
				//�\���͕ς���Ă��Ȃ��Ƒz��
				for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
					if (m_CpuSTree.Node(i).isLeaf) {
						m_CpuSTree.Node(i).dTree.SetSum(dTreeWrappers[cpuDTreeIndex].building.sum);
						m_CpuSTree.Node(i).dTree.SetStatisticalWeight(dTreeWrappers[cpuDTreeIndex].building.statisticalWeight);
						for (size_t j = 0; j < m_CpuSTree.Node(i).dTree.GetNumNodes(); ++j) {
							//SUMS
							m_CpuSTree.Node(i).dTree.Node(j) = dTreeNodes[cpuDTreeNodeOffset + j];
						}
						cpuDTreeNodeOffset += m_CpuSTree.Node(i).dTree.GetNumNodes();
						cpuDTreeIndex++;
					}
				}
			}
		}
		auto GetGpuHandle()const noexcept -> STree {
			STree sTree;
			sTree.aabbMax = m_CpuSTree.GetAabbMax();
			sTree.aabbMin = m_CpuSTree.GetAabbMin();
			sTree.nodes   = m_GpuSTreeNodes.getDevicePtr();
			return sTree;
		}
		void Reset(int samplePerAll) {
			int iter = std::log2(samplePerAll);
			size_t sTreeTh = std::pow(2.0, iter) / 4.0f * 4000;

			m_CpuSTree.Refine(sTreeTh,600);
			for (int i = 0; i < m_CpuSTree.GetNumNodes(); ++i) {
				if (m_CpuSTree.Node(i).isLeaf) {
					auto dTree = m_CpuSTree.Node(i).dTree;
					m_CpuSTree.Node(i).dTree.Reset(dTree,20,0.001);
				}
			}
		}
		void Build() {
			for (int i = 0; i < m_CpuSTree.GetNumNodes(); ++i) {
				if (m_CpuSTree.Node(i).isLeaf) {
					m_CpuSTree.Node(i).dTree.Build();
				}
			}
			int   maxDepth = 0;
			int   minDepth = std::numeric_limits<int>::max();
			float avgDepth = 0.0f;
			float maxAvgRadiance = 0.0f;
			float minAvgRadiance = std::numeric_limits<float>::max();
			float avgAvgRadiance = 0.0f;
			size_t maxNodes = 0;
			size_t minNodes = std::numeric_limits<size_t>::max();
			float avgNodes  = 0.0f;
			float maxStatisticalWeight = 0;
			float minStatisticalWeight = std::numeric_limits<float>::max();
			float avgStatisticalWeight = 0;

			int nPoints = 0;
			int nPointsNodes = 0;

			for (int i = 0; i < m_CpuSTree.GetNumNodes(); ++i) {
				if (m_CpuSTree.Node(i).isLeaf) {
					auto& dTree = m_CpuSTree.Node(i).dTree;
					const int depth = dTree.GetDepth();
					maxDepth = std::max<int>(maxDepth, depth);
					minDepth = std::min<int>(minDepth, depth);
					avgDepth += depth;

					const float avgRadiance = dTree.GetMean();
					maxAvgRadiance = std::max<float>(maxAvgRadiance, avgRadiance);
					minAvgRadiance = std::min<float>(minAvgRadiance, avgRadiance);
					avgAvgRadiance += avgRadiance;

					if (dTree.GetNumNodes() >= 1) {

						const size_t numNodes = dTree.GetNumNodes();
						maxNodes              = std::max<size_t>(maxNodes, numNodes);
						minNodes              = std::min<size_t>(minNodes, numNodes);
						avgNodes             += numNodes;
						++nPointsNodes;
					}

					const auto statisticalWeight = dTree.GetStatisticalWeight();
					maxStatisticalWeight         = std::max<float>(maxStatisticalWeight, statisticalWeight);
					minStatisticalWeight         = std::min<float>(minStatisticalWeight, statisticalWeight);
					avgStatisticalWeight        += statisticalWeight;

					++nPoints;
				}
			}

			if (nPoints > 0) {
				avgDepth /= nPoints;
				avgAvgRadiance /= nPoints;
				if (nPointsNodes) {
					avgNodes /= nPointsNodes;
				}
				avgStatisticalWeight /= nPoints;
			}
			std::cout << "SDTree Build Statistics\n";
			std::cout << "Depth(STree):      " << m_CpuSTree.GetDepth() << std::endl;
			std::cout << "Depth(DTree):      " << minDepth       << "," << avgDepth       << "," << maxDepth       << std::endl;
			std::cout << "Node count:        " << minNodes       << "," << avgNodes       << "," << maxNodes       << std::endl;
			std::cout << "Mean Radiance:     " << minAvgRadiance << "," << avgAvgRadiance << "," << maxAvgRadiance << std::endl;
			std::cout << "statisticalWeight: " << minStatisticalWeight << "," << avgStatisticalWeight << "," << maxStatisticalWeight << std::endl;
		}
	private:
		RTSTree                         m_CpuSTree;
		rtlib::CUDABuffer<STreeNode>    m_GpuSTreeNodes         = {};//����
		rtlib::CUDABuffer<DTreeWrapper> m_GpuDTreeWrappers      = {};//���L
		rtlib::CUDABuffer<DTreeNode>    m_GpuDTreeNodesBuilding = {};//������
		rtlib::CUDABuffer<DTreeNode>    m_GpuDTreeNodesSampling = {};//������
		
	};
}
#endif
