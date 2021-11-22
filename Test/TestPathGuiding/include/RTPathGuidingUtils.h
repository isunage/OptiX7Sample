#ifndef RT_PATH_GUIDING_UTILS_H
#define RT_PATH_GUIDING_UTILS_H
#include "../cuda/PathGuiding.h"
#include <vector>
#include <iterator>
#include <fstream>
#include <stack>
#include <algorithm>
#include <functional>
#include <stb_image_write.h>
#include <RTLib/CUDA.h>
#include <RTLib/ext/Math/VectorFunction.h>
namespace test {
	using  RTDTreeNode = ::DTreeNode;
	class  RTDTree {
	public:
		RTDTree()noexcept {
			m_MaxDepth = 0;
			m_Nodes.emplace_back();
			m_Nodes.front().SetSumAll(0.0f);
			m_Sum  = 0.0f;
			m_Area = 0.0f;
			m_StatisticalWeight = 0.0f;
		}
		void Reset(const RTDTree& prvDTree, int newMaxDepth, float subDivTh) {
			m_Area              = 0.0f;
			m_Sum               = 0.0f;
			m_StatisticalWeight = 0.0f;
			m_MaxDepth          = 0;
			m_Nodes.clear();
			m_Nodes.emplace_back();
			struct StackNode {
				size_t         dstNodeIdx;
				//const RTDTree* dstDTree = this;
				size_t         srcNodeIdx;
				const RTDTree* srcDTree;
				int            depth;
				auto GetSrcNode()const -> const RTDTreeNode& {
					return srcDTree->Node(srcNodeIdx);
				}
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
					//this
					const auto fraction = total > 0.0f ? (sNode.GetSrcNode().GetSum(i) / total) : std::pow(0.25f, sNode.depth);
					if (sNode.depth < newMaxDepth && fraction > subDivTh) {
						if (!sNode.GetSrcNode().IsLeaf(i)) {
							if (sNode.srcDTree != &prvDTree) {
								std::cout << "sNode.srcDTree != &prvDTree!\n";
							}
							//Not Leaf -> Copy Child
							stackNodes.push({ m_Nodes.size(), sNode.GetSrcNode().GetChild(i),&prvDTree,sNode.depth + 1 });
						}
						else {
							//    Leaf -> Copy Itself
							stackNodes.push({ m_Nodes.size(), m_Nodes.size()                , this    ,sNode.depth + 1 });
						}
						m_Nodes[sNode.dstNodeIdx].SetChild(i, static_cast<unsigned short>(m_Nodes.size()));
						m_Nodes.emplace_back();
						auto& backNode = m_Nodes.back();
						backNode.SetSumAll(sNode.GetSrcNode().GetSum(i) / 4.0f);
						if (m_Nodes.size() > std::numeric_limits<uint16_t>::max())
						{
							std::cout << "DTreeWrapper hit maximum count!\n";
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
			m_Area     = root.GetArea(m_Nodes);
			float sum  = 0.0f;
			for (int i = 0; i < 4; ++i) {
				sum   += root.sums[i];
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
		auto GetMean()const noexcept -> float {
			if (m_StatisticalWeight * m_Area<= 0.0f) { return 0.0f; }
			const float factor = 1.0f / (4.0f * RTLIB_M_PI * m_Area * m_StatisticalWeight);
			return factor * m_Sum;
		}
		auto GetArea()const noexcept -> float {
			return m_Area;
		}
		template<typename RNG>
		auto Sample(RNG& rng)const noexcept -> float3 {
			if (GetMean() <= 0.0f) {
				return rtlib::canonical_to_dir(rtlib::random_float2(rng));
			}
			return rtlib::canonical_to_dir(m_Nodes[0].Sample(rng, m_Nodes.data()));
		}
		auto Pdf(const float3& dir)const noexcept -> float {
			if (GetMean() <= 0.0f) {
				return 1.0f / (4.0f * RTLIB_M_PI);
			}
			auto dir2 = rtlib::dir_to_canonical(dir);
			return m_Area * m_Nodes[0].Pdf(dir2,m_Nodes.data()) / (4.0f * RTLIB_M_PI);
		}
		void Dump(std::fstream& jsonFile)const noexcept {
			jsonFile << "{\n";
			jsonFile << "\"sum\"              : " << m_Sum               << ",\n";
			jsonFile << "\"statisticalWeight\": " << m_StatisticalWeight << ",\n";
			jsonFile << "\"maxDepth\"         : " << m_MaxDepth          << ",\n";
			jsonFile << "\"root\"             :  \n";
			m_Nodes[0].Dump(jsonFile, m_Nodes);
			jsonFile << "\n";
			jsonFile << "}";
		}
		void SavePDFImage(int dTreeId)const {
			int width  = 256;
			int height = 256;
			std::unique_ptr<unsigned char[]> pixels(new unsigned char[width * height * 4]);
			for (int j = 0; j < height; ++j) {
				for (int i = 0; i < width; ++i) {
					auto dir2 = make_float2((float)i / (float)width, (float)j / (float)height);
					auto pdf  = m_Nodes[0].Pdf(dir2, m_Nodes.data());
					if (pdf <= 0.0f) {
						pixels[4 * (width * j + i) + 0] = 0;
						pixels[4 * (width * j + i) + 1] = 0;
						pixels[4 * (width * j + i) + 2] = 0;
						pixels[4 * (width * j + i) + 3] = 255;
					}
					else {
						pixels[4 * (width * j + i) + 0] = 0;
						pixels[4 * (width * j + i) + 1] = 0;
						pixels[4 * (width * j + i) + 2] = static_cast<unsigned char>(255.99f*1.0f/pdf);
						pixels[4 * (width * j + i) + 3] = 255;
					}
				}
			}
			stbi_write_png((std::string("images/dTree") + std::to_string(dTreeId) + ".png").c_str(), width, height, 4, pixels.get(), 4 * width);
		}
		auto Nodes()noexcept -> std::vector<RTDTreeNode>& {
			return m_Nodes;
		}
		auto Node(size_t idx)const noexcept -> const RTDTreeNode& {
			return m_Nodes[idx];
		}
		auto Node(size_t idx) noexcept -> RTDTreeNode& {
			return m_Nodes[idx];
		}
	private:
		std::vector<RTDTreeNode> m_Nodes;
		float                    m_Area;
		float                    m_Sum;
		float                    m_StatisticalWeight;
		int                      m_MaxDepth;
	};
	struct RTDTreeWrapper {
		auto GetApproxMemoryFootPrint()const noexcept->size_t {
			return building.GetApproxMemoryFootPrint() + sampling.GetApproxMemoryFootPrint();
		}
		auto GetStatisticalWeightSampling()const noexcept -> float {
			return sampling.GetStatisticalWeight();
		}
		auto GetStatisticalWeightBuilding()const noexcept -> float {
			return building.GetStatisticalWeight();
		}
		void SetStatisticalWeightSampling(float val)noexcept {
			sampling.SetStatisticalWeight(val);
		}
		void SetStatisticalWeightBuilding(float val)noexcept {
			building.SetStatisticalWeight(val);
		}
		template<typename RNG>
		auto  Sample(RNG& rng)const noexcept -> float3 {
			return sampling.Sample(rng);
		}
		auto  Pdf(const float3& dir)const noexcept -> float {
			return sampling.Pdf(dir);
		}
		auto  GetNumNodes()const noexcept->size_t {
			return sampling.GetNumNodes();
		}
		auto  GetMean()const noexcept->float {
			return sampling.GetMean();
		}
		auto  GetArea()const noexcept->float {
			return sampling.GetArea();
		}
		auto  GetDepth()const noexcept -> int {
			return sampling.GetDepth();
		}
		void  Build() {
			//一層にする→うまくいく
			building.Build();
			sampling = building;
		}
		void  Reset(int newMaxDepth, float subDivTh) {
			//Buildingを削除し、samplingで得た新しい構造に変更
			building.Reset(sampling, newMaxDepth, subDivTh);
		}
		RTDTree    building;
		RTDTree    sampling;
	};
	struct RTSTreeNode {
		RTSTreeNode()noexcept : dTree(), isLeaf{ true }, axis{ 0 }, children{}{}
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
		auto GetDTree(float3 p, float3& size, const std::vector<RTSTreeNode>& nodes)const noexcept -> const RTDTreeWrapper* {
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
		auto GetDTreeWrapper()const noexcept -> const RTDTreeWrapper* {
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
		void Dump(std::fstream& jsonFile, size_t sTreeNodeIdx, const std::vector<RTSTreeNode>& nodes)const noexcept {
			jsonFile << "{\n";
			jsonFile << "\"isLeaf\"  : " << (this->isLeaf ? "true" : "false") << ",\n";
			jsonFile << "\"axis\"    : " << (int)this->axis << ",\n";
			if (!this->isLeaf) {
				jsonFile << "\"children\": [\n";
				nodes[this->children[0]].Dump(jsonFile, children[0], nodes);
				jsonFile << ",\n";
				nodes[this->children[1]].Dump(jsonFile, children[1], nodes);
				jsonFile << "\n";

				jsonFile << "]\n";
			}
			else {
				jsonFile << "\"dTree\": \"" << "dTree" << sTreeNodeIdx << "\"";
			}
			jsonFile << "}";
		}
		RTDTreeWrapper           dTree;
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
				nodes[idx].dTree.building.SetStatisticalWeight(cur.dTree.building.GetStatisticalWeight()/2.0f);
			}
			cur.isLeaf = false;
			cur.dTree  = {};
		}
		auto GetDTree(float3 p, float3& size)const noexcept -> const RTDTreeWrapper* {
			size = m_AabbMax - m_AabbMin;
			p    = p - m_AabbMin;
			p   /= size;
			return m_Nodes[0].GetDTree(p, size, m_Nodes);
		}
		auto GetDTree(const float3& p)const noexcept ->const RTDTreeWrapper* {
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
			//std::cout << node.dTree.GetStatisticalWeight() << "vs " << samplesRequired << std::endl;
			return m_Nodes.size() < std::numeric_limits<uint32_t>::max() - 1 && node.dTree.building.GetStatisticalWeight() > samplesRequired;
		}
		void Refine(size_t sTreeTh, int maxMB) {
			if (maxMB >= 0) {
				size_t approxMemoryFootPrint = 0;
				for (const auto& node : m_Nodes)
				{
					approxMemoryFootPrint += node.GetDTreeWrapper()->GetApproxMemoryFootPrint();
				}
				if (approxMemoryFootPrint / 1000000 >= maxMB) {
					return;
				}
			}

			struct StackNode {
				size_t index;
				int    depth;
			};
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
		void Dump(std::fstream& jsonFile)const noexcept{
			jsonFile << "{\n";
			jsonFile << "\"aabbMin\" : [" << m_AabbMin.x << ", " << m_AabbMin.y << ", " << m_AabbMin.z << "],\n";
			jsonFile << "\"aabbMax\" : [" << m_AabbMax.x << ", " << m_AabbMax.y << ", " << m_AabbMax.z << "],\n";
			jsonFile << "\"root\"    : \n";
			m_Nodes[0].Dump(jsonFile,0, m_Nodes);
			jsonFile << "\n";
			jsonFile << "}\n";
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
			//Uploadは両方必要
			const size_t gpuSTreeNodeCnt   = m_CpuSTree.GetNumNodes();
			size_t gpuDTreeCnt             = 0;
			size_t gpuDTreeNodeCntBuilding = 0;
			size_t gpuDTreeNodeCntSampling = 0;
			for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
				if (m_CpuSTree.Node(i).isLeaf) {
					gpuDTreeCnt++;
					gpuDTreeNodeCntBuilding += m_CpuSTree.Node(i).dTree.building.GetNumNodes();
					gpuDTreeNodeCntSampling += m_CpuSTree.Node(i).dTree.sampling.GetNumNodes();
				}
			}
			//CPU Upload Memory
			std::vector<STreeNode>    sTreeNodes(gpuSTreeNodeCnt);
			std::vector<DTreeWrapper> dTreeWrappers(gpuDTreeCnt);
			std::vector<DTreeNode>    dTreeNodesBuilding(gpuDTreeNodeCntBuilding);
			std::vector<DTreeNode>    dTreeNodesSampling(gpuDTreeNodeCntSampling);
			//GPU Upload Memory
			m_GpuSTreeNodes.resize(sTreeNodes.size());
			m_GpuDTreeWrappers.resize(dTreeWrappers.size());
			m_GpuDTreeNodesBuilding.resize(dTreeNodesBuilding.size());
			m_GpuDTreeNodesSampling.resize(dTreeNodesSampling.size());
			{
				size_t dTreeIndex      = 0;
				size_t dTreeNodeOffsetBuilding = 0;
				size_t dTreeNodeOffsetSampling = 0;
				for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
					sTreeNodes[i].axis        = m_CpuSTree.Node(i).axis;
					sTreeNodes[i].children[0] = m_CpuSTree.Node(i).children[0];
					sTreeNodes[i].children[1] = m_CpuSTree.Node(i).children[1];
					if (m_CpuSTree.Node(i).isLeaf) {
						//DTREE
						sTreeNodes[i].dTree   = m_GpuDTreeWrappers.getDevicePtr() + dTreeIndex;
						//BUILDING
						dTreeWrappers[dTreeIndex].building.area              = m_CpuSTree.Node(i).dTree.building.GetArea();
						dTreeWrappers[dTreeIndex].building.sum               = m_CpuSTree.Node(i).dTree.building.GetSum();
						dTreeWrappers[dTreeIndex].building.statisticalWeight = m_CpuSTree.Node(i).dTree.building.GetStatisticalWeight();
						dTreeWrappers[dTreeIndex].building.nodes             = m_GpuDTreeNodesBuilding.getDevicePtr() + dTreeNodeOffsetBuilding;
						for (size_t j = 0; j < m_CpuSTree.Node(i).dTree.building.GetNumNodes(); ++j) {
							//SUM
							dTreeNodesBuilding[dTreeNodeOffsetBuilding + j]  = m_CpuSTree.Node(i).dTree.building.Node(j);
						}
						dTreeNodeOffsetBuilding += m_CpuSTree.Node(i).dTree.building.GetNumNodes();
						//SAMPLING
						dTreeWrappers[dTreeIndex].sampling.area              = m_CpuSTree.Node(i).dTree.sampling.GetArea();
						dTreeWrappers[dTreeIndex].sampling.sum               = m_CpuSTree.Node(i).dTree.sampling.GetSum();
						dTreeWrappers[dTreeIndex].sampling.statisticalWeight = m_CpuSTree.Node(i).dTree.sampling.GetStatisticalWeight();
						dTreeWrappers[dTreeIndex].sampling.nodes             = m_GpuDTreeNodesSampling.getDevicePtr() + dTreeNodeOffsetSampling;
						for (size_t j = 0; j < m_CpuSTree.Node(i).dTree.sampling.GetNumNodes(); ++j) {
							//SUMS
							dTreeNodesSampling[dTreeNodeOffsetSampling + j]  = m_CpuSTree.Node(i).dTree.sampling.Node(j);
						}
						dTreeNodeOffsetSampling += m_CpuSTree.Node(i).dTree.sampling.GetNumNodes();
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
			m_GpuDTreeNodesBuilding.upload(dTreeNodesBuilding);
			m_GpuDTreeNodesSampling.upload(dTreeNodesSampling);
#if 0
			std::cout << "Upload(Info)\n";
			std::cout << "GpuSTreeNodes          : " << m_GpuSTreeNodes.getSizeInBytes()         / (1024.0f * 1024.0f) << "MB\n";
			std::cout << "GpuDTreeNodes(Building): " << m_GpuDTreeNodesBuilding.getSizeInBytes() / (1024.0f * 1024.0f) << "MB\n";
			std::cout << "GpuDTreeNodes(Sampling): " << m_GpuDTreeNodesSampling.getSizeInBytes() / (1024.0f * 1024.0f) << "MB\n";
#endif
		}
		void Download() noexcept{
			//ダウンロードが必要なのはBuildingだけ
			const size_t gpuSTreeNodeCnt   = m_CpuSTree.GetNumNodes();
			size_t gpuDTreeCnt             = 0;
			size_t gpuDTreeNodeCntBuilding = 0;
			for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
				if (m_CpuSTree.Node(i).isLeaf) {
					gpuDTreeCnt++;
					gpuDTreeNodeCntBuilding += m_CpuSTree.Node(i).dTree.building.GetNumNodes();
				}
			}
			std::vector<DTreeWrapper> dTreeWrappers(gpuDTreeCnt);
			std::vector<DTreeNode>    dTreeNodesBuilding(gpuDTreeNodeCntBuilding);
			m_GpuDTreeWrappers.download(dTreeWrappers);
			m_GpuDTreeNodesBuilding.download(dTreeNodesBuilding);
			{
				size_t cpuDTreeIndex      = 0;
				size_t cpuDTreeNodeOffsetBuilding = 0;
				for (size_t i = 0; i < gpuSTreeNodeCnt; ++i) {
					if (m_CpuSTree.Node(i).isLeaf) {
						m_CpuSTree.Node(i).dTree.building.SetSum(dTreeWrappers[cpuDTreeIndex].building.sum);
						m_CpuSTree.Node(i).dTree.building.SetStatisticalWeight(dTreeWrappers[cpuDTreeIndex].building.statisticalWeight);
						for (size_t j = 0; j < m_CpuSTree.Node(i).dTree.building.GetNumNodes(); ++j) {
							//SUMS
							m_CpuSTree.Node(i).dTree.building.Node(j) = dTreeNodesBuilding[cpuDTreeNodeOffsetBuilding + j];
						}
						cpuDTreeNodeOffsetBuilding += m_CpuSTree.Node(i).dTree.building.GetNumNodes();
						cpuDTreeIndex++;
					}
				}
			}
		}
		void Clear() {
			m_CpuSTree = RTSTree(m_CpuSTree.GetAabbMin(), m_CpuSTree.GetAabbMax());
		}
		auto GetGpuHandle()const noexcept -> STree {
			STree sTree;
			sTree.aabbMax = m_CpuSTree.GetAabbMax();
			sTree.aabbMin = m_CpuSTree.GetAabbMin();
			sTree.nodes   = m_GpuSTreeNodes.getDevicePtr();
			return sTree;
		}
		void Reset(int iter, int samplePerPasses) {
			if (iter <= 0) {
				return;
			}
			size_t sTreeTh = std::sqrt(std::pow(2.0, iter) * samplePerPasses / 4.0f )* 4000;
			m_CpuSTree.Refine(sTreeTh,2000);
			for (int i = 0; i < m_CpuSTree.GetNumNodes(); ++i) {
				if (m_CpuSTree.Node(i).isLeaf) {
					m_CpuSTree.Node(i).dTree.Reset(20,0.01);
				}
			}
		}
		void Build() {
			size_t bestIdx = 0;
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
			bool isSaved = false;
			for (int i = 0; i < m_CpuSTree.GetNumNodes(); ++i) {
				if (m_CpuSTree.Node(i).isLeaf) {
					auto& dTree = m_CpuSTree.Node(i).dTree;
					//printf("Area = %f\n", dTree.sampling.GetArea());
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

					const auto statisticalWeight = dTree.GetStatisticalWeightSampling();
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
#if 0
			std::cout << "SDTree Build Statistics\n";
			std::cout << "Depth(STree):      " << m_CpuSTree.GetDepth() << std::endl;
			std::cout << "Depth(DTree):      " << minDepth       << "," << avgDepth       << "," << maxDepth       << std::endl;
			std::cout << "Node count:        " << minNodes       << "," << avgNodes       << "," << maxNodes       << std::endl;
			std::cout << "Mean Radiance:     " << minAvgRadiance << "," << avgAvgRadiance << "," << maxAvgRadiance << std::endl;
			std::cout << "statisticalWeight: " << minStatisticalWeight << "," << avgStatisticalWeight << "," << maxStatisticalWeight << std::endl;
#endif
		}
		void Dump(std::string filename) {
			std::fstream jsonFile(filename, std::ios::binary | std::ios::out);
			jsonFile << "{\n";
			jsonFile << "\"STree\":\n";
			m_CpuSTree.Dump(jsonFile);
			jsonFile << "}\n";
			jsonFile.close();
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
