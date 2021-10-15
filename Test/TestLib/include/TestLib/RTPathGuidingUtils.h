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
#include <RTLib/VectorFunction.h>
namespace test {
	using  RTDTreeNode = ::DTreeNode;
	class  RTDTree {
	public:
		RTDTree()noexcept;
		void Reset(const RTDTree& prvDTree, int newMaxDepth, float subDivTh);
		void Build();
		auto GetSum()const noexcept -> float;
		void SetSum(float val)noexcept;
		void SetStatisticalWeight(float val)noexcept;
		auto GetStatisticalWeight()const noexcept -> float;
		auto GetApproxMemoryFootPrint()const noexcept -> size_t;
		auto GetNumNodes()const noexcept -> size_t;
		auto GetDepth()const noexcept -> int;
		auto GetMean()const noexcept -> float;
		auto GetArea()const noexcept -> float;
		template<typename RNG>
		auto Sample(RNG& rng)const noexcept -> float3 {
			if (GetMean() <= 0.0f) {
				return rtlib::canonical_to_dir(rtlib::random_float2(rng));
			}
			return rtlib::canonical_to_dir(m_Nodes[0].Sample(rng, m_Nodes.data()));
		}
		auto Pdf(const float3& dir)const noexcept -> float;
		void Dump(std::fstream& jsonFile)const noexcept;
		auto Nodes()noexcept -> std::vector<RTDTreeNode>&;
		auto Node(size_t idx)const noexcept -> const RTDTreeNode&;
		auto Node(size_t idx) noexcept -> RTDTreeNode&;
	private:
		std::vector<RTDTreeNode> m_Nodes;
		float                    m_Area;
		float                    m_Sum;
		float                    m_StatisticalWeight;
		int                      m_MaxDepth;
	};
	class  RTDTree2 {
		std::vector<unsigned int> indices;
		std::vector<float4>       sums;
	};
	struct RTDTreeWrapper {
		auto GetApproxMemoryFootPrint()const noexcept->size_t;
		auto GetStatisticalWeightSampling()const noexcept -> float;
		auto GetStatisticalWeightBuilding()const noexcept -> float;
		void SetStatisticalWeightSampling(float val)noexcept;
		void SetStatisticalWeightBuilding(float val)noexcept;
		template<typename RNG>
		auto  Sample(RNG& rng)const noexcept -> float3 {
			return sampling.Sample(rng);
		}
		auto  Pdf(const float3& dir)const noexcept -> float;
		auto  GetNumNodes()const noexcept->size_t;
		auto  GetMean()const noexcept->float;
		auto  GetArea()const noexcept->float;
		auto  GetDepth()const noexcept -> int;
		void  Build();
		void  Reset(int newMaxDepth, float subDivTh) {
			//Buildingを削除し、samplingで得た新しい構造に変更
			building.Reset(sampling, newMaxDepth, subDivTh);
		}
		RTDTree    building;
		RTDTree    sampling;
	};
	struct RTSTreeNode {
		RTSTreeNode()noexcept : dTree(), isLeaf{ true }, axis{ 0 }, children{}, padding{}{}
		auto GetChildIdx(float3& p)const noexcept -> int;
		auto GetNodeIdx(float3& p)const noexcept -> unsigned int;
		auto GetDTree(float3 p, float3& size, const std::vector<RTSTreeNode>& nodes)const noexcept -> const RTDTreeWrapper*;
		auto GetDTreeWrapper()const noexcept -> const RTDTreeWrapper*;
		auto GetDepth(const std::vector<RTSTreeNode>& nodes)const-> int;
		void Dump(std::fstream& jsonFile, size_t sTreeNodeIdx, const std::vector<RTSTreeNode>& nodes)const noexcept;
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
		void Clear()noexcept;
		void SubDivideAll();
		void SubDivide(int nodeIdx, std::vector<RTSTreeNode>& nodes);
		auto GetDTree(float3 p, float3& size)const noexcept -> const RTDTreeWrapper*;
		auto GetDTree(const float3& p)const noexcept ->const RTDTreeWrapper*;
		auto GetDepth()const -> int;
		auto Node(size_t idx)const noexcept -> const RTSTreeNode&;
		auto Node(size_t idx) noexcept -> RTSTreeNode&;
		auto GetNumNodes()const noexcept -> size_t;
		bool ShallSplit(const RTSTreeNode& node, int depth, size_t samplesRequired)const noexcept;
		void Refine(size_t sTreeTh, int maxMB);
		auto GetAabbMin()const->float3;
		auto GetAabbMax()const->float3;
		void Dump(std::fstream& jsonFile)const noexcept;
	private:
		std::vector<RTSTreeNode> m_Nodes;
		float3                   m_AabbMin;
		float3                   m_AabbMax;
	};
	class  RTSTreeWrapper {
	public:
		RTSTreeWrapper(const float3& aabbMin, const float3& aabbMax, unsigned int maxDTreeDepth = 20)noexcept :m_CpuSTree{ aabbMin,aabbMax }, m_MaxDTreeDepth{maxDTreeDepth}{}
		void Upload()noexcept;
		void Download() noexcept;
		void Clear();
		auto GetGpuHandle()const noexcept -> STree;
		void Reset(int iter, int samplePerPasses);
		void Build();
		void Dump(std::string filename);
	private:
		RTSTree                         m_CpuSTree;
		rtlib::CUDABuffer<STreeNode>    m_GpuSTreeNodes         = {};//����
		rtlib::CUDABuffer<DTreeWrapper> m_GpuDTreeWrappers      = {};//���L
		rtlib::CUDABuffer<DTreeNode>    m_GpuDTreeNodesBuilding = {};//������
		rtlib::CUDABuffer<DTreeNode>    m_GpuDTreeNodesSampling = {};//������
		unsigned int                    m_MaxDTreeDepth         = 0;
		
	};
}
#endif
