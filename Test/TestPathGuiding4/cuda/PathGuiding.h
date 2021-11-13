#ifndef PATH_GUIDING_H
#define PATH_GUIDING_H
#include <RTLib/Math.h>
#include <RTLib/Random.h>
#include <RTLib/VectorFunction.h>
#include <PathGuidingConfig.h>
#ifndef __CUDA_ARCH__
#include <fstream>
#include <stack>
#include <vector>
#endif
enum   SpatialFilter
{
	//Suitable in GPU
	SpatialFilterNearest, 
	//Too Slow in GPU
	SpatialFilterBox,
};
enum   DirectionalFilter {
	//Suitable in GPU
	DirectionalFilterNearest,
	//Too Slow in GPU
	DirectionalFilterBox,
};
struct DTreeNode {
	RTLIB_INLINE RTLIB_HOST_DEVICE DTreeNode()noexcept {
		for (int i = 0; i < 4; ++i) {
			children[i] = 0;
			sums[i] = 0.0f;
		}
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE DTreeNode(const DTreeNode& node)noexcept {
		CopyFrom(node);
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE DTreeNode& operator=(const DTreeNode& node)noexcept {
		CopyFrom(node);
		return *this;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE void CopyFrom(const DTreeNode& node)noexcept {
		for (int i = 0; i < 4; ++i) {
			sums[i] = node.sums[i];
			children[i] = node.children[i];
		}
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE bool IsLeaf(int childIdx)const noexcept {
		return children[childIdx] == 0;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto GetChildIdx(float2& p)const noexcept -> int {
		int result = 0;
		float* p_A = reinterpret_cast<float*>(&p);
		for (int i = 0; i < 2; ++i) {
			if (p_A[i] < 0.5f) {
				MoveToLeft(p_A[i]);
			}
			else {
				MoveToRight(p_A[i]);
				result |= (1 << i);
			}
		}
		return result;
	}
	//SUM
	RTLIB_INLINE RTLIB_HOST_DEVICE auto GetSum(int childIdx)const noexcept -> float {
		return sums[childIdx];
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE void SetSum(int childIdx, float val)noexcept {
		sums[childIdx] = val;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE void SetSumAll(float val)noexcept {
		for (int i = 0; i < 4; ++i) {
			sums[i] = val;
		}
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto GetSumOfAll()const noexcept -> float {
		return sums[0] + sums[1] + sums[2] + sums[3];
	}
	RTLIB_INLINE RTLIB_DEVICE      void AddSumAtomic(int idx, float val)noexcept {
#ifdef __CUDA_ARCH__
		atomicAdd(&sums[idx], val);
#else
		sums[idx] += val;
#endif
	}
	//CHILD
	RTLIB_INLINE RTLIB_HOST_DEVICE auto GetChild(int childIdx)const noexcept -> unsigned short {
		return children[childIdx];
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE void SetChild(int childIdx, unsigned short val)noexcept {
		children[childIdx] = val;
	}
	//RECORD
	RTLIB_INLINE RTLIB_DEVICE      void Record(float2& p, float irradiance, DTreeNode* nodes)noexcept {
		DTreeNode* cur = this;
		int        idx = cur->GetChildIdx(p);
		int      depth = 1;
		while (!cur->IsLeaf(idx)) {
			//Leafだったら加算する
			cur = &nodes[cur->children[idx]];
			idx = cur->GetChildIdx(p);
			++depth;
		}
		cur->AddSumAtomic(idx, irradiance);
		return;
	}
	RTLIB_INLINE RTLIB_DEVICE      void  Record(const float2& origin, float size, float2 nodeOrigin, float nodeSize, float value, DTreeNode* nodes) noexcept {
		struct StackNode {
			DTreeNode* curNode;
			float2     nodeOrigin;
			float      nodeSize;
		};

		StackNode stackNodes[PATH_GUIDING_DTREE_MAX_DEPTH*3]  = {};
		int       stackNodeSize     = 1;
		const int stackNodeCapacity = PATH_GUIDING_DTREE_MAX_DEPTH *3;

		stackNodes[0].curNode       = this;
		stackNodes[0].nodeOrigin    = nodeOrigin;
		stackNodes[0].nodeSize      = nodeSize;

		while (stackNodeSize > 0) {

			auto  top = stackNodes[stackNodeSize - 1];
			stackNodeSize--;

			if (stackNodeSize > stackNodeCapacity - 4) {
				continue;
			}

			float childSize = top.nodeSize / 2.0f;
			for (int i = 0; i < 4; ++i) {
				float2 childOrigin = top.nodeOrigin;
				if (i & 1) { childOrigin.x += childSize; }
				if (i & 2) { childOrigin.y += childSize; }
				float w = ComputeOverlappingVolume(origin, origin + make_float2(size), childOrigin, childOrigin + make_float2(childSize));
				if (w > 0.0f) {
					if (top.curNode->IsLeaf(i)) {
						top.curNode->AddSumAtomic(i, value * w);
					}
					else {
						stackNodeSize++;
						stackNodes[stackNodeSize - 1].curNode = &nodes[top.curNode->GetChild(i)];
						stackNodes[stackNodeSize - 1].nodeOrigin = childOrigin;
						stackNodes[stackNodeSize - 1].nodeSize = childSize;
					}
				}
			}
		}
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE static auto ComputeOverlappingVolume(const float2& min1, const float2& max1, const float2& min2, const float2& max2)noexcept->float {
		float lengths[2] = {
			fmaxf(fminf(max1.x,max2.x) - fmaxf(min1.x,min2.x),0.0f),
			fmaxf(fminf(max1.y,max2.y) - fmaxf(min1.y,min2.y),0.0f)
		};
		return lengths[0] * lengths[1];
	}
	template<typename RNG>
	RTLIB_INLINE RTLIB_HOST_DEVICE auto Sample(RNG& rng, const DTreeNode* nodes)const noexcept -> float2 {
		const DTreeNode* cur = this;
		float2 result        = make_float2(0.0f);
		double size          = 1.0f;
		for (;;) {
			int   idx      = 0;
			float topLeft  = cur->sums[0];
			float topRight = cur->sums[1];
			float partial  = cur->sums[0] + cur->sums[2];
			float total    = cur->GetSumOfAll();
			//use Two RND Value
			//use Signle RND Value
			//(s0+s2)/(s0+s1+s2+s3)
			float boundary = partial / total;
			auto  origin   = make_float2(0.0f);
			float sample   = rtlib::random_float1(rng);

			if (sample < boundary)
			{
				sample   = sample  / boundary;
				boundary = topLeft / partial;
			}
			else
			{
				partial  = total - partial;
				origin.x = 0.5f;
				sample   = (sample - boundary) / (1.0f - boundary);
				boundary = topRight / partial;
				idx     |= (1 << 0);
			}
			if (sample >= boundary){
				origin.y = 0.5f;
				idx |= (1 << 1);
			}

			result += size * origin;
			size   *= 0.5f;

			if (cur->IsLeaf(idx)||cur->sums[idx]<=0.0f)
			{
				result += size * rtlib::random_float2(rng);
				break;
			}

			cur = &nodes[cur->children[idx]];
		}
		return result;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto Pdf(float2& p, const DTreeNode* nodes)const noexcept -> float
	{

		float         result = 1.0f;
		const DTreeNode* cur = this;
		int              idx = cur->GetChildIdx(p);
		int            depth = 1;
		for (;;) {
			auto total = cur->GetSumOfAll();
			//if (total <= 0.0f) {
				//break;
#if 0
			float mulOfSum = cur->sums[0] * cur->sums[1] * cur->sums[2] * cur->sums[3];
			//if( total <=0.0f){
			if (mulOfSum <= 0.0f)
#else
			if (cur->GetSum(idx) <= 0.0f)
#endif
			{
				result = 0.0f;
				break;
			}
			//if (isnan(cur->sums[idx])) {
			//	printf("sums = (%f, %f, %f, %f)\n", cur->sums[0], cur->sums[1], cur->sums[2], cur->sums[3]);
			//}
			//if (cur == nullptr) {
			//	printf("cur = nullptr\n");
			//}
			const auto factor = 4.0f * cur->GetSum(idx) / total;
			result *= factor;
			if (cur->IsLeaf(idx)) {
				break;
			}
			cur = &nodes[cur->children[idx]];
			idx = cur->GetChildIdx(p);
			++depth;
		}
		//if (isnan(result)) {
		//	printf("result is nan\n");
		//}
		return result;
	}

	RTLIB_INLINE RTLIB_HOST_DEVICE auto GetDepth(float2& p, const DTreeNode* nodes)const noexcept -> int {
		const DTreeNode* cur = this;
		int              idx = cur->GetChildIdx(p);
		int            depth = 1;
		while (!cur->IsLeaf(idx))
		{
			depth++;
			cur = &nodes[cur->GetChild(idx)];
			idx = cur->GetChildIdx(p);
		}
		return depth;
	}

	RTLIB_INLINE RTLIB_HOST_DEVICE static void MoveToLeft(float& p)noexcept {
		p *= 2.0f;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE static void MoveToRight(float& p)noexcept {
		p = 2.0f * p - 1.0f;
	}
#ifndef __CUDA_ARCH__
	auto GetArea(const std::vector<DTreeNode>& nodes)const noexcept -> float {
		float  result = 0.0f;
		for (int i = 0; i < 4; ++i) {
			if (GetSum(i) > 0.0f) {
				if (IsLeaf(i)) {
					result += 1.0f / 4.0f;
				}
				else {
					result += nodes[GetChild(i)].GetArea(nodes) / 4.0f;
				}
			}
		}
		return result;
	}
	void Build(std::vector<DTreeNode>& nodes) {
		for (int i = 0; i < 4; ++i) {
			if (this->IsLeaf(i)) {
				continue;
			}
			auto& c = nodes[children[i]];
			c.Build(nodes);
			float sum = 0.0f;
			for (int j = 0; j < 4; ++j) {
				sum += c.sums[j];
			}
			sums[i] = sum;
		}
	}
	void Dump(std::fstream& jsonFile, const std::vector<DTreeNode>& nodes)const noexcept
	{
		jsonFile << "{\n";
		jsonFile << "\"sums\"             : [" << sums[0] << ", " << sums[1] << ", " << sums[2] << ", " << sums[3] << "],\n";
		jsonFile << "\"children\"         : [\n";
		if (!IsLeaf(0)) {
			nodes[children[0]].Dump(jsonFile, nodes);
			jsonFile << ",\n";
		}
		else {
			jsonFile << "{},\n";
		}
		if (!IsLeaf(1)) {
			nodes[children[1]].Dump(jsonFile, nodes);
			jsonFile << ",\n";
		}
		else {
			jsonFile << "{},\n";
		}
		if (!IsLeaf(2)) {
			nodes[children[2]].Dump(jsonFile, nodes);
			jsonFile << ",\n";
		}
		else {
			jsonFile << "{},\n";
		}
		if (!IsLeaf(3)) {
			nodes[children[3]].Dump(jsonFile, nodes);
			jsonFile << "\n";
		}
		else {
			jsonFile << "{}\n";
		}
		jsonFile << "]\n";
		jsonFile << "}";
	}
#endif
	float          sums[4];
	unsigned short children[4];
};
struct DTree {
	template<DirectionalFilter dFilter>
	struct Impl;
	template<>
	struct Impl<DirectionalFilterNearest> {
		RTLIB_INLINE RTLIB_DEVICE static void RecordIrradiance(DTree& dTree, float2 p, float irradiance, float statisticalWeight)noexcept
		{
			if (isfinite(statisticalWeight) && statisticalWeight > 0.0f) {
				dTree.AddStatisticalWeightAtomic(statisticalWeight);
				if (isfinite(irradiance) && irradiance > 0.0f) {
					dTree.nodes[0].Record(p, irradiance * statisticalWeight, dTree.nodes);
				}
			}
		}
	};
	template<>
	struct Impl<DirectionalFilterBox> {
		RTLIB_INLINE RTLIB_DEVICE static void RecordIrradiance(DTree& dTree, float2 p, float irradiance, float statisticalWeight)noexcept
		{
			if (isfinite(statisticalWeight) && statisticalWeight > 0.0f) {
				dTree.AddStatisticalWeightAtomic(statisticalWeight);
				if (isfinite(irradiance) && irradiance > 0.0f) {
					const int depth = dTree.GetDepth(p);
					float size      = powf(0.5f, depth);
					auto  origin    = p - make_float2(size / 2.0f);
					dTree.nodes[0].Record(origin, size, make_float2(0.0f), 1.0f, irradiance * statisticalWeight / (size * size), dTree.nodes);
				}
			}
		}
	};
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetArea()const noexcept -> float {
		return area;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetMean()const noexcept -> float {
		if (statisticalWeight <= 0.0f) { return 0.0f; }
		const float factor = 1.0f / (4.0f * RTLIB_M_PI * statisticalWeight);
		return factor * sum;
	}
	RTLIB_INLINE RTLIB_DEVICE      void  AddStatisticalWeightAtomic(float val)noexcept {
#ifdef __CUDA_ARCH__
		atomicAdd(&statisticalWeight, val);
#else
		statisticalWeight += val;
#endif
	}
	RTLIB_INLINE RTLIB_DEVICE      void  AddSumAtomic(float val)noexcept {
#ifdef __CUDA_ARCH__
		atomicAdd(&sum, val);
#else
		sum += val;
#endif
	}
	template<DirectionalFilter dFilter>
	RTLIB_INLINE RTLIB_DEVICE      void  RecordIrradiance(float2 p, float irradiance, float statisticalWeight)noexcept {
		Impl<dFilter>::RecordIrradiance(*this, p, irradiance, statisticalWeight);
	}
	template<typename RNG>
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  Sample(RNG& rng)const noexcept -> float2 {
		if (GetMean() <= 0.0f) {
			return rtlib::random_float2(rng);
		}
		return rtlib::clamp(nodes[0].Sample(rng, nodes), make_float2(0.0f), make_float2(1.0f));
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  Pdf(float2 p)const noexcept -> float {
		if (GetMean() <= 0.0f) {
			return 1.0f / (4.0f * RTLIB_M_PI);
		}
		return nodes[0].Pdf(p, nodes) / (4.0f * RTLIB_M_PI);
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto GetDepth(float2 p)const noexcept -> int {
		return nodes[0].GetDepth(p, nodes);
	}
	DTreeNode* nodes;
	float      area;
	float      sum;
	float      statisticalWeight;
};
struct DTreeRecord {
	float3 direction;
	float  radiance;
	float  product;
	float  woPdf, bsdfPdf, dTreePdf;
	float  statisticalWeight;
	bool   isDelta;
};
struct DTreeWrapper {
	template<DirectionalFilter dFilter>
	RTLIB_INLINE RTLIB_DEVICE      void  Record(const DTreeRecord& rec) noexcept {
		if (!rec.isDelta) {
			float irradiance = rec.radiance / rec.woPdf;
			building.RecordIrradiance<dFilter>(rtlib::dir_to_canonical(rec.direction), irradiance, rec.statisticalWeight);
		}
	}
	template<typename RNG>
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  Sample(RNG& rng)const noexcept -> float3 {
		return rtlib::canonical_to_dir(sampling.Sample(rng));
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  Pdf(const float3& dir)const noexcept -> float {
		return sampling.Pdf(rtlib::dir_to_canonical(dir));
	}
	DTree building;
	DTree sampling;
};
struct STreeNode {
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetNodeIdx(float3& p)const noexcept -> unsigned int {
		return children[GetChildIdx(p)];
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetChildIdx(float3& p)const noexcept -> int {
		float* p_A = reinterpret_cast<float*>(&p);
		p_A[axis] *= 2.0f;
		int    idx = p_A[axis];
		p_A[axis] -= static_cast<float>(idx);
		return idx;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE bool  IsLeaf()const noexcept {
		return dTree != nullptr;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper(float3& p, float3& size, const STreeNode* nodes)const noexcept -> DTreeWrapper* {
		const STreeNode* cur = this;
		int              idx = cur->GetChildIdx(p);
		int            depth = 1;
		unsigned int  t_axis = this->axis;
		while (!cur->IsLeaf()) {
			reinterpret_cast<float*>(&size)[t_axis] /= 2.0f;
			cur  = &nodes[cur->children[idx]];
			idx  = cur->GetChildIdx(p);
			t_axis = (t_axis + 1) % 3;
			++depth;
		}
		return cur->dTree;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper()const noexcept -> DTreeWrapper* {
		return dTree;
	}
	template<DirectionalFilter dFilter>
	RTLIB_INLINE RTLIB_DEVICE      void  Record(const float3& min1, const float3& max1, float3 min2, float3 size2, const DTreeRecord& rec, STreeNode* nodes) noexcept {
		struct StackNode {
			STreeNode* curNode;
			float3     min2;
			float3     size2;
		};
		StackNode stackNodes[PATH_GUIDING_STREE_MAX_DEPTH] = {};
		int       stackNodeSize     = 1;
		const int stackNodeCapacity = PATH_GUIDING_STREE_MAX_DEPTH;
		stackNodes[0].curNode       = this;
		stackNodes[0].min2          = min2;
		stackNodes[0].size2         = size2;
		while (stackNodeSize > 0) {
			auto  top = stackNodes[stackNodeSize - 1];
			stackNodeSize--;
			float w   = ComputeOverlappingVolume(min1, max1, top.min2, top.min2 + top.size2);
			if (w > 0.0f) {
				if (top.curNode->IsLeaf()) {
					top.curNode->dTree->Record<dFilter>({ rec.direction,rec.radiance,rec.product,rec.woPdf,rec.bsdfPdf,rec.dTreePdf,rec.statisticalWeight * w,rec.isDelta });
				}
				else if (stackNodeSize < stackNodeCapacity ) {
					float3 t_size2 = top.size2;
					int    t_axis  = top.curNode->axis;
					reinterpret_cast<float*>(&t_size2)[t_axis] /= 2.0f;
					for (int i = 0; i < 2; ++i) {
						float3 t_min2 = top.min2;
						if (i & 1) {
							reinterpret_cast<float*>(&t_min2)[t_axis] += reinterpret_cast<float*>(&t_size2)[t_axis];
						}
						stackNodes[stackNodeSize  + i].curNode = &nodes[top.curNode->children[i]];
						stackNodes[stackNodeSize  + i].min2    = t_min2;
						stackNodes[stackNodeSize  + i].size2   = t_size2;
					}
					stackNodeSize += 2;
				}
			}
		}
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE static auto ComputeOverlappingVolume(const float3& min1, const float3& max1, const float3& min2, const float3& max2)noexcept->float {
		float lengths[3] = {
			fmaxf(fminf(max1.x,max2.x) - fmaxf(min1.x,min2.x),0.0f),
			fmaxf(fminf(max1.y,max2.y) - fmaxf(min1.y,min2.y),0.0f),
			fmaxf(fminf(max1.z,max2.z) - fmaxf(min1.z,min2.z),0.0f)
		};
		return lengths[0] * lengths[1] * lengths[2];
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE static void MoveToLeft(float& p)noexcept {
		p *= 2.0f;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE static void MoveToRight(float& p)noexcept {
		p = 2.0f * p - 1.0f;
	}
	unsigned char axis;
	unsigned int  children[2];
	DTreeWrapper* dTree;
};
struct STree {
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper(float3 p, float3& size)const noexcept -> DTreeWrapper* {
		size = aabbMax - aabbMin;
		p    = p - aabbMin;
		p   /= size;
		return nodes[0].GetDTreeWrapper(p, size, nodes);
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper(float3 p)const noexcept -> DTreeWrapper* {
		float3 size;
		return GetDTreeWrapper(p, size);
	}
	template<DirectionalFilter dFilter>
	RTLIB_INLINE RTLIB_DEVICE      void  Record(const float3& p, const float3& dVoxelSize, DTreeRecord rec)
	{
		float volume           = dVoxelSize.x * dVoxelSize.y * dVoxelSize.z;
		rec.statisticalWeight /= volume;
		nodes[0].Record<dFilter>(p - dVoxelSize * 0.5f, p + dVoxelSize * 0.5f, aabbMin, aabbMax - aabbMin, rec, nodes);
	}
	STreeNode* nodes;
	float3     aabbMin;
	float3     aabbMax;
	float      fraction;
};
struct STreeNode2 {
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetNodeIdx(float3& p)const noexcept -> unsigned int {
		return children[GetChildIdx(p)];
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetChildIdx(float3& p)const noexcept -> int {
		int i_x = 2.0f * p.x;
		int i_y = 2.0f * p.y;
		int i_z = 2.0f * p.z;
		p *= 2.0f;
		p -= make_float3((float)i_x, (float)i_y, (float)i_z);
		return i_x + (1 << 1) * i_y + (1 << 2) * i_z;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE bool  IsLeaf()const noexcept {
		return dTree != nullptr;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper(float3& p, float3& size, const STreeNode2* nodes)const noexcept -> DTreeWrapper* {
		const STreeNode2* cur = this;
		int               idx = cur->GetChildIdx(p);
		while (!cur->IsLeaf()) {
			size  /= 2.0f;
			cur    = &nodes[cur->children[idx]];
			idx    = cur->GetChildIdx(p);
		}
		return cur->dTree;
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper()const noexcept -> DTreeWrapper* {
		return dTree;
	}
	template<DirectionalFilter dFilter>
	RTLIB_INLINE RTLIB_DEVICE      void  Record(const float3& min1, const float3& max1, float3 min2, float3 size2, const DTreeRecord& rec, STreeNode* nodes) noexcept {
		struct StackNode {
			STreeNode* curNode;
			float3     min2;
			float3     size2;
		};
		StackNode stackNodes[7*PATH_GUIDING_STREE_MAX_DEPTH] = {};
		int       stackNodeSize = 1;
		const int stackNodeCapacity = 7*PATH_GUIDING_STREE_MAX_DEPTH;
		stackNodes[0].curNode = this;
		stackNodes[0].min2    = min2;
		stackNodes[0].size2   = size2;
		while (stackNodeSize > 0) {
			auto  top = stackNodes[stackNodeSize - 1];
			stackNodeSize--;
			float w = ComputeOverlappingVolume(min1, max1, top.min2, top.min2 + top.size2);
			if (w > 0.0f) {
				if (top.curNode->IsLeaf()) {
					top.curNode->dTree->Record<dFilter>({ rec.direction,rec.radiance,rec.product,rec.woPdf,rec.bsdfPdf,rec.dTreePdf,rec.statisticalWeight * w,rec.isDelta });
				}
				else if (stackNodeSize < stackNodeCapacity) {
					float3 t_size2 = top.size2;
					t_size2       /= 2.0f;
					for (int i = 0; i < 8; ++i) {
						float3 t_min2 = top.min2;
						if (i & 1) {
							t_min2.x += t_size2.x;
						}
						if (i & 2) {
							t_min2.y += t_size2.y;
						}

						if (i & 4) {
							t_min2.z += t_size2.z;
						}
						stackNodes[stackNodeSize + i].curNode = &nodes[top.curNode->children[i]];
						stackNodes[stackNodeSize + i].min2    = t_min2;
						stackNodes[stackNodeSize + i].size2   = t_size2;
					}
					stackNodeSize += 8;
				}
			}
		}
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE static auto ComputeOverlappingVolume(const float3& min1, const float3& max1, const float3& min2, const float3& max2)noexcept->float {
		float lengths[3] = {
			fmaxf(fminf(max1.x,max2.x) - fmaxf(min1.x,min2.x),0.0f),
			fmaxf(fminf(max1.y,max2.y) - fmaxf(min1.y,min2.y),0.0f),
			fmaxf(fminf(max1.z,max2.z) - fmaxf(min1.z,min2.z),0.0f)
		};
		return lengths[0] * lengths[1] * lengths[2];
	}
	unsigned int  children[8];
	DTreeWrapper* dTree;
};
struct STree2 {
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper(float3 p, float3& size)const noexcept -> DTreeWrapper* {
		size = aabbMax - aabbMin;
		p = p - aabbMin;
		p /= size;
		return nodes[0].GetDTreeWrapper(p, size, nodes);
	}
	RTLIB_INLINE RTLIB_HOST_DEVICE auto  GetDTreeWrapper(float3 p)const noexcept -> DTreeWrapper* {
		float3 size;
		return GetDTreeWrapper(p, size);
	}
	template<DirectionalFilter dFilter>
	RTLIB_INLINE RTLIB_DEVICE      void  Record(const float3& p, const float3& dVoxelSize, DTreeRecord rec)
	{
		float volume = dVoxelSize.x * dVoxelSize.y * dVoxelSize.z;
		rec.statisticalWeight /= volume;
		nodes[0].Record<dFilter>(p - dVoxelSize * 0.5f, p + dVoxelSize * 0.5f, aabbMin, aabbMax - aabbMin, rec, nodes);
	}
	STreeNode2* nodes;
	float3      aabbMin;
	float3      aabbMax;
	float       fraction;
};
struct TraceVertex {
	template<SpatialFilter sFilter>
	struct Impl;
	
	template<>
	struct Impl<SpatialFilter::SpatialFilterNearest>
	{
		template<DirectionalFilter dFilter, typename STreeType>
		RTLIB_INLINE RTLIB_HOST_DEVICE static void Record(TraceVertex& v, STreeType& sTree, const DTreeRecord& rec)
		{
			v.dTree->Record<dFilter>(rec);
		}
	};
	
	template<>
	struct Impl<SpatialFilter::SpatialFilterBox>
	{
		template<DirectionalFilter dFilter, typename STreeType>
		RTLIB_INLINE RTLIB_HOST_DEVICE static void Record(TraceVertex& v, STreeType& sTree, const DTreeRecord& rec)
		{
			sTree.Record<dFilter>(v.rayOrigin,v.dTreeVoxelSize, rec);
		}
	};

	DTreeWrapper* dTree;
	float3        dTreeVoxelSize;
	float3        rayOrigin;
	float3        rayDirection;
	float3        throughPut;
	float3        bsdfVal;
	float3        radiance;
	float         woPdf;
	float         bsdfPdf;
	float         dTreePdf;
	float         cosine;
	bool          isDelta;
	RTLIB_INLINE RTLIB_HOST_DEVICE void Record(const float3& r) noexcept {
		radiance += r;
	}
	template<SpatialFilter sFilter, DirectionalFilter dFilter, typename STreeType>
	RTLIB_INLINE RTLIB_HOST_DEVICE void Commit(STreeType& sTree,float statisticalWeight)noexcept
	{
		if (!dTree) {
			return;
		}
		bool isValidRadiance = (isfinite(radiance.x) && radiance.x >= 0.0f) &&
			(isfinite(radiance.y) && radiance.y >= 0.0f) &&
			(isfinite(radiance.z) && radiance.z >= 0.0f);
		bool isValidBsdfVal  = (isfinite(bsdfVal.x) && bsdfVal.x >= 0.0f) &&
			(isfinite(bsdfVal.y) && bsdfVal.y >= 0.0f) &&
			(isfinite(bsdfVal.z) && bsdfVal.z >= 0.0f);
		if (woPdf <= 0.0f || !isValidRadiance || !isValidBsdfVal)
		{
			return;
		}
		auto localRadiance = make_float3(0.0f);
		if (throughPut.x * woPdf > 1e-4f) {
			localRadiance.x = radiance.x / throughPut.x;
		}
		if (throughPut.y * woPdf > 1e-4f) {
			localRadiance.y = radiance.y / throughPut.y;
		}
		if (throughPut.z * woPdf > 1e-4f) {
			localRadiance.z = radiance.z / throughPut.z;
		}
#if PATH_GUIDING_LI_COSINE
		localRadiance *= fabsf(cosine);
#endif
		//printf("localRadiance=(%f,%f,%f)\n",localRadiance.x,localRadiance.y,localRadiance.z);
		float3 product         =  localRadiance   * bsdfVal;
		float localRadianceAvg = (localRadiance.x + localRadiance.y + localRadiance.z) / 3.0f;
		float productAvg       = (product.x + product.y + product.z) / 3.0f;
		DTreeRecord rec{ rayDirection,localRadianceAvg ,productAvg,woPdf,bsdfPdf,dTreePdf,statisticalWeight,isDelta };
		Impl<sFilter>::Record<dFilter>(*this, sTree, rec);
	}
};
#endif