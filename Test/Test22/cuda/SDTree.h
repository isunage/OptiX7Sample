#ifndef SDTREE_H
#define SDTREE_H
#include <RTLib/Preprocessors.h>
#include <RTLib/VectorFunction.h>
namespace sdtree_utils
{
    //StackObject
    template<typename T, unsigned int N>
    class Stack{
    public:
        RTLIB_INLINE RTLIB_HOST_DEVICE      Stack()noexcept:m_Data{},m_Len{}{}
        RTLIB_INLINE RTLIB_HOST_DEVICE void Push(const T& val)noexcept{
            if(m_Len<N){
                //データの更新を行う
                m_Data[m_Len] = val;
                m_Len++;
            }
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto Top()const noexcept -> T{
            if(m_Len>0){
                return m_Data[m_Len];
            }
            return T{};
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto Pop()noexcept{
            if(m_Len>0){
                //データの削除はしない
                m_Len--;
            }
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto GetSize()const noexcept -> unsigned int{
            return m_Len;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE auto GetCapacity()const noexcept -> unsigned int{
            return N;
        }
        RTLIB_INLINE RTLIB_HOST_DEVICE bool IsEmpty()const noexcept{
            return m_Len == 0;
        }
    private:
        T            m_Data[N];
        unsigned int m_Len;
    };
}
class DTreeNode
{
public:
    RTLIB_INLINE RTLIB_HOST_DEVICE      DTreeNode()noexcept:m_Sums{},m_Children{}{}
    //Sums
    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetSum(int idx)const noexcept -> float
    {
        return m_Sums[idx];
    }
    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetSumOfAll()  const noexcept -> float{
        return m_Sums[0]+m_Sums[1];
    }
    RTLIB_INLINE RTLIB_HOST_DEVICE void SetSum(int idx, float val)noexcept
    {
        m_Sums[idx] = val;
    }
    RTLIB_INLINE RTLIB_HOST_DEVICE void SetSum(float val)noexcept 
    {
        m_Sums[0] = val;
        m_Sums[1] = val;
        m_Sums[2] = val;
        m_Sums[3] = val;
    }
    RTLIB_INLINE RTLIB_HOST_DEVICE auto AddSum(int idx, float val)noexcept{
        m_Sums[idx] += val;
    }
    RTLIB_INLINE RTLIB_DEVICE      auto AddSumAtomic(int idx, float val)noexcept
    {
        (void)sdtree_utils::AtomicAddFloat(m_Sums[idx],val);
    }
    //Child
    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetChild(int idx)const noexcept -> unsigned short
    {
        return m_Children[idx];
    }
    RTLIB_INLINE RTLIB_HOST_DEVICE void SetChild(int idx, unsigned short val)noexcept
    {
        m_Children[idx] = val;
    }
    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetChildIdx(float2& p)const noexcept -> int{
        int idx = 0;
        if (p.x<0.5f){
            p.x*=2.0f;
        }else{
            p.x =2.0f*p.x-1.0f;
            idx|=(1<<0);
        }
        if (p.y<0.5f){
            p.y*=2.0f;
        }else{
            p.y =2.0f*p.y-1.0f;
            idx|=(1<<1);
        }
        return idx;
    }
    //Record
    RTLIB_INLINE RTLIB_DEVICE      void Record(float2& p, float irradiance, DTreeNode* nodes)noexcept
    {
        DTreeNode* cur = this;
        int idx        = cur->GetChildIdx(p);
        for(;;;){
            if(cur->IsLeaf(idx))
            {
                cur->AddSumAtomic(idx,irradiance);
                break;
            }
            cur = &nodes[cur->GetChild(idx)];
            idx = cur->GetChildIdx(p);
        }
    }
    template<typename RNG>
    RTLIB_INLINE RTLIB_HOST_DEVICE auto Sample(RNG& rng, const DTreeNode* nodes)const noexcept -> float2{
        DTreeNode* cur = this;
        auto       res = make_float2(0.0f);
        auto       off = make_float2(0.0f);
        auto       siz = 1.0f;
        for(;;;)
        {
            int  idx  = 0;
            auto topL = sums[0];
            auto topR = sums[1];
            auto part = topL+topR;
            auto total= cur->GetSumOfAll();
        }
    }
private:
    float          m_Sums[4];
    unsigned short m_Children[4];
};
#endif