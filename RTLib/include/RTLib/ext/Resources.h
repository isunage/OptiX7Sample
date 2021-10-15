#ifndef RTLIB_EXT_RESOURCES_H
#define RTLIB_EXT_RESOURCES_H
#include <memory>
#include <unordered_map>
#include <vector>
namespace rtlib
{
    namespace ext
    {
        //vertexBuffer
        class BufferComponent
        {
        public:
            virtual void Init(const void* cpuData, size_t sizeInBytes) = 0;
            virtual void Allocate(size_t sizeInBytes) = 0;
            virtual bool Resize(size_t sizeInBytes) = 0;
            virtual void Upload(const void* cpuData, size_t sizeInBytes) = 0;
            virtual void Reset() = 0;
            virtual ~BufferComponent()noexcept {}
        };
        using BufferComponentPtr = std::shared_ptr<BufferComponent>;
        template<typename T>
        class CustomBuffer
        {
        public:
            CustomBuffer()noexcept {}
            CustomBuffer(CustomBuffer&& vb)noexcept
            {
                m_CpuData = std::move(vb.m_CpuData);
                m_GpuComponents = std::move(vb.m_GpuComponents);
            }
            CustomBuffer(const CustomBuffer&) = delete;
            CustomBuffer& operator=(CustomBuffer&& vb)noexcept
            {
                // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
                if (this != &vb) {
                    Reset();
                    m_CpuData = std::move(vb.m_CpuData);
                    m_GpuComponents = std::move(vb.m_GpuComponents);
                }
                return *this;
            }
            CustomBuffer& operator=(const CustomBuffer&) = delete;
            CustomBuffer(std::vector<T>&& vec)noexcept
            {
                m_CpuData = std::move(vec);
            }
            CustomBuffer& operator=(std::vector<T>&& vb)noexcept
            {
                // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
                Reset();
                m_CpuData = std::move(vb);
                return *this;
            }
            auto operator[](size_t idx)noexcept->T& { return m_CpuData[idx]; }
            auto operator[](size_t idx)const noexcept->const T& { return m_CpuData[idx]; }
            auto At(size_t idx) -> T& { return m_CpuData.at(idx); }
            auto At(size_t idx)const ->const T& { return m_CpuData.at(idx); }
            auto GetCpuData()const noexcept -> const T* { return m_CpuData.data(); }
            auto GetCpuData() noexcept ->  T* { return m_CpuData.data() ; }
            void PushBack(T&& val)noexcept { m_CpuData.push_back(val); }
            void PushBack(const T& val)noexcept { m_CpuData.push_back(val); }
            auto Size()const noexcept -> size_t { return m_CpuData.size();}
            bool Empty()const noexcept { return m_CpuData.empty(); }
            auto begin()const noexcept { return m_CpuData.begin(); }
            auto begin() { return m_CpuData.begin(); }
            auto end()const noexcept { return m_CpuData.end(); }
            auto end() { return m_CpuData.end(); }
            //Gpu
            void Allocate()
            {
                for (auto& [name, component] : m_GpuComponents)
                {
                    component->Allocate(m_CpuData.size() * sizeof(T));
                }
            }
            void Upload()
            {
                for (auto& [name, component] : m_GpuComponents)
                {
                    component->Upload(m_CpuData.data(), m_CpuData.size() * sizeof(T));
                }
            }
            bool Resize(size_t count)
            {
                if (count == m_CpuData.size()) { return false; }
                m_CpuData.resize(count);
                for (auto& [name, component] : m_GpuComponents)
                {
                    component->Resize(count * sizeof(T));
                    component->Upload(m_CpuData.data(), m_CpuData.size() * sizeof(T));
                }
                return true;
            }
            void Reset()
            {
                m_CpuData.clear();
                for (auto& [name, component] : m_GpuComponents)
                {
                    component->Reset();
                }
            }
            template<typename S, bool sharedRes = std::is_base_of_v<BufferComponent, S>>
            void AddGpuComponent(const std::string& key)
            {
                m_GpuComponents[key] = std::shared_ptr<S>(new S());
                m_GpuComponents[key]->Init(m_CpuData.data(),m_CpuData.size()*sizeof(T));
            }
            template<typename S, bool sharedRes = std::is_base_of_v<BufferComponent, S>>
            auto GetGpuComponent(const std::string& key)const -> std::shared_ptr<S> {
                return std::static_pointer_cast<S>(m_GpuComponents.at(key));
            }
            template<typename S, bool sharedRes = std::is_base_of_v<BufferComponent, S>>
            auto PopGpuComponent(const std::string& key) -> std::shared_ptr<S> {
                if (m_GpuComponents.count(key) == 0) {
                    return nullptr;
                }
                auto ptr = std::static_pointer_cast<S>(m_GpuComponents.at(key));
                m_GpuComponents.erase(key);
                return ptr;
            }
            bool HasGpuComponent(const std::string& key)const noexcept { return m_GpuComponents.count(key) > 0; }
            ~CustomBuffer()noexcept {}
        private:
            std::vector<T>                                      m_CpuData       = {};
            std::unordered_map<std::string, BufferComponentPtr> m_GpuComponents = {};
        };
    }
}
#endif
