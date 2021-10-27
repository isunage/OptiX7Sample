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
            virtual void     Init(const void* cpuData, size_t sizeInBytes) = 0;
            virtual void Allocate(size_t sizeInBytes) = 0;
            virtual bool   Resize(size_t sizeInBytes) = 0;
            virtual void   Upload(const void* cpuData, size_t sizeInBytes) = 0;
            virtual void    Reset() = 0;
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
        class Image2DComponent
        {
        public:
            virtual void     Init(const void* cpuData, size_t rowPitch, size_t slicePitch) = 0;
            virtual void Allocate(size_t rowPitch, size_t slicePitch) = 0;
            virtual bool   Resize(size_t rowPitch, size_t slicePitch) = 0;
            virtual void   Upload(const void* cpuData, size_t rowPitch, size_t slicePitch) = 0;
            virtual void    Reset() = 0;
            virtual ~Image2DComponent()noexcept {}
        };
        using Image2DComponentPtr = std::shared_ptr<Image2DComponent>;
        template<typename T>
        class CustomImage2D
        {
        public:
            CustomImage2D()noexcept {}
            CustomImage2D(CustomImage2D&& img)noexcept
            {
                m_Width         = img.m_Width;
                m_Height        = img.m_Height;
                m_CpuData       = std::move(img.m_CpuData);
                m_GpuComponents = std::move(img.m_GpuComponents);
                img.m_Width  = 0;
                img.m_Height = 0;
            }
            CustomImage2D(const CustomImage2D&) = delete;
            CustomImage2D& operator=(CustomImage2D&& img)noexcept
            {
                // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
                if (this != &img) {
                    Reset();
                    m_Width         = img.m_Width;
                    m_Height        = img.m_Height;
                    m_CpuData       = std::move(img.m_CpuData);
                    m_GpuComponents = std::move(img.m_GpuComponents);
                    img.m_Width  = 0;
                    img.m_Height = 0;
                }
                return *this;
            }
            CustomImage2D& operator=(const CustomImage2D&) = delete;
            CustomImage2D(size_t width, size_t height, const std::vector<T>& data)
            {
                m_Width  = width;
                m_Height = height;
                m_CpuData.resize(m_Width * m_Height);
                for (auto h = 0; h < m_Height; ++h)
                {
                    for (auto w = 0; w < m_Width; ++w)
                    {
                        m_CpuData[GetIndex(w, h)] = data[GetIndex(w, h)];
                    }
                }
            }
            auto operator()(size_t x, size_t y)noexcept->T& { return m_CpuData[GetIndex(x,y)]; }
            auto operator()(size_t x, size_t y)const noexcept->const T& { return m_CpuData[GetIndex(x, y)]; }
            auto At(size_t x, size_t y) -> T& { return m_CpuData.at(GetIndex(x,y)); }
            auto At(size_t x, size_t y)const ->const T& { return m_CpuData.at(GetIndex(x, y)); }
            auto GetCpuData()const noexcept -> const T* { return m_CpuData.data(); }
            auto GetCpuData() noexcept  -> T* { return m_CpuData.data(); }
            auto Width()const noexcept  -> size_t { return m_Width; }
            auto Height()const noexcept -> size_t { return m_Height; }
            bool Empty()const noexcept { return m_CpuData.empty(); }
            auto begin()const noexcept { return m_CpuData.begin(); }
            auto begin() { return m_CpuData.begin(); }
            auto end()const noexcept { return m_CpuData.end(); }
            auto end()   { return m_CpuData.end();   }
            //Gpu
            void Allocate()
            {
                for (auto& [name, component] : m_GpuComponents)
                {
                    component->Allocate(m_Width*sizeof(T), m_Width * m_Height * sizeof(T));
                }
            }
            void Upload()
            {
                for (auto& [name, component] : m_GpuComponents)
                {
                    component->Upload(m_CpuData.data(), m_Width * sizeof(T), m_Width * m_Height * sizeof(T));
                }
            }
            bool Resize(size_t w, size_t h)
            {
                if (w == m_Width && h == m_Height) { return false; }
                std::vector<T> newCpuData(w * h);
                for (size_t j = 0; j < std::min(h, m_Height); ++j)
                {
                    for (size_t i = 0; i < std::min(w, m_Width); ++i)
                    {
                        newCpuData[w * j + i] = m_CpuData[m_Width * j + i];
                    }
                }
                m_Width   = w;
                m_Height  = h;
                m_CpuData = std::move(newCpuData);
                for (auto& [name, component] : m_GpuComponents)
                {
                    component->Resize(m_Width * sizeof(T), m_Width * m_Height * sizeof(T));
                    component->Upload(m_CpuData.data(), m_Width * sizeof(T), m_Width * m_Height * sizeof(T));
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
            template<typename S, bool sharedRes = std::is_base_of_v<Image2DComponent, S>>
            void AddGpuComponent(const std::string& key)
            {
                m_GpuComponents[key] = std::shared_ptr<S>(new S());
                m_GpuComponents[key]->Init(m_CpuData.data(), m_Width * sizeof(T), m_Width * m_Height * sizeof(T));
            }
            template<typename S, bool sharedRes = std::is_base_of_v<Image2DComponent, S>>
            auto GetGpuComponent(const std::string& key)const -> std::shared_ptr<S> {
                return std::static_pointer_cast<S>(m_GpuComponents.at(key));
            }
            template<typename S, bool sharedRes = std::is_base_of_v<Image2DComponent, S>>
            auto PopGpuComponent(const std::string& key) -> std::shared_ptr<S> {
                if (m_GpuComponents.count(key) == 0) {
                    return nullptr;
                }
                auto ptr = std::static_pointer_cast<S>(m_GpuComponents.at(key));
                m_GpuComponents.erase(key);
                return ptr;
            }
            bool HasGpuComponent(const std::string& key)const noexcept { return m_GpuComponents.count(key) > 0; }
        private:
            auto GetIndex(size_t x, size_t y)const noexcept -> size_t
            {
                return m_Width * y + x;
            }
        private:
            size_t                                               m_Width         = 0;
            size_t                                               m_Height        = 0;
            std::vector<T>                                       m_CpuData       = {};
            std::unordered_map<std::string, Image2DComponentPtr> m_GpuComponents = {};
        };
    }
}
#endif
