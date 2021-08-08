#ifndef RTLIB_GL_H
#define RTLIB_GL_H
#include <glad/glad.h>
#include <string>
#include <vector>
#include <array>
#include "Preprocessors.h"
#include "PixelFormat.h"
namespace rtlib{
    //GL Buffer
    template<typename T>
    class GLBuffer{
        GLuint m_ID     = 0;
        size_t m_Count  = 0;
        GLenum m_Target;
        GLenum m_Usage;
    private:
        void clear()noexcept{
            m_ID     = 0;
            m_Count  = 0;
            m_Target = GL_ARRAY_BUFFER;
            m_Usage  = GL_STATIC_DRAW;
        }
        void upload_unsafe(const T* hostPtr,size_t count){
            glBufferSubData(m_Target,0,sizeof(T)*count,(const void*)hostPtr);
        }
        void download_unsafe(T* hostPtr,size_t count){
            glGetBufferSubData(m_Target,0,sizeof(T)*count,(void*)hostPtr);
        }
    public:
        //constructor,copy,move 
        GLBuffer()noexcept:m_Target{GL_ARRAY_BUFFER},m_Usage{GL_STATIC_DRAW}{}
        GLBuffer(const GLBuffer&)noexcept = delete;
        GLBuffer(GLBuffer&& buffer)noexcept{
            this->m_ID     = buffer.m_ID;
            this->m_Count  = buffer.m_Count;
            this->m_Target = buffer.m_Target;
            this->m_Usage  = buffer.m_Usage;
            buffer.clear();
        }
        GLBuffer& operator=(const GLBuffer& )noexcept = delete;
        GLBuffer& operator=(GLBuffer&& buffer){
            if(this!=&buffer){
                this->reset();
                this->m_ID     = buffer.m_ID;
                this->m_Count  = buffer.m_Count;
                this->m_Target = buffer.m_Target;
                this->m_Usage  = buffer.m_Usage;
                buffer.clear();
            }
            return *this;
        }
        explicit GLBuffer(GLenum target,GLenum usage)noexcept:m_Target{target},m_Usage{usage}{}
        //user constructor
        explicit GLBuffer(const T* hostPtr,size_t count, GLenum target = GL_ARRAY_BUFFER, GLenum usage = GL_STATIC_DRAW)
        :GLBuffer(target,usage){
            this->allocate(count);
            this->upload_unsafe(hostPtr,count);
        }
        explicit GLBuffer(const T& hostData,GLenum target, GLenum usage)
        :GLBuffer(&hostData,1,target,usage){}
        template<size_t N>
        explicit GLBuffer(const T (&hostData)[N],GLenum target, GLenum usage)
        :GLBuffer(std::data(hostData),std::size(hostData),target,usage){}
        template<size_t N>
        explicit GLBuffer(const std::array<T,N>& hostData,GLenum target, GLenum usage)
        :GLBuffer(std::data(hostData),std::size(hostData),target,usage){}
        explicit GLBuffer(const std::vector<T>& hostData, GLenum target, GLenum GLenum) 
        :GLBuffer(std::data(hostData),std::size(hostData),target,usage){}
        //bool
        explicit operator bool()const noexcept{
            return m_ID!=0;
        }
        //Get And Set
        RTLIB_DECLARE_GET_AND_SET_BY_VALUE(GLBuffer,GLenum,Target, m_Target);
        RTLIB_DECLARE_GET_AND_SET_BY_VALUE(GLBuffer,GLenum, Usage, m_Usage);
        //Get
        RTLIB_DECLARE_GET_BY_VALUE(GLBuffer,GLuint,ID,m_ID);
        RTLIB_DECLARE_GET_BY_VALUE(GLBuffer,size_t,Count, m_Count);
        RTLIB_DECLARE_GET_BY_VALUE(GLBuffer,size_t,SizeInBytes, m_Count*sizeof(T));
        //まだallocateされていないことを前提にする
        void  allocate(size_t count){
            glGenBuffers(1,&m_ID);
            this->bind();
            glBufferData(m_Target,sizeof(T)*count,nullptr,m_Usage);
            m_Count = count;
        }
        bool  resize(size_t count){
            if(m_Count!=count){
                this->reset();
                this->allocate(count);
                return true;
            }
            return false;
        }
        //upload
        void  upload(const T* hostPtr,size_t count,bool isBinded = true){
            if(isBinded){
                this->bind();
            }
            this->upload_unsafe(hostPtr,std::min(count,m_Count));
        }
        void  upload(const std::vector<T>& hostArray, bool isBinded = true) {
            if (isBinded) {
                this->bind();
            }
            this->upload_unsafe(hostArray.data(), std::min(hostArray.size(), m_Count));
        }
        //download
        void  download(    T* hostPtr,size_t count,bool isBinded = true){
            if(isBinded){
                this->bind();
            }
            count = std::min(count,m_Count);
            this->download_unsafe(hostPtr,std::min(count,m_Count));
        }
        void  download(std::vector<T>&    hostData,bool isBinded = true){
            if(isBinded){
                this->bind();
            }
            hostData.resize(m_Count);
            this->download_unsafe(hostData.data(),hostData.size());
        }
        //bind and unbind
        void  bind()const{
            glBindBuffer(m_Target,m_ID);
        }
        void  unbind()const{
            glBindBuffer(m_Target,0);
        }
        void  reset(bool isUnBinded = true){
            if(m_ID > 0){
                if(isUnBinded){
                    this->unbind();
                }
                glDeleteBuffers(1,&m_ID);
                this->clear ();
            }
        }
        ~GLBuffer()noexcept{
            //std::cout << "Destroy GL Buffer" << std::endl;
            try{
                this->reset();
            }catch(...){
                /*no operation*/
            }
        }
    };
    //GL Texture
    template<typename PixelType>
    class GLTexture2D{
    public:
        struct View{
            size_t       width  = 1;
            size_t       height = 1;
            const void*  data   = nullptr;
        };
        using Views = std::vector<GLTexture2D::View>;
    private:
        GLuint        m_ID             = 0;
        size_t        m_NumLevels      = 0;
        GLenum        m_Target         = GL_TEXTURE_2D;
        GLenum        m_InternalFormat = GLPixelTraits<PixelType>::internalFormat;
        Views         m_Views          = {};
    public:
        GLTexture2D()noexcept{}
        GLTexture2D(const GLTexture2D&)noexcept = delete;
        GLTexture2D(GLTexture2D&& tex)noexcept{
            std::swap(m_ID    ,tex.m_ID);
            std::swap(m_Width ,tex.m_Width);
            std::swap(m_Height,tex.m_Height);
            std::swap(m_Target,tex.m_Target);
            std::swap(m_Views ,tex.m_Views);
        }
        GLTexture2D& operator=(const GLTexture2D& )noexcept = delete;
        GLTexture2D& operator=(GLTexture2D&& tex)noexcept{
            if(this!=&tex){
                this->reset();
                std::swap(m_ID    ,tex.m_ID);
                std::swap(m_Width ,tex.m_Width);
                std::swap(m_Height,tex.m_Height);
                std::swap(m_Target,tex.m_Target);
                std::swap(m_Views ,tex.m_Views);
            }
            return *this;
        }
        explicit operator bool()const noexcept{
            return m_ID!=0;
        }
        RTLIB_DECLARE_GET_AND_SET_BY_VALUE(GLTexture2D,GLenum,Target,m_Target);
        RTLIB_DECLARE_GET_BY_VALUE(GLTexture2D, GLuint,ID,m_ID);
        RTLIB_DECLARE_GET_BY_VALUE(GLTexture2D,size_t,NumLevels,m_NumLevels);
        RTLIB_DECLARE_GET_BY_REFERENCE(GLTexture2D, Views, Views, m_Views);
        //allocate
        void allocate(const View*  views, size_t numLevels,GLenum target = GL_TEXTURE_2D,bool useSRGB = false){
            m_Views     = Views(views,views+numLevels);
            m_Target    = target;
            if(useSRGB){
                m_InternalFormat = GLPixelTraits<PixelType>::internalFormatSRGB;
            }
            constexpr size_t pixelSizeInBytes = sizeof(GLPixelTraits<PixelType>::base_type)*GLPixelTraits<PixelType>::numChannels;
            size_t unpackAlignments = 1;
            if(pixelSizeInBytes%8==0){
                unpackAlignments = 8;
            }else if(pixelSizeInBytes%4==0){
                unpackAlignments = 4;
            }else if(pixelSizeInBytes%2==0){
                unpackAlignments = 2;
            }
            //TODO Pixel Storei
            glGenTextures(1,&m_ID);
            this->bind();
            size_t level = 0;
            glPixelStorei(GL_UNPACK_ALIGNMENT,unpackAlignments);
            for(const auto& view:m_Views){
                glTexImage2D(m_Target,
                level,
                m_InternalFormat,
                view.width,
                view.height,0,
                GLPixelTraits<PixelType>::format,
                GLPixelTraits<PixelType>::type,
                view.data);
                ++level;
            }
            m_NumLevels = level;
            glTexParameteri(m_Target,GL_TEXTURE_MAX_LEVEL, m_NumLevels);
        }
        void allocate(const Views& views, GLenum target = GL_TEXTURE_2D, bool useSRGB = false){
            this->allocate(views.data(),views.size(),target,useSRGB);
        }
        void allocate(const View&  view , GLenum target = GL_TEXTURE_2D, bool useSRGB = false){
            this->allocate(&view,1,target,useSRGB);
        }
        void allocateWithMipLevel(const View& view, size_t numMipLevels, GLenum target = GL_TEXTURE_2D, bool useSRGB = false) {
            Views views(numMipLevels);
            size_t t_Width  = view.width;
            size_t t_Height = view.height;
            for (auto& t_view : views) {
                t_view.width  = t_Width;
                t_view.height = t_Height;
                t_view.data   = nullptr;
                t_Width  /= 2;
                t_Height /= 2;

            }
            views[0].data = view.data;
            this->allocate(views.data(), views.size(), target, useSRGB);
        }
        //resize->複数のミップマップを仮定しているので定義できない
        //upload
        bool upload(  size_t level, const void* data, size_t w_offset, size_t h_offset,size_t w_size, size_t h_size, bool isBinded = true){
            if(level>=m_Views.size()){
                return false;
            }
            w_offset = std::min(w_offset,m_Views[level].width );
            h_offset = std::min(h_offset,m_Views[level].height);
            w_size   = std::min(w_offset+w_size,m_Views[level].width ) - w_offset;
            h_size   = std::min(h_offset+h_size,m_Views[level].height) - h_offset;
            if(isBinded){
                this->bind();
            }
            constexpr size_t pixelSizeInBytes = sizeof(GLPixelTraits<PixelType>::base_type)*GLPixelTraits<PixelType>::numChannels;
            size_t unpackAlignments = 1;
            if(pixelSizeInBytes%8==0){
                unpackAlignments = 8;
            }else if(pixelSizeInBytes%4==0){
                unpackAlignments = 4;
            }else if(pixelSizeInBytes%2==0){
                unpackAlignments = 2;
            }
            glPixelStorei(GL_UNPACK_ALIGNMENT,unpackAlignments);
            glTexSubImage2D(m_Target,level,w_offset,h_offset,w_size,h_size,GLPixelTraits<PixelType>::format,GLPixelTraits<PixelType>::type,data);
            return true;
        }
        bool upload(  size_t level, const GLBuffer<PixelType>&        buffer, size_t w_offset, size_t h_offset, size_t w_size, size_t h_size, bool isBinded = true) {
            if (level >= m_Views.size()) {
                return false;
            }
            w_offset = std::min(w_offset, m_Views[level].width);
            h_offset = std::min(h_offset, m_Views[level].height);
            w_size = std::min(w_offset + w_size, m_Views[level].width) - w_offset;
            h_size = std::min(h_offset + h_size, m_Views[level].height) - h_offset;
            if (isBinded) {
                this->bind();
            }
            constexpr size_t pixelSizeInBytes = sizeof(GLPixelTraits<PixelType>::base_type) * GLPixelTraits<PixelType>::numChannels;
            size_t unpackAlignments = 1;
            if (pixelSizeInBytes % 8 == 0) {
                unpackAlignments = 8;
            }
            else if (pixelSizeInBytes % 4 == 0) {
                unpackAlignments = 4;
            }
            else if (pixelSizeInBytes % 2 == 0) {
                unpackAlignments = 2;
            }
            glPixelStorei(GL_UNPACK_ALIGNMENT, unpackAlignments);
            buffer.bind();
            glTexSubImage2D(m_Target, level, w_offset, h_offset, w_size, h_size, GLPixelTraits<PixelType>::format, GLPixelTraits<PixelType>::type, nullptr);
            return true;
        }
        //download
        bool download(size_t level,       void* data, bool isBinded = true){
            if(level>=m_Views.size()){
                return false;
            }
            if(isBinded){
                this->bind();
            }
            glGetTexImage(m_Target,level,GLPixelTraits<PixelType>::format,GLPixelTraits<PixelType>::type,data);
            return true;
        }
        //Parameteri
        void setParameteri(GLenum pname, GLint param, bool isBinded=true){
            if(isBinded){
                this->bind();
            }
            glTexParameteri(m_Target,pname,param);
        }
        //Mipmap
        void generateMipmaps(bool isBinded = true) {
            if (isBinded) {
                this->bind();
            }
            glGenerateMipmap(m_Target);
        }
        void bind()const{
            glBindTexture(m_Target,m_ID);
        }
        void unbind()const{
            glBindTexture(m_Target,0);
        }
        void reset(){
            if(m_ID!=0){
                glDeleteTextures(1,&m_ID);
                m_ID     = 0;
                m_Views.clear();
            }
        }
        //Image
        void bindImage(GLuint unit,GLint level,GLenum access){
            glBindImageTexture(unit,m_ID,level,GL_FALSE,0,access, GLPixelTraits<PixelType>::format);
        }
        ~GLTexture2D(){
            this->reset();
        }
    };
    //   GLVertexShader
    class GLVertexShader{
        GLuint m_ID = 0;
    public:
        GLVertexShader()noexcept{}
        GLVertexShader(const GLVertexShader&) = delete;
        GLVertexShader(GLVertexShader&& vs)noexcept{
            m_ID = vs.m_ID;
            vs.m_ID = 0;
        }
        GLVertexShader& operator=(const GLVertexShader& vs)noexcept = delete;
        GLVertexShader& operator=(GLVertexShader&&      vs){
            if(this!=&vs){
                this->destroy();
                m_ID = vs.m_ID;
                vs.m_ID = 0;
            }
            return *this;
        }
        explicit GLVertexShader(const std::string& source) :GLVertexShader(){
            this->create(source);
        }
        explicit operator bool()const noexcept{
            return this->m_ID!=0;
        }
        RTLIB_DECLARE_GET_BY_VALUE(GLVertexShader,GLuint,ID,m_ID);
        void create (const std::string& source){
            const char* p_Src = source.data();
            this->m_ID = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(m_ID,1,&p_Src,nullptr);
        }
        void destroy() {
            if (this->m_ID > 0) {
                glDeleteShader(m_ID);
                this->m_ID = 0;
            }
        }
        bool compile()const noexcept{
            glCompileShader(m_ID);
            GLint res;
            glGetShaderiv(m_ID,GL_COMPILE_STATUS,&res);
            return res==GL_TRUE;
        }
        std::string getLog()const noexcept{
            GLint len;
            glGetShaderiv(m_ID, GL_INFO_LOG_LENGTH,&len);
            std::string log = {};
            log.resize(len + 1,'\0');
            glGetShaderInfoLog(m_ID,len,nullptr,log.data());
            log.resize(len);
            return log;
        }
        void attach(GLuint programID)const noexcept{
            glAttachShader(programID,m_ID);
        }
        ~GLVertexShader()noexcept{
            this->destroy();
        }
    };
    //   GLFragmentShader
    class GLFragmentShader{
        GLuint m_ID = 0;
    public:
        GLFragmentShader()noexcept{}
        GLFragmentShader(const GLFragmentShader&) = delete;
        GLFragmentShader(GLFragmentShader&& fs)noexcept{
            m_ID = fs.m_ID;
            fs.m_ID = 0;
        }
        GLFragmentShader& operator=(const GLFragmentShader& fs)noexcept = delete;
        GLFragmentShader& operator=(GLFragmentShader&&      fs){
            if(this!=&fs){
                this->destroy();
                m_ID = fs.m_ID;
                fs.m_ID = 0;
            }
            return *this;
        }
        explicit GLFragmentShader(const std::string& source) :GLFragmentShader() {
            this->create(source);
        }
        explicit operator bool()const noexcept{
            return this->m_ID!=0;
        }
        RTLIB_DECLARE_GET_BY_VALUE(GLFragmentShader,GLuint,ID,m_ID);
        void create (const std::string& source){
            const char* p_Src = source.data();
            this->m_ID = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(m_ID,1,&p_Src,nullptr);
        }
        void destroy() {
            if (this->m_ID > 0) {
                glDeleteShader(m_ID);
                this->m_ID = 0;
            }
        }
        bool compile()const noexcept{
            glCompileShader(m_ID);
            GLint res;
            glGetShaderiv(m_ID, GL_COMPILE_STATUS,&res);
            return res==GL_TRUE;
        }
        std::string getLog()const noexcept{
            GLint len;
            glGetShaderiv(m_ID,GL_INFO_LOG_LENGTH,&len);
            std::string log = {};
            log.resize(len+1,'\0');
            glGetShaderInfoLog(m_ID,len,nullptr,log.data());
            log.resize(len);
            return log;
        }
        void attach(GLuint programID)const noexcept{
            glAttachShader(programID,m_ID);
        }
        ~GLFragmentShader()noexcept{
            this->destroy();
        }
    };
    //   GLComputeShader
    class GLComputeShader{
        GLuint m_ID = 0;
    public:
        GLComputeShader()noexcept{}
        GLComputeShader(const GLComputeShader&) = delete;
        GLComputeShader(GLComputeShader&& cs)noexcept{
            m_ID = cs.m_ID;
            cs.m_ID = 0;
        }
        GLComputeShader& operator=(const GLComputeShader& cs)noexcept = delete;
        GLComputeShader& operator=(GLComputeShader&&      cs){
            if(this!=&cs){
                this->destroy();
                m_ID = cs.m_ID;
                cs.m_ID = 0;
            }
            return *this;
        }
        explicit GLComputeShader(const std::string& source) :GLComputeShader() {
            this->create(source);
        }
        explicit operator bool()const noexcept{
            return this->m_ID!=0;
        }
        RTLIB_DECLARE_GET_BY_VALUE(GLComputeShader,GLuint,ID,m_ID);
        void create (const std::string& source){
            const char* p_Src = source.data();
            this->m_ID = glCreateShader(GL_COMPUTE_SHADER);
            glShaderSource(m_ID,1,&p_Src,nullptr);
        }
        void destroy() {
            if (this->m_ID > 0) {
                glDeleteShader(m_ID);
                this->m_ID = 0;
            }
        }
        bool compile()const noexcept{
            glCompileShader(m_ID);
            GLint res;
            glGetShaderiv(m_ID, GL_COMPILE_STATUS,&res);
            return res==GL_TRUE;
        }
        std::string getLog()const noexcept{
            GLint len;
            glGetShaderiv(m_ID,GL_INFO_LOG_LENGTH,&len);
            std::string log = {};
            log.resize(len+1,'\0');
            glGetShaderInfoLog(m_ID,len,nullptr,log.data());
            log.resize(len);
            return log;
        }
        void attach(GLuint programID)const noexcept{
            glAttachShader(programID,m_ID);
        }
        ~GLComputeShader()noexcept{
            this->destroy();
        }
    };
    //   GLProgram
    class GLProgram {
        GLuint m_ID = 0;
    public:
        GLProgram()noexcept{}
        GLProgram(const GLProgram&) = delete;
        GLProgram(GLProgram&& prog)noexcept {
            m_ID = prog.m_ID;
            prog.m_ID = 0;
        }
        GLProgram& operator=(const GLProgram& prog)noexcept = delete;
        GLProgram& operator=(GLProgram&& prog) {
            if (this != &prog) {
                this->destroy();
                m_ID = prog.m_ID;
                prog.m_ID = 0;
            }
            return *this;
        }
        explicit operator bool()const noexcept {
            return this->m_ID != 0;
        }
        RTLIB_DECLARE_GET_BY_VALUE(GLProgram, GLuint, ID, m_ID);
        auto getComputeWorkGroupSize()const->std::array<GLint, 3>{
            std::array<GLint, 3> work_group_size;
            glGetProgramiv(m_ID, GL_COMPUTE_WORK_GROUP_SIZE, work_group_size.data());
            return work_group_size;
        }
        void create() {
            this->m_ID = glCreateProgram();
        }
        void destroy() {
            if (this->m_ID > 0) {
                glDeleteProgram(m_ID);
                this->m_ID = 0;
            }
        }
        void attach(const GLVertexShader  & vs)const noexcept {
            vs.attach(m_ID);
        }
        void attach(const GLFragmentShader& fs)const noexcept {
            fs.attach(m_ID);
        }
        void attach(const GLComputeShader& cs)const noexcept {
            cs.attach(m_ID);
        }
        
        bool link()const noexcept {
            glLinkProgram(m_ID);
            GLint res;
            glGetProgramiv(m_ID, GL_LINK_STATUS, &res);
            return res == GL_TRUE;
        }
        std::string getLog()const noexcept {
            GLint len;
            glGetProgramiv(m_ID, GL_INFO_LOG_LENGTH, &len);
            std::string log = {};
            log.resize(len + 1, '\0');
            glGetProgramInfoLog(m_ID, len, nullptr, log.data());
            log.resize(len);
            return log;
        }
        void use()const noexcept {
            glUseProgram(m_ID);
        }
        GLint getUniformLocation(const std::string& name)const noexcept {
            return glGetUniformLocation(m_ID, name.data());
        }
        ~GLProgram()noexcept {
            this->destroy();
        }
    };
}
#endif

