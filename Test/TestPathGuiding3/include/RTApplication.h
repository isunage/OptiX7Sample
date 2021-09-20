#ifndef TEST_RT_APPLICATION_H
#define TEST_RT_APPLICATION_H
#include <range/v3/view.hpp>
#include <memory>
#include <vector>
namespace test{
    class RTAppEvent
    {
    public:
        //Initialize
        virtual void OnInit() {}
        //Loop
        virtual bool OnQuitFrame() { return false; }
        virtual void OnEachFrame() {}
        //CleanUp
        virtual void OnDestroy() {}
        virtual ~RTAppEvent(){}
    };
    using RTAppEventPtr = std::shared_ptr<RTAppEvent>;
    class RTApplication
    {
    public:
        //Initialize
        virtual void Initialize() {
            for(auto& event:m_Events){
                if(event){
                    event->OnInit();
                }
            }
        }
        //MainLoop
        virtual void MainLoop()     {
            while(!QuitLoop()){
                for(auto& event:m_Events){
                    if(event&&!event->OnQuitFrame()){
                        event->OnEachFrame();
                    }
                }
            }
        }
        //QuitLoop
        virtual bool QuitLoop()   {
            return true;
        }
        //CleanUp
        virtual void CleanUp()    {
            for(auto& event: m_Events | ranges::v3::view::reverse){
                if(event){
                    event->OnDestroy();
                    event.reset();
                }
            }
        }
        void         AttachEvent(const RTAppEventPtr& pEvent) {
            m_Events.push_back(pEvent);
        }
        virtual ~RTApplication(){}
    private:
        std::vector<RTAppEventPtr> m_Events = {};
    };
    
}
#endif