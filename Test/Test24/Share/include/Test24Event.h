#ifndef TEST_TEST24_EVENT_H
#define TEST_TEST24_EVENT_H
enum Test24EventFlag: unsigned int 
{
    TEST24_EVENT_FLAG_NONE          = 0b000000,
    TEST24_EVENT_FLAG_FLUSH_FRAME   = 0b000001,
    TEST24_EVENT_FLAG_RESIZE_FRAME  = 0b001011,
    TEST24_EVENT_FLAG_CHANGE_TRACE  = 0b000101,
    TEST24_EVENT_FLAG_UPDATE_CAMERA = 0b001001,
    TEST24_EVENT_FLAG_UPDATE_LIGHT  = 0b010001,
    TEST24_EVENT_FLAG_LOCK          = 0b100000,
};
#endif