#ifndef LOBE_TYPE_H
#define LOBE_TYPE_H
enum LobeType
{
    //Reflection: Diffuse
    LOBE_TYPE_REFLECTION_DIFFUSE_LAMBERT   = (1<<0),
    LOBE_TYPE_REFLECTION_DIFFUSE_DISNEY    = (1<<1),
    //Reflection: Specular
    LOBE_TYPE_REFLECTION_SPECULAR_DELTA    = (1<<2),
    LOBE_TYPE_REFLECTION_SPECULAR_BECKMANN = (1<<3),
    LOBE_TYPE_REFLECTION_SPECULAR_GGX      = (1<<4),
    //Refraction:
    LOBE_TYPE_REFRACTION_DELTA             = (1<<5)
}
#endif
