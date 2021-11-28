Class 設計について
各クラスはあくまでパース上で必要なもので、実際にどのように処理するかは
未定義

Sphere の場合
Sphere--Triangule-->Mesh Input->Default Trace/Default Rasterizer
Gen AABB ->AABB Input-> Custom Trace/ Custom Rasterizer
Cube   の場合
  Cube--Triangule-->Mesh Input->Default Trace/Default Rasterizer
ObjMeshの場合
Mesh Input-->Default Trace/Default Rasterizer

Material
ObjMtlの場合
ObjMtl-->ObjMtlConverter(Phong/Diffuse/DeltaReflection/DeltaDielectric)->Custom DC

SceneGraphの作成

SceneGraphにはそれぞれInstancingGraph, GraphArray, ShapeArrayの3つが含まれる
これらはAcceleration Structure作成時のヒントとなる
Instancing -> Instancingを試みる
ShapeArray -> BLAS作成  を試みる
GraphArray -> TLAS作成　を試みる

実際にどのようなAcceleration Structureが作成されるかは定義せず、最適化の余地を残す

Scene.json 
-> Scene_RT.json
-> Scene_RP.json

