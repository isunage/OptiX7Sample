#include <TestLibConfig.h>
#include <RTObjMesh.h>
#include <memory>
int main() {
	auto objMeshReader		  = std::make_shared<test::RTObjMeshReader>();
	auto inputJsonData		  = nlohmann::json();
	inputJsonData["type"    ] = "shape";
	inputJsonData["plugin"  ] = "objModel";
	inputJsonData["filename"] = TEST_TESTLIB_DATA_PATH"\\Models\\Sponza\\Sponza.obj";
	inputJsonData["meshname"] = "sponza_00";
	auto objMesh			  = objMeshReader->LoadJsonData(inputJsonData);
	return 0;
}