# OptiX7Sample
NVIDIA OptiX7.3を用いたpath tracing実装
# 前提ライブラリ
NVIDIA OptiX 7.3 SDK
NVIDIA CUDA ToolKit
# Data
ルートディレクトリ上に以下のディレクトリとファイルを作成
Data/
Data/Textures/
Data/Textures/white.png
Data/Textures/sample.png
Data/Models/
Data/Models/CornellBox/
Data/Models/Sponza/
Data/Textures/white.pngは512x512x4の白色画像を配置
Data/Textures/sample.pngは任意のサンプル画像を配置
Data/Models/*/にはhttps://casual-effects.com/data/からそれぞれCrytek Sponza、CornellBoxをダウンロードして配置
# Build 
ルートディレクトリ上でbuildディレクトリを作成し、そのディレクトリで cmake ../ -G "Visual Studio 16 2019"を実行