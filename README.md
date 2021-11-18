# OptiX7Sample
NVIDIA OptiX7.3を用いたpath tracing実装
# 前提ライブラリ
- NVIDIA OptiX 7 SDK  ( 7.3で動作確認、おそらく7.0, 7.1, 7.2は動作)
- NVIDIA CUDA ToolKit (11.0で動作確認、おそらく10.0以降であれば動作)
# Data
ルートディレクトリ上に以下のディレクトリとファイルを作成
- Data/
- Data/Textures/
- Data/Textures/white.png
- Data/Textures/sample.png
- Data/Models/
- Data/Models/CornellBox/
- Data/Models/Sponza/
- Data/Textures/white.pngは512x512x4の白色画像を配置
- Data/Textures/sample.pngは任意のサンプル画像を配置
- Data/Models/*/にはhttps://casual-effects.com/data/ からそれぞれCrytek Sponza、CornellBoxをダウンロードして配置
# Install
レポジトリのクローン後、git submodule init && git submodule updateを実行 
# Build 
- Windows OS, CMake Version 3.19 以上, Visual Studio 2019を前提
- ルートディレクトリ上でbuildディレクトリを作成し、そのディレクトリで cmake ../ -G "Visual Studio 16 2019"を実行
# Tips
- Bound Valueによる最適化は、Module内で多数のレイ呼び出しを行う場合かえって低速化する可能性がある
(OptiX 7.4のSampleでは1,2 sppでは高速化したが、サンプル数が増えるにつれて若干低速化している)