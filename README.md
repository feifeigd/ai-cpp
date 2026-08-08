# ai-cpp
ai-cpp

## 生成 vcpkg.json + vcpkg-configuration.json
```shell
vcpkg new --application
vcpkg add port spdlog	
vcpkg format-manifest 
vcpkg install
```

## 更新 baseline
vcpkg x-update-baseline

## 配置
cmake -B build -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

## 配置
cmake --preset windows-x64
## 构建
cmake --build --preset windows-x64-debug
cmake --build --preset windows-x64-release
