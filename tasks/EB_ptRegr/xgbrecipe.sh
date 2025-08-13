cd utils
git clone --recursive https://github.com/dmlc/xgboost.git
mkdir -p xgboost/plugin/L1Loss
cp l1loss.cc xgboost/plugin/L1Loss/l1loss.cc
echo "target_sources(objxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/L1Loss/l1loss.cc)" >> xgboost/plugin/CMakeLists.txt
cd xgboost
mkdir build
cd build
cmake .. -DUSE_PLUGIN=ON
make -j$(nproc)
cd ../..