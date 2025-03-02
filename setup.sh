# Check which platform is being used and source the correct cvmfs script
source /etc/os-release
if [[ "$PLATFORM_ID" == "platform:el8" ]]; then
    source /cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el8-gcc11-opt/setup.sh
elif [[ "$PLATFORM_ID" == "platform:el9" ]]; then
    source /cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el9-gcc11-opt/setup.sh
else
    echo "Unsupported platform: $PLATFORM_ID. You must use el8 or el9 (preferred)"
    return 1
fi
source .venv/bin/activate
