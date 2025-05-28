echo ""
echo "########## Clean Up Workspace ##########"
echo ""
rm -r AsyncDiff DistriFuser xDiT backend_pb2.py backend_pb2_grpc.py



echo ""
echo "########## Clone AsyncDiff ##########"
echo ""
bash scripts/clone_asyncdiff_repo.sh

echo ""
echo "########## Patching AsyncDiff ##########"
echo ""
python3 scripts/PatchAsyncDiff.py



echo ""
echo "########## Clone DistriFuser ##########"
echo ""
bash scripts/clone_distrifuser_repo.sh

echo ""
echo "########## Patching DistriFuser ##########"
echo ""
python3 scripts/PatchDistriFuser.py



echo ""
echo "########## Clone xDiT ##########"
echo ""
bash scripts/clone_xdit_repo.sh

echo ""
echo "########## Patching xDiT ##########"
echo ""
python3 scripts/PatchxDiT.py



echo ""
echo "########## Make Backend ##########"
echo ""
make protogen
