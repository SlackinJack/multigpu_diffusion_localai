echo ""
echo "########## Clean Up Workspace ##########"
echo ""
rm -r multigpu_diffusion



echo ""
echo "########## Clone multigpu_diffusion ##########"
echo ""
git clone https://github.com/slackinjack/multigpu_diffusion



echo ""
echo "########## Setup multigpu_diffusion ##########"
echo ""
cd multigpu_diffusion
bash setup.sh
cd ..



echo ""
echo "########## Make Backend ##########"
echo ""
make protogen

