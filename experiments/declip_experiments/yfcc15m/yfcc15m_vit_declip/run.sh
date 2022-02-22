mkdir -p petrel.log
PYTHONPATH=../../../../:$PYTHONPATH GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --mpi=pmi2 -n$1  --gpu --cpus-per-task=6 --job-name=$2 $3 $4 $5 $6 \
"python -u -m prototype.solver.declip_solver --config config.yaml"
# -x SH-IDC1-10-198-34-[65,67-71,73,101,41,43,47,50,56,74,94,146,149,151,154,156] \
# -p GVM \
#  --evaluate
#-x SH-IDC1-10-5-36-[101] 
