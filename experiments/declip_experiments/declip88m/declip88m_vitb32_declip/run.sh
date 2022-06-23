mkdir -p petrel.log
PYTHONPATH=$PYTHONPATH:../../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --mpi=pmi2 -n$1 --gpu  --cpus-per-task=6 --job-name $2 $3 $4 $5 $6 \
"python -u -m prototype.solver.declip_solver --config config.yaml --evaluate"
# "python -u -m prototype.solver.clip_solver_simsiam_2_view_MLM --config config.yaml --evaluate"
# --evaluate
#-x SH-IDC1-10-5-36-[101]
