aggr="import sys,numpy as np; \
	a=np.array(list(map(float,sys.stdin.readlines()))); \
	b=a.shape[0]; print(a.mean(),a.std()/(b**0.5),b)"
echo $aggr
logdir=$1
for dat in boston concrete kin8mn naval power protein winered yacht energy
do
	echo $dat
	tail -n 1 ${logdir}*${dat}*/stdout | grep rmse | \
		awk '{print substr($5,1,length($5)-1)}' | python -c "$aggr"
	tail -n 1 ${logdir}*${dat}*/stdout | grep rmse | \
		awk '{print $8}' | python -c "$aggr"
done
