set -e

resdir=`mktemp -d`
logdir=${1%%/}

for i in `ls $logdir`; do
	if [[ $i == _* ]]; then
		echo $logdir/$i
		cat $logdir/$i/stdout|grep umulative|sed '{s/Uniform Sampling/UniformSampling/}'>$resdir/$i.out
	fi
done

ls $resdir/* -d | python gather-result.py

# rm -rf $resdir
