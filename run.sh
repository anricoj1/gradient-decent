sec=15
secs=15

function countdown() {
  while [ $sec -ge 0 ]; do
    echo -ne "Seconds Till Start: $sec\033[0K\r"
    let "sec=sec-1"
    sleep 1
  done
}

function newcountdown() {
  while [ $secs -ge 0 ]; do
    echo -ne "Seconds Till Start: $secs\033[0K\r"
    let "secs=secs-1"
    sleep 1
  done
}


function batch_gradient_descent() {
	echo "Starting Batch Gradient Descent In..";
	countdown;
	python3 batch.py;
}

function stochastic_gradient() {
	echo "Staring Batch Gradient Descent In..";
	newcountdown;
	python3 stoch.py
}

batch_gradient_descent

stochastic_gradient

echo
