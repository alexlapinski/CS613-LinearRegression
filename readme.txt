CS613 - HW2 - Alex Lapinski

In order to run the code contained within this homework submission first make sure the following are available.
 * Python 2.7 (but not python 3)
 * Windowing System (Matplotlib doesn't always work well with python virtual environments)
 * make

Then run the following commands using Make:

To setup & install dependencies run one of the following:
    * make
    * pip install -r requirements.txt

To run the Closed Form Linear Regression example run one of the following (from the hw2 directory):
    * make part2
    * python src/hw2.py --cflr --data ./x06Simple.csv

To run the S-Folds example run one of the following (from the hw2 directory):
    * make part3
    * python src/hw2.py --s-folds --data ./x06Simple.csv
    * python src/hw2/py --s-folds --num-folds 7 --data ./x06Simple.csv

To run the Locally Weighted example run one of the following (from the hw2 directory):
    * make part4
    * python src/hw2.py --lwlr --data ./x06Simple.csv
    * python src/hw2/py --lwlr --k-value 0.66 --data ./x06Simple.csv

To run the Gradient Descent example run one of the following (from the hw2 directory):
    * make part5
    * python src/hw2.py --gradient --data ./x06Simple.csv
    * python src/hw2/py --gradient --learning-rate 0.1 --data ./x06Simple.csv


I've included a help feature of the hw2.py module, just run "python src/hw2.py -h".
Or to run all parts just run "make all", and above are optional parameters for testing the tuning parameters of each
algorithm.