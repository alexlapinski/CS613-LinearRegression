CS613 - HW1 - Alex Lapinski

In order to run the code contained within this homework submission first make sure the following are available.
 * Python 2.7 (but not python 3)
 * Windowing System (Matplotlib doesn't always work well with python virtual environments)
 * make

Then run the following commands using Make:

To setup & install dependencies run one of the following:
    * make
    * pip install -r requirements.txt

To run the PCA example run one of the following (from the hw1 directory):
    * make pca
    * python src/hw1.py --pca --data ./diabetes.csv

To run the KMeans example run one of the following (from the hw1 directory):
    * make kmeans
    * python src/hw1.py --kmeans --data ./diabetes.csv


I've included a help feature of the hw1.py module, just run "python src/hw1.py -h".