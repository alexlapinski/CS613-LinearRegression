init:
	pip install -r requirements.txt

clean:
	rm -f graphs/**/*.png tex/*.aux tex/*.log tex/*.synctex.gz

part2:
	python src/hw2.py -c

part3:
	python src/hw2.py -s

part4:
	python src/hw2.py -l

part5:
	python src/hw2.py -g

all:
	python src/hw2.py -c -s -l -g

.PHONY: init test run clean
