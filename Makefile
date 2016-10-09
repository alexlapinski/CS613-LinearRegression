init:
	pip install -r requirements.txt

clean:
	rm -f graphs/**/*.png tex/*.aux tex/*.log tex/*.synctex.gz

.PHONY: init test run clean
