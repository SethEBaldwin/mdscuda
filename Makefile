deploy:
	python3 -m build
	python3 -m twine upload --skip-existing --repository pypi dist/*

clean:
	rm -r build
	rm -r dist
	rm -r mdscuda.egg-info
	rm -r ./mdscuda/__pycache__
	rm -r __pycache__
