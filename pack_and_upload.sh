rm -rf dist/ build/ confignest.egg-info/
python setup.py sdist bdist_wheel 
twine upload dist/*