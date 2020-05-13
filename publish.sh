rm -rf build dist fitloop.egg-info
python setup.py sdist bdist_wheel 
twine upload dist/*