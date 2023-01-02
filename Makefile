clean: clean-pyc clean-test
quality: set-style-dep check-quality
style: set-style-dep set-style
setup: set-precommit set-style-dep set-test-dep set-git set-dev set-dataset
test: set-test-dep set-test
coverage: check-coverage
profile: check-profile


##### basic #####
set-git:
	git config --local commit.template .gitmessage
	git config --local core.editor "code --wait"

set-style-dep:
	pip3 install isort==5.10.1 black==22.3.0 flake8==4.0.1

set-test-dep:
	pip3 install pytest==7.0.1

set-precommit:
	pip3 install pre-commit==2.17.0
	pre-commit install

set-dev:
	pip3 install -r requirements.txt

set-test:
	python3 -m pytest tests/

set-style:
	black --config pyproject.toml .
	isort --settings-path pyproject.toml .
	flake8 .

check-quality:
	black --config pyproject.toml --check .
	isort --settings-path pyproject.toml --check-only .
	flake8 .

#####  clean  #####
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache

##### for competition #####
set-dataset:
	wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000228/data/data.tar.gz
	tar -xf data.tar.gz
	rm data.tar.gz

set-directory:
	mkdir -p ./src/prediction
	mkdir -p ./src/logs
	mkdir -p ./src/best_model
	mkdir -p ./src/results

check-coverage:
	coverage run -m unittest
	coverage report -m

check-profile:
	python tests/profile_retrieval.py
