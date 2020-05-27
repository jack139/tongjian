PY = python3.5 -O -m compileall -b -q -f
SRC = jiebazhc utils predict.py train.py
TARGETS = build

all: clean $(TARGETS)

$(TARGETS):
	@echo "Compiling ..."
	@for target in $(SRC) ; do \
		if [ -d "$$target" ]; then \
			mkdir -p $(TARGETS)/$$target ; \
			cp -r $$target/* $(TARGETS)/$$target/ ; \
		else \
			cp $$target $(TARGETS)/ ; \
		fi ; \
	done
	-$(PY) $(TARGETS)
	@find $(TARGETS) -name '*.py' -delete
	@find $(TARGETS) -name "__pycache__" |xargs rm -rf

clean:
	@echo "Clean ..." 
	@find . -name "__pycache__" |xargs rm -rf
	@find . -name '*.pyc' -delete
	@rm -rf $(TARGETS)
