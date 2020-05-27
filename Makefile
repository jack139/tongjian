PY = python3.6 -O -m compileall -b -q -f
SRC = jiebazhc jiebazhc2
TARGETS = build

all: clean $(TARGETS)

$(TARGETS):
	@echo "Compiling ..."
	@for target in $(SRC) ; do \
		mkdir -p $(TARGETS)/$$target ; \
		cp -r $$target/* $(TARGETS)/$$target/ ; \
	done
	-$(PY) $(TARGETS)
	@find $(TARGETS) -name '*.py' -delete
	@find $(TARGETS) -name "__pycache__" |xargs rm -rf

clean:
	@echo "Clean ..." 
	@find . -name "__pycache__" |xargs rm -rf
	@find . -name '*.pyc' -delete
	@rm -rf $(TARGETS)
