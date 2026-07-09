all: build/main.pdf

# hier Python-Skripte:
build/IV.pdf: IV.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python IV.py
build/pedestal.pdf: pedestal.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python pedestal.py
build/calibration.pdf: calibration.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python calibration.py
build/calibration_fix.pdf: calibration.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python calibration.py
build/calibration_0.pdf: calibration.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python calibration.py
build/laser.pdf: laser.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python laser.py
# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/main.pdf: build/IV.pdf
build/main.pdf: build/pedestal.pdf
build/main.pdf: build/calibration.pdf
build/main.pdf: build/calibration_fix.pdf
build/main.pdf: build/calibration_0.pdf
build/main.pdf: build/laser.pdf

build/main.pdf: FORCE | build
	  TEXINPUTS=build: \
	  BIBINPUTS=build: \
	  max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	main.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
