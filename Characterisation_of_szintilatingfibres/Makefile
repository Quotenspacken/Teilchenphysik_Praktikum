all: build/main.pdf

# hier Python-Skripte:
build/SimData.pdf: SimData.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python SimData.py
build/hist2d_clad.pdf: SimData.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python SimData.py
build/hist2d_clad.pdf: SimData.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python SimData.py
build/sim.pdf: SimData.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python SimData.py

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/main.pdf: build/SimData.pdf
build/main.pdf: build/hist2d_core.pdf
build/main.pdf: build/hist2d_clad.pdf
build/main.pdf: build/sim.pdf

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
