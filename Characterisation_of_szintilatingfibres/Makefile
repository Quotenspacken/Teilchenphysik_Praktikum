all: build/main.pdf

# hier Python-Skripte:
build/SimData.pdf: SimData.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python SimData.py
build/hist2d_core_clad.pdf: SimData.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python SimData.py
build/atten.pdf: SimData.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python SimData.py
build/radial.pdf: radial.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python radial.py
build/intensity.pdf: intensity.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python intensity.py
build/spectra.pdf: spect.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python spect.py
build/spectra_dc.pdf: spect.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python spect.py


# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/main.pdf: build/SimData.pdf
build/main.pdf: build/hist2d_core_clad.pdf
build/main.pdf: build/radial.pdf
build/main.pdf: build/intensity.pdf
build/main.pdf: build/spectra.pdf
build/main.pdf: build/spectra_dc.pdf
build/main.pdf: build/atten.pdf

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
