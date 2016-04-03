.PHONY: all

all: density.png

density.pdf: density.tex
	pdflatex density.tex

density.png: density.pdf
	convert -density 300 density.pdf -quality 90 density.png
